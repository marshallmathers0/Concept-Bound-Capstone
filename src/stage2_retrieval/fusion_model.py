"""
Stage 2 — Segment Fusion Encoder and Query Projector.

These are the two trainable modules of Stage 2:

  - SegmentFusionEncoder: takes the three frozen modality embeddings of a
    concept-coherent segment
        e_ocr        (Sentence-BERT embedding of aggregated OCR text)
        e_transcript (Sentence-BERT embedding of aggregated ASR transcript)
        e_visual     (CLIP embedding of representative keyframe(s))
    projects each to a common hidden dimension, treats them as a 3-token
    sequence, applies multi-head self/cross-attention across the tokens, and
    mean-pools the result into a single segment representation z_seg.

  - QueryProjector: a small MLP that maps a frozen Sentence-BERT query
    embedding into the same space as z_seg, so that
        score(query, segment) = cos_sim(QueryProjector(e_query), z_seg)
    Both modules are trained jointly with an InfoNCE contrastive loss
    (see `train.py`).

Fusion modes (the fusion ablation):
  - "cross_attention": the three modality tokens attend to each other in one
    transformer block, mean-pooled (default).
  - "concat": projections are concatenated, no attention.
  - "pairwise": adapted from Arora et al. (LREC-COLING 2024), text-to-
    multimodal retrieval with bimodal fusion in a shared cross-modal
    transformer. Each modality first passes through its own unimodal expert;
    ONE shared transformer block then fuses every modality pair
    (OCR+transcript, transcript+visual, OCR+visual), and the segment embedding
    is pooled from the three pair representations. `forward_with_parts` also
    returns each pair representation so training can add a contrastive loss
    per pair, as in their pairwise loss combinations.
"""

import torch
import torch.nn as nn


class ZScoreNorm(nn.Module):
    """Z-score normalization across the batch dimension.

    Normalizes each modality embedding to zero mean and unit variance before
    fusion. This prevents dominant modalities (e.g., text features with larger
    magnitudes) from overpowering weaker modalities (e.g., visual/audio vectors)
    in the cross-attention fusion step.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(0) <= 1:
            # Can't compute meaningful statistics for batch size 1
            return x
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True) + 1e-8
        return (x - mean) / std


class SegmentFusionEncoder(nn.Module):
    def __init__(self, ocr_dim: int, transcript_dim: int, visual_dim: int,
                 hidden_dim: int = 384, out_dim: int = 384,
                 num_heads: int = 4, dropout: float = 0.1,
                 fusion_mode: str = "cross_attention"):
        super().__init__()
        self.fusion_mode = fusion_mode

        self.ocr_proj = nn.Linear(ocr_dim, hidden_dim)
        self.transcript_proj = nn.Linear(transcript_dim, hidden_dim)
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)

        # Z-score normalization applied after projection, before fusion.
        # This calibrates all modalities to comparable magnitudes so that
        # no single modality dominates the cross-attention scores.
        self.zscore_norm = ZScoreNorm()

        # Learned modality-type embeddings (added so attention can
        # distinguish the OCR/transcript/visual tokens).
        self.modality_embed = nn.Parameter(torch.randn(3, hidden_dim) * 0.02)

        if fusion_mode in ("cross_attention", "pairwise"):
            # In "pairwise" mode this block is the shared cross-modal fusion
            # transformer, applied to each modality pair with the same weights.
            self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            self.norm2 = nn.LayerNorm(hidden_dim)
            fused_dim = hidden_dim
            if fusion_mode == "pairwise":
                self.experts = nn.ModuleList([
                    nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
                    for _ in range(3)
                ])
        elif fusion_mode == "concat":
            fused_dim = hidden_dim * 3
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

        self.out_proj = nn.Sequential(
            nn.Linear(fused_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    PAIRS = {"ocr_transcript": (0, 1), "transcript_visual": (1, 2), "ocr_visual": (0, 2)}

    def _fuse_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)
        tokens = self.norm2(tokens + self.ffn(tokens))
        return tokens.mean(dim=1)

    def forward_with_parts(self, ocr_emb, transcript_emb, visual_emb):
        """Pairwise mode: (z_seg, {pair_name: z_pair}), all L2-normalized."""
        if self.fusion_mode != "pairwise":
            return self.forward(ocr_emb, transcript_emb, visual_emb), {}
        h = [self.zscore_norm(self.ocr_proj(ocr_emb)) + self.modality_embed[0],
             self.zscore_norm(self.transcript_proj(transcript_emb)) + self.modality_embed[1],
             self.zscore_norm(self.visual_proj(visual_emb)) + self.modality_embed[2]]
        h = [x + expert(x) for x, expert in zip(h, self.experts)]  # unimodal experts (residual)
        pair_h = {name: self._fuse_tokens(torch.stack([h[a], h[b]], dim=1))
                  for name, (a, b) in self.PAIRS.items()}
        z = nn.functional.normalize(self.out_proj(torch.stack(list(pair_h.values())).mean(0)), dim=-1)
        parts = {k: nn.functional.normalize(self.out_proj(v), dim=-1) for k, v in pair_h.items()}
        return z, parts

    def forward(self, ocr_emb: torch.Tensor, transcript_emb: torch.Tensor,
                visual_emb: torch.Tensor) -> torch.Tensor:
        """All inputs: (B, modality_dim) -> output: (B, out_dim), L2-normalized."""
        if self.fusion_mode == "pairwise":
            return self.forward_with_parts(ocr_emb, transcript_emb, visual_emb)[0]
        # Project each modality to shared hidden dimension
        ocr_h = self.ocr_proj(ocr_emb)
        trans_h = self.transcript_proj(transcript_emb)
        vis_h = self.visual_proj(visual_emb)

        # Z-score normalize to calibrate modality magnitudes
        ocr_h = self.zscore_norm(ocr_h) + self.modality_embed[0]
        trans_h = self.zscore_norm(trans_h) + self.modality_embed[1]
        vis_h = self.zscore_norm(vis_h) + self.modality_embed[2]

        if self.fusion_mode == "cross_attention":
            fused = self._fuse_tokens(torch.stack([ocr_h, trans_h, vis_h], dim=1))  # (B, H)
        else:  # concat
            fused = torch.cat([ocr_h, trans_h, vis_h], dim=-1)

        z = self.out_proj(fused)
        return nn.functional.normalize(z, dim=-1)


class QueryProjector(nn.Module):
    """Projects a frozen Sentence-BERT query embedding into the segment
    embedding space."""

    def __init__(self, in_dim: int, out_dim: int = 384, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, query_emb: torch.Tensor) -> torch.Tensor:
        z = self.net(query_emb)
        return nn.functional.normalize(z, dim=-1)


class CrossModalRetrievalModel(nn.Module):
    """Bundles the SegmentFusionEncoder and QueryProjector for convenient
    checkpointing and joint training."""

    def __init__(self, ocr_dim: int, transcript_dim: int, visual_dim: int, query_dim: int,
                 hidden_dim: int = 384, out_dim: int = 384,
                 num_heads: int = 4, dropout: float = 0.1,
                 fusion_mode: str = "cross_attention"):
        super().__init__()
        self.segment_encoder = SegmentFusionEncoder(
            ocr_dim, transcript_dim, visual_dim, hidden_dim, out_dim, num_heads, dropout, fusion_mode
        )
        self.query_projector = QueryProjector(query_dim, out_dim, dropout)

    def encode_segments(self, ocr_emb, transcript_emb, visual_emb):
        return self.segment_encoder(ocr_emb, transcript_emb, visual_emb)

    def encode_queries(self, query_emb):
        return self.query_projector(query_emb)

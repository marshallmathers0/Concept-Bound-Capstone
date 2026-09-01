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

An ablation switch (`fusion_mode="concat"`) is provided to reproduce the
"concatenation vs. cross-attention" ablation study described in the
proposal.
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

        if fusion_mode == "cross_attention":
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
        elif fusion_mode == "concat":
            fused_dim = hidden_dim * 3
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

        self.out_proj = nn.Sequential(
            nn.Linear(fused_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, ocr_emb: torch.Tensor, transcript_emb: torch.Tensor,
                visual_emb: torch.Tensor) -> torch.Tensor:
        """All inputs: (B, modality_dim) -> output: (B, out_dim), L2-normalized."""
        # Project each modality to shared hidden dimension
        ocr_h = self.ocr_proj(ocr_emb)
        trans_h = self.transcript_proj(transcript_emb)
        vis_h = self.visual_proj(visual_emb)

        # Z-score normalize to calibrate modality magnitudes
        ocr_h = self.zscore_norm(ocr_h) + self.modality_embed[0]
        trans_h = self.zscore_norm(trans_h) + self.modality_embed[1]
        vis_h = self.zscore_norm(vis_h) + self.modality_embed[2]

        if self.fusion_mode == "cross_attention":
            tokens = torch.stack([ocr_h, trans_h, vis_h], dim=1)  # (B, 3, H)
            attn_out, _ = self.attn(tokens, tokens, tokens)
            tokens = self.norm1(tokens + attn_out)
            tokens = self.norm2(tokens + self.ffn(tokens))
            fused = tokens.mean(dim=1)  # (B, H)
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

"""
Central configuration for the Concept-Aware Lecture Video Retrieval system.

All scripts import settings from here so that paths, model dimensions and
hyperparameters stay consistent across the feature-extraction, training,
indexing and serving stages.
"""

import os
from dataclasses import dataclass, field
from typing import List

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


@dataclass
class PathConfig:
    raw_videos_dir: str = os.path.join(PROJECT_ROOT, "data", "raw_videos")
    features_dir: str = os.path.join(PROJECT_ROOT, "data", "features")
    segments_dir: str = os.path.join(PROJECT_ROOT, "data", "segments")
    index_dir: str = os.path.join(PROJECT_ROOT, "data", "index")
    annotations_dir: str = os.path.join(PROJECT_ROOT, "data", "annotations")
    checkpoints_dir: str = os.path.join(PROJECT_ROOT, "checkpoints")


@dataclass
class FeatureConfig:
    # Time-step (in seconds) at which the multimodal signal stream is sampled.
    time_step_sec: float = 3.0

    # Dimensionality of the projected CLIP visual embedding used by Stage 1
    # and Stage 2. The pretrained checkpoints in this repo expect 32-d visual
    # features, so the raw 512-d CLIP output is reduced before it is stored.
    clip_dim: int = 32

    # Dimensionality of the Sentence-BERT text embedding (all-MiniLM-L6-v2 -> 384).
    text_embed_dim: int = 384

    # Number of "hand-crafted" scalar signals fused per time-step:
    #   (1) OCR text-change rate
    #   (2) ASR transcript topic-drift
    #   (3) visual-change rate (optical flow magnitude)
    num_scalar_signals: int = 3

    @property
    def stage1_input_dim(self) -> int:
        """Dimensionality of the per-time-step feature vector fed to Stage 1."""
        return self.num_scalar_signals + self.clip_dim


@dataclass
class Stage1Config:
    hidden_dim: int = 128
    embed_dim: int = 64
    encoder_type: str = "bilstm"  # "bilstm", "transformer", or "cnn"
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1

    # Training
    batch_size: int = 4
    epochs: int = 30
    lr: float = 1e-3
    margin: float = 0.5
    window_neg_samples: int = 4

    # Pseudo-label generation
    peak_smoothing_window: int = 3
    peak_min_distance_steps: int = 4
    peak_prominence: float = 0.15

    # Boundary post-processing
    boundary_threshold: float = 0.5  # used only if an absolute threshold is explicitly requested
    adaptive_threshold_std: float = 1.0  # default: mean + std * adaptive_threshold_std
    min_segment_len_steps: int = 3


@dataclass
class Stage2Config:
    fusion_hidden_dim: int = 384
    fusion_out_dim: int = 384
    num_heads: int = 4
    dropout: float = 0.1

    # Training
    batch_size: int = 16
    epochs: int = 20
    lr: float = 2e-4
    temperature: float = 0.07


@dataclass
class EvalConfig:
    boundary_tolerance_steps: int = 4   # +/- steps allowed (4 × 3s = ±12s ≈ ±10s tolerance)
    retrieval_ks: List[int] = field(default_factory=lambda: [1, 5, 10])
    retrieval_iou_threshold: float = 0.3
    retrieval_tolerance_sec: float = 10.0  # ±10s window for gold-standard evaluation


@dataclass
class ModelNames:
    sentence_bert: str = "sentence-transformers/all-MiniLM-L6-v2"
    clip_model: str = "openai/clip-vit-base-patch32"
    whisper_model: str = "base"
    ocr_engine: str = "easyocr"  # "easyocr" or "tesseract"


PATHS = PathConfig()
FEATURES = FeatureConfig()
STAGE1 = Stage1Config()
STAGE2 = Stage2Config()
EVAL = EvalConfig()
MODELS = ModelNames()


def ensure_dirs():
    for d in [
        PATHS.raw_videos_dir,
        PATHS.features_dir,
        PATHS.segments_dir,
        PATHS.index_dir,
        PATHS.annotations_dir,
        PATHS.checkpoints_dir,
    ]:
        os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("Project root:", PROJECT_ROOT)
    print("Stage1 input dim:", FEATURES.stage1_input_dim)

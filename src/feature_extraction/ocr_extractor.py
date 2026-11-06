"""
OCR-based board/slide text-change signal.

For each sampled frame, OCR is run on the (cropped) board/slide region of the
frame. The signal fed into Stage 1 is the *rate of textual change* between
consecutive sampled frames - a proxy for "the instructor wrote/erased/changed
the slide", which is one of the strongest cues for a concept boundary in a
lecture video.

Two backends are supported:
  - EasyOCR (default, GPU-accelerated if available)
  - Tesseract via pytesseract (lighter, CPU-only)

Both are *frozen* - no training happens here. If the requested engine cannot
be loaded, `get_ocr_backend` raises instead of silently returning an empty
backend: an all-zero OCR signal would quietly remove a whole modality from
every downstream experiment.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import difflib

import numpy as np


@dataclass
class OCRFrameResult:
    timestamp: float
    text: str


class BaseOCRBackend:
    def read_text(self, frame: np.ndarray) -> str:
        raise NotImplementedError


class EasyOCRBackend(BaseOCRBackend):
    def __init__(self, languages: Optional[List[str]] = None, gpu: bool = False):
        import easyocr  # lazy import

        self.reader = easyocr.Reader(languages or ["en"], gpu=gpu)

    def read_text(self, frame: np.ndarray) -> str:
        import cv2
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This dramatically improves contrast for handwriting on chalkboards,
        # whiteboards, and digital tablets (e.g. OneNote screen recordings)
        # regardless of whether it's light mode or dark mode.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        results = self.reader.readtext(enhanced, detail=0, paragraph=True)
        return " ".join(results)


class TesseractBackend(BaseOCRBackend):
    def __init__(self, lang: str = "eng"):
        self.lang = lang

    def read_text(self, frame: np.ndarray) -> str:
        import pytesseract  # lazy import
        from PIL import Image

        img = Image.fromarray(frame)
        return pytesseract.image_to_string(img, lang=self.lang)


class DummyOCRBackend(BaseOCRBackend):
    """Fallback backend used only when no real OCR engine is available.

    Returns an empty string, which yields a zero text-change signal. This
    keeps the pipeline runnable end-to-end (e.g. for unit tests) but should
    NOT be used for real experiments.
    """

    def read_text(self, frame: np.ndarray) -> str:
        return ""


def get_ocr_backend(engine: str = "easyocr", gpu: Optional[bool] = None,
                    allow_dummy: bool = False) -> BaseOCRBackend:
    """Load the requested OCR engine.

    Raises RuntimeError if it cannot be loaded, unless `allow_dummy=True`
    (smoke tests only). EasyOCR downloads its weights on first use; if that
    download fails, fetch craft_mlt_25k.pth and english_g2.pth into
    ~/.EasyOCR/model/ manually (see README).
    """
    # Auto-detect GPU if not explicitly provided
    if gpu is None:
        try:
            import torch
            gpu = torch.cuda.is_available()
        except ImportError:
            gpu = False

    try:
        if engine == "easyocr":
            return EasyOCRBackend(gpu=gpu)
        if engine == "tesseract":
            import pytesseract
            pytesseract.get_tesseract_version()  # fails fast if binary is missing
            return TesseractBackend()
        raise ValueError(f"Unknown OCR engine '{engine}'")
    except Exception as e:
        if allow_dummy:
            print(f"[OCR] WARNING: {engine} unavailable ({e}); using EMPTY dummy OCR.")
            return DummyOCRBackend()
        raise RuntimeError(f"OCR engine '{engine}' could not be loaded: {e}") from e


def board_region(frame: np.ndarray, bbox: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
    """Crop the board/slide region of a frame.

    `bbox` is given as fractional (x_min, y_min, x_max, y_max) in [0, 1].
    Use `detect_screen_bbox` (or a manual entry in data/ocr_regions.json) to
    get it. The fallback default trims the frame edges, where recorder
    overlays (CCTV clock, watermarks) live: OCR-ing a ticking clock would make
    the text-change signal fire on every frame.
    """
    h, w = frame.shape[:2]
    if bbox is None:
        bbox = (0.05, 0.10, 0.95, 0.85)
    x0, y0, x1, y1 = bbox
    return frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def detect_screen_bbox(frames: List[np.ndarray], num_samples: int = 60,
                       min_area_frac: float = 0.03,
                       pad_frac: float = 0.01) -> Optional[Tuple[float, float, float, float]]:
    """Locate the projected slide / screen in a fixed-camera classroom video.

    The projected screen is the largest bright, roughly rectangular region
    that stays put across the lecture. We take the per-pixel median over
    `num_samples` evenly spaced frames (which removes the moving instructor),
    Otsu-threshold it, and return the bounding box of the largest bright
    blob with a screen-like aspect ratio. Returns None if nothing plausible
    is found (caller then falls back to `board_region`'s default crop).
    """
    import cv2

    if not frames:
        return None
    idx = np.linspace(0, len(frames) - 1, min(num_samples, len(frames))).astype(int)
    gray = np.stack([cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY) for i in idx])
    median = np.median(gray, axis=0).astype(np.uint8)
    h, w = median.shape

    _, mask = cv2.threshold(cv2.GaussianBlur(median, (5, 5), 0), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best, best_area = None, 0.0
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        aspect = bw / max(bh, 1)
        fill = cv2.contourArea(c) / max(area, 1)
        if area < min_area_frac * h * w or not (1.0 <= aspect <= 2.6) or fill < 0.6:
            continue
        if area > best_area:
            best, best_area = (x, y, x + bw, y + bh), area
    if best is None:
        return None

    x0, y0, x1, y1 = best
    return (max(0.0, x0 / w - pad_frac), max(0.0, y0 / h - pad_frac),
            min(1.0, x1 / w + pad_frac), min(1.0, y1 / h + pad_frac))


def resolve_ocr_bboxes(video_id: str, frames: List[np.ndarray],
                       overrides_path: Optional[str] = None,
                       chunk_steps: int = 100) -> Tuple[List[Optional[Tuple[float, float, float, float]]], dict]:
    """Return one crop bbox per frame, plus a summary of where they came from.

    Priority: a manual entry in data/ocr_regions.json (applied to the whole
    video) > automatic screen detection per chunk of `chunk_steps` frames
    (camera framing can change mid-recording) > detection over the whole
    video > None (board_region's default edge-trim crop).
    """
    import json
    import os

    T = len(frames)
    if overrides_path and os.path.exists(overrides_path):
        with open(overrides_path, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        if video_id in overrides:
            bbox = tuple(float(v) for v in overrides[video_id])
            return [bbox] * T, {"manual": T}

    global_bbox = detect_screen_bbox(frames)
    bboxes, summary = [], {"auto_chunk": 0, "auto_global": 0, "default": 0}
    for start in range(0, T, chunk_steps):
        chunk = frames[start:start + chunk_steps]
        bbox = detect_screen_bbox(chunk, num_samples=20)
        if bbox is not None:
            key = "auto_chunk"
        elif global_bbox is not None:
            bbox, key = global_bbox, "auto_global"
        else:
            key = "default"
        bboxes.extend([bbox] * len(chunk))
        summary[key] += len(chunk)
    return bboxes, summary


def compute_ssim(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Compute Structural Similarity Index (SSIM) between two frames.

    Uses OpenCV to compute a fast grayscale SSIM. Returns a value in [0, 1]
    where 1.0 means the frames are identical.
    """
    import cv2

    def _to_gray(f):
        if f.ndim == 3:
            return cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        return f

    gray_a = _to_gray(frame_a)
    gray_b = _to_gray(frame_b)

    # Resize to match if shapes differ (shouldn't happen, but safety)
    if gray_a.shape != gray_b.shape:
        gray_b = cv2.resize(gray_b, (gray_a.shape[1], gray_a.shape[0]))

    # Use cv2 matchTemplate with TM_CCOEFF_NORMED as a fast SSIM proxy,
    # or compute proper SSIM via statistics.
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    a = gray_a.astype(np.float64)
    b = gray_b.astype(np.float64)

    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)

    mu_a_sq = mu_a ** 2
    mu_b_sq = mu_b ** 2
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(a ** 2, (11, 11), 1.5) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(b ** 2, (11, 11), 1.5) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab

    ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / \
               ((mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2))

    return float(np.mean(ssim_map))


def ssim_gate(prev_frame: np.ndarray, curr_frame: np.ndarray,
              threshold: float = 0.95) -> bool:
    """Return True if the frames are similar enough to SKIP OCR.

    When SSIM >= threshold (default 0.95, i.e. <5% visual change),
    the board/slide is considered unchanged and OCR can be skipped.
    """
    return compute_ssim(prev_frame, curr_frame) >= threshold


def text_change_rate(prev_text: str, curr_text: str) -> float:
    """Compute a normalized text-change score in [0, 1].

    0 -> texts are identical (no board/slide change)
    1 -> texts are completely different
    """
    if not prev_text and not curr_text:
        return 0.0
    matcher = difflib.SequenceMatcher(None, prev_text, curr_text)
    similarity = matcher.ratio()
    return float(1.0 - similarity)


def extract_ocr_signal(
    frames: List[np.ndarray],
    backend: Optional[BaseOCRBackend] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Run OCR on a sequence of sampled frames and return the text-change
    signal alongside the raw OCR text for each frame (used later to build
    the segment-level OCR text for Stage 2).

    Returns
    -------
    signal : np.ndarray of shape (T,)
        signal[0] = 0 by convention; signal[t] = text_change_rate(t-1, t)
    texts : List[str]
        OCR text recognized for each frame.
    """
    backend = backend or get_ocr_backend()
    texts = [backend.read_text(board_region(f, bbox)) for f in frames]

    signal = np.zeros(len(frames), dtype=np.float32)
    for t in range(1, len(frames)):
        signal[t] = text_change_rate(texts[t - 1], texts[t])
    return signal, texts
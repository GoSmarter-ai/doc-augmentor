#!/usr/bin/env python3
"""Document Augmentor — PDF to augmented training images for document intelligence.

Generates realistic variations of PDF documents for training classification,
extraction, and OCR models. Powered by augraphy for industrial-grade
document degradation simulation.

Usage:
    python doc_augmentor.py input.pdf -n 10
    python doc_augmentor.py ./pdfs/ -n 5 --dpi 300
    python doc_augmentor.py input.pdf -n 3 --preset light
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np
from augraphy import (
    AugmentationSequence,
    AugraphyPipeline,
    BadPhotoCopy,
    BleedThrough,
    Brightness,
    BrightnessTexturize,
    ColorPaper,
    ColorShift,
    DirtyDrum,
    DirtyRollers,
    Faxify,
    Folding,
    Gamma,
    InkBleed,
    InkMottling,
    Jpeg,
    LightingGradient,
    LowInkPeriodicLines,
    LowInkRandomLines,
    Markup,
    NoiseTexturize,
    PageBorder,
    ReflectedLight,
    ShadowCast,
    Squish,
    Stains,
    SubtleNoise,
)


# ---------------------------------------------------------------------------
# Preset pipelines
# ---------------------------------------------------------------------------


def _ink_phase(preset: str) -> AugmentationSequence:
    if preset == "light":
        return AugmentationSequence(
            [
                InkBleed(p=0.3),
                InkMottling(p=0.3),
            ]
        )
    if preset == "medium":
        return AugmentationSequence(
            [
                InkBleed(p=0.5),
                InkMottling(p=0.5),
                LowInkRandomLines(p=0.3),
                LowInkPeriodicLines(p=0.2),
            ]
        )
    return AugmentationSequence(
        [
            InkBleed(p=0.7),
            InkMottling(p=0.6),
            LowInkRandomLines(p=0.5),
            LowInkPeriodicLines(p=0.4),
            BleedThrough(p=0.3),
        ]
    )


def _paper_phase(preset: str) -> AugmentationSequence:
    if preset == "light":
        return AugmentationSequence(
            [
                ColorPaper(p=0.3),
                NoiseTexturize(p=0.3),
                BrightnessTexturize(p=0.3),
            ]
        )
    if preset == "medium":
        return AugmentationSequence(
            [
                ColorPaper(p=0.5),
                NoiseTexturize(p=0.5),
                BrightnessTexturize(p=0.4),
                Stains(p=0.2),
                Folding(p=0.2),
            ]
        )
    return AugmentationSequence(
        [
            ColorPaper(p=0.6),
            NoiseTexturize(p=0.6),
            BrightnessTexturize(p=0.5),
            Stains(p=0.4),
            Folding(p=0.3),
            DirtyDrum(p=0.3),
            DirtyRollers(p=0.3),
        ]
    )


def _post_phase(preset: str) -> AugmentationSequence:
    if preset == "light":
        return AugmentationSequence(
            [
                Brightness(p=0.4),
                Gamma(p=0.3),
                SubtleNoise(p=0.4),
                Jpeg(p=0.3),
            ]
        )
    if preset == "medium":
        return AugmentationSequence(
            [
                Brightness(p=0.5),
                Gamma(p=0.4),
                SubtleNoise(p=0.5),
                Jpeg(p=0.4),
                LightingGradient(p=0.3),
                ColorShift(p=0.3),
                ShadowCast(p=0.2),
            ]
        )
    return AugmentationSequence(
        [
            Brightness(p=0.6),
            Gamma(p=0.5),
            SubtleNoise(p=0.5),
            Jpeg(p=0.5),
            LightingGradient(p=0.4),
            ColorShift(p=0.4),
            ShadowCast(p=0.3),
            BadPhotoCopy(p=0.3),
            Faxify(p=0.2),
            ReflectedLight(p=0.2),
            Squish(p=0.2),
            PageBorder(p=0.3),
            Markup(p=0.1),
        ]
    )


def build_pipeline(preset: str = "medium") -> AugraphyPipeline:
    return AugraphyPipeline(
        ink_phase=[_ink_phase(preset)],
        paper_phase=[_paper_phase(preset)],
        post_phase=[_post_phase(preset)],
    )


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------


def extract_audit(result_dict: dict) -> list[dict]:
    """Extract which augmentations fired and their parameters."""
    params = result_dict["log"]["augmentation_parameters"]
    audit = []
    for param in params:
        if "augmentations" not in param or "results" not in param:
            continue
        for aug, res in zip(param["augmentations"], param["results"]):
            name = type(aug).__name__
            fired = res is not None
            clean_params = {}
            for k, v in vars(aug).items():
                if k.startswith("_") or k in ("mask", "keypoints", "bounding_boxes", "result"):
                    continue
                if isinstance(v, (str, int, float, bool, type(None))):
                    clean_params[k] = v
                elif isinstance(v, tuple):
                    clean_params[k] = list(v)
            audit.append({"augmentation": name, "applied": fired, "parameters": clean_params})
    return audit


def augment_with_audit(pipeline: AugraphyPipeline, image):
    result = pipeline.augment(image, return_dict=1)
    audit = extract_audit(result)
    return result["output"], audit


# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------


def render_pdf_pages(pdf_path: Path, dpi: int = 200) -> list[np.ndarray]:
    doc = fitz.open(pdf_path)
    pages = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        pages.append(img)
    doc.close()
    return pages


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


def augment_pdf(
    pdf_path: Path,
    output_dir: Path,
    n_variations: int = 10,
    dpi: int = 200,
    preset: str = "medium",
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem

    print(f"Rendering {pdf_path.name} at {dpi} DPI...")
    pages = render_pdf_pages(pdf_path, dpi)
    print(f"  {len(pages)} page(s) rendered")

    saved = []
    manifest = {
        "source": pdf_path.name,
        "dpi": dpi,
        "preset": preset,
        "pages": len(pages),
        "images": [],
    }

    for page_idx, page_img in enumerate(pages):
        orig_path = output_dir / f"{stem}_p{page_idx:03d}_original.png"
        cv2.imwrite(str(orig_path), page_img)
        saved.append(orig_path)
        manifest["images"].append(
            {
                "file": orig_path.name,
                "page": page_idx,
                "type": "original",
                "augmentations": [],
            }
        )

    for var_idx in range(n_variations):
        pipeline = build_pipeline(preset)
        for page_idx, page_img in enumerate(pages):
            augmented, audit = augment_with_audit(pipeline, page_img)
            var_path = output_dir / f"{stem}_p{page_idx:03d}_var{var_idx:03d}.png"
            cv2.imwrite(str(var_path), augmented)
            saved.append(var_path)
            manifest["images"].append(
                {
                    "file": var_path.name,
                    "page": page_idx,
                    "type": "augmented",
                    "variation": var_idx,
                    "augmentations": audit,
                }
            )
        print(f"  Variation {var_idx + 1}/{n_variations} done")

    manifest_path = output_dir / f"{stem}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: {manifest_path}")

    return saved


def process_input(input_path: Path, output_dir: Path, n_variations: int, dpi: int, preset: str):
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        pdfs = [input_path]
    elif input_path.is_dir():
        pdfs = sorted(input_path.glob("*.pdf"))
        if not pdfs:
            print(f"No PDFs found in {input_path}")
            sys.exit(1)
    else:
        print(f"Not a PDF or directory: {input_path}")
        sys.exit(1)

    total_saved = []
    for pdf in pdfs:
        saved = augment_pdf(pdf, output_dir, n_variations, dpi, preset)
        total_saved.extend(saved)

    originals = len([p for p in total_saved if "original" in p.name])
    variations = len(total_saved) - originals
    print(f"\nDone. {originals} originals + {variations} augmented = {len(total_saved)} images")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Augment PDFs for document intelligence training")
    parser.add_argument("input", type=Path, help="PDF file or directory of PDFs")
    parser.add_argument(
        "-n", "--variations", type=int, default=10, help="Variations per page (default: 10)"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: ./augmented/<name>/)",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Render DPI (default: 200)")
    parser.add_argument(
        "--preset",
        choices=["light", "medium", "heavy"],
        default="medium",
        help="Augmentation intensity (default: medium)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    output_dir = args.output or Path("augmented") / args.input.stem
    process_input(args.input, output_dir, args.variations, args.dpi, args.preset)


if __name__ == "__main__":
    main()

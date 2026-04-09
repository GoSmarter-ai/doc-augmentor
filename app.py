#!/usr/bin/env python3
"""Document Augmentor UI — Drag & drop PDFs, generate augmented training data.

Configurable output directory, batch mode for category folders,
full audit trail, and 25 individual augmentation controls.

Usage:
    python app.py
    python app.py --output ./my_training_data --source ./my_documents --port 7860
"""

import argparse
import json
from pathlib import Path

import cv2
import fitz
import gradio as gr
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
    Rescale,
    ShadowCast,
    Squish,
    Stains,
    SubtleNoise,
)
from doc_augmentor import extract_audit, render_pdf_pages

# Configured at startup via CLI args
OUTPUT_DIR = Path("./augmented")
SOURCE_DIR = None  # Optional: for batch mode

_current_pages = []


def _format_audit(audit: list[dict]) -> str:
    applied = [a for a in audit if a["applied"]]
    skipped = [a for a in audit if not a["applied"]]
    lines = []
    if applied:
        lines.append("APPLIED:")
        for a in applied:
            params = ", ".join(
                f"{k}={v}" for k, v in a["parameters"].items() if k not in ("numba_jit", "mask")
            )
            lines.append(f"  + {a['augmentation']}" + (f"  [{params}]" if params else ""))
    if skipped:
        lines.append("SKIPPED:")
        for a in skipped:
            lines.append(f"  - {a['augmentation']}")
    return "\n".join(lines)


def _build_custom_pipeline(
    ink_bleed_on,
    ink_bleed_p,
    ink_mottling_on,
    ink_mottling_p,
    low_ink_random_on,
    low_ink_random_p,
    low_ink_periodic_on,
    low_ink_periodic_p,
    bleed_through_on,
    bleed_through_p,
    color_paper_on,
    color_paper_p,
    noise_texturize_on,
    noise_texturize_p,
    brightness_texturize_on,
    brightness_texturize_p,
    stains_on,
    stains_p,
    folding_on,
    folding_p,
    dirty_drum_on,
    dirty_drum_p,
    dirty_rollers_on,
    dirty_rollers_p,
    brightness_on,
    brightness_p,
    gamma_on,
    gamma_p,
    gamma_range_lo,
    gamma_range_hi,
    subtle_noise_on,
    subtle_noise_p,
    jpeg_on,
    jpeg_p,
    jpeg_quality_lo,
    jpeg_quality_hi,
    lighting_gradient_on,
    lighting_gradient_p,
    color_shift_on,
    color_shift_p,
    shadow_cast_on,
    shadow_cast_p,
    bad_photocopy_on,
    bad_photocopy_p,
    faxify_on,
    faxify_p,
    reflected_light_on,
    reflected_light_p,
    squish_on,
    squish_p,
    page_border_on,
    page_border_p,
    markup_on,
    markup_p,
    rescale_on,
    rescale_factor,
):
    ink = []
    if ink_bleed_on:
        ink.append(InkBleed(p=ink_bleed_p))
    if ink_mottling_on:
        ink.append(InkMottling(p=ink_mottling_p))
    if low_ink_random_on:
        ink.append(LowInkRandomLines(p=low_ink_random_p))
    if low_ink_periodic_on:
        ink.append(LowInkPeriodicLines(p=low_ink_periodic_p))
    if bleed_through_on:
        ink.append(BleedThrough(p=bleed_through_p))

    paper = []
    if color_paper_on:
        paper.append(ColorPaper(p=color_paper_p))
    if noise_texturize_on:
        paper.append(NoiseTexturize(p=noise_texturize_p))
    if brightness_texturize_on:
        paper.append(BrightnessTexturize(p=brightness_texturize_p))
    if stains_on:
        paper.append(Stains(p=stains_p))
    if folding_on:
        paper.append(Folding(p=folding_p))
    if dirty_drum_on:
        paper.append(DirtyDrum(p=dirty_drum_p))
    if dirty_rollers_on:
        paper.append(DirtyRollers(p=dirty_rollers_p))

    post = []
    if brightness_on:
        post.append(Brightness(p=brightness_p))
    if gamma_on:
        post.append(Gamma(gamma_range=(gamma_range_lo, gamma_range_hi), p=gamma_p))
    if subtle_noise_on:
        post.append(SubtleNoise(p=subtle_noise_p))
    if jpeg_on:
        post.append(Jpeg(quality_range=(int(jpeg_quality_lo), int(jpeg_quality_hi)), p=jpeg_p))
    if lighting_gradient_on:
        post.append(LightingGradient(p=lighting_gradient_p))
    if color_shift_on:
        post.append(ColorShift(p=color_shift_p))
    if shadow_cast_on:
        post.append(ShadowCast(p=shadow_cast_p))
    if bad_photocopy_on:
        post.append(BadPhotoCopy(p=bad_photocopy_p))
    if faxify_on:
        post.append(Faxify(p=faxify_p))
    if reflected_light_on:
        post.append(ReflectedLight(p=reflected_light_p))
    if squish_on:
        post.append(Squish(p=squish_p))
    if page_border_on:
        post.append(PageBorder(p=page_border_p))
    if markup_on:
        post.append(Markup(p=markup_p))
    if rescale_on:
        post.append(Rescale(target_dpi=int(rescale_factor * 200), p=1.0))

    return AugraphyPipeline(
        ink_phase=[AugmentationSequence(ink)] if ink else [],
        paper_phase=[AugmentationSequence(paper)] if paper else [],
        post_phase=[AugmentationSequence(post)] if post else [],
    )


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def load_pdf(file, dpi):
    global _current_pages
    if file is None:
        _current_pages = []
        return gr.update(choices=[], value=None), "Upload a PDF", [], []

    pdf_path = Path(file.name)
    _current_pages = render_pdf_pages(pdf_path, int(dpi))
    n = len(_current_pages)
    choices = [f"Page {i + 1}" for i in range(n)]
    originals_gallery = [
        (cv2.cvtColor(page, cv2.COLOR_BGR2RGB), f"Page {i + 1}")
        for i, page in enumerate(_current_pages)
    ]
    return (
        gr.update(choices=choices, value=choices[0] if choices else None),
        f"{n} page(s) loaded",
        originals_gallery,
        [],
    )


def preview_page(page_select, *args):
    global _current_pages
    if not _current_pages or page_select is None:
        return None, None, "No pages loaded", ""

    page_idx = int(page_select.split(" ")[1]) - 1
    if page_idx >= len(_current_pages):
        return None, None, "Page index out of range", ""

    page_img = _current_pages[page_idx]
    original = cv2.cvtColor(page_img, cv2.COLOR_BGR2RGB)
    pipeline = _build_custom_pipeline(*args)
    result = pipeline.augment(page_img, return_dict=1)
    audit = extract_audit(result)
    augmented_rgb = cv2.cvtColor(result["output"], cv2.COLOR_BGR2RGB)
    applied_count = sum(1 for a in audit if a["applied"])
    status = f"{page_select} of {len(_current_pages)} — {applied_count} augmentations applied"
    return original, augmented_rgb, status, _format_audit(audit)


def augment_all_pages(*args):
    global _current_pages
    if not _current_pages:
        return [], "No pages loaded"

    gallery = []
    audit_lines = []
    for i, page_img in enumerate(_current_pages):
        pipeline = _build_custom_pipeline(*args)
        result = pipeline.augment(page_img, return_dict=1)
        audit = extract_audit(result)
        rgb = cv2.cvtColor(result["output"], cv2.COLOR_BGR2RGB)
        applied = [a["augmentation"] for a in audit if a["applied"]]
        gallery.append((rgb, f"Page {i + 1}"))
        audit_lines.append(f"Page {i + 1}: {', '.join(applied) if applied else 'none'}")
    return gallery, "\n".join(audit_lines)


def run_augmentation(file, n_variations, dpi, *args, progress=gr.Progress()):
    global _current_pages
    if file is None:
        return "Upload a PDF first"

    pdf_path = Path(file.name)
    stem = pdf_path.stem
    output_dir = OUTPUT_DIR / stem
    output_dir.mkdir(parents=True, exist_ok=True)

    if not _current_pages:
        _current_pages = render_pdf_pages(pdf_path, int(dpi))

    pages = _current_pages
    n_pages = len(pages)
    n_vars = int(n_variations)
    total_images = n_pages * (1 + n_vars)
    manifest = {
        "source": pdf_path.name,
        "dpi": int(dpi),
        "pages": n_pages,
        "variations_per_page": n_vars,
        "images": [],
    }

    count = 0
    for page_idx, page_img in enumerate(pages):
        orig_path = output_dir / f"{stem}_p{page_idx:03d}_original.png"
        cv2.imwrite(str(orig_path), page_img)
        count += 1
        progress(count / total_images, desc=f"Original page {page_idx + 1}")
        manifest["images"].append(
            {
                "file": orig_path.name,
                "page": page_idx,
                "type": "original",
                "augmentations": [],
            }
        )

    for var_idx in range(n_vars):
        pipeline = _build_custom_pipeline(*args)
        for page_idx, page_img in enumerate(pages):
            result = pipeline.augment(page_img, return_dict=1)
            audit = extract_audit(result)
            var_path = output_dir / f"{stem}_p{page_idx:03d}_var{var_idx:03d}.png"
            cv2.imwrite(str(var_path), result["output"])
            count += 1
            progress(
                count / total_images,
                desc=f"Var {var_idx + 1}/{n_vars}, page {page_idx + 1}/{n_pages}",
            )
            manifest["images"].append(
                {
                    "file": var_path.name,
                    "page": page_idx,
                    "type": "augmented",
                    "variation": var_idx,
                    "augmentations": audit,
                }
            )

    manifest_path = output_dir / f"{stem}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total = n_pages + n_pages * n_vars
    size_mb = sum(f.stat().st_size for f in output_dir.glob("*.png")) / 1024 / 1024
    return (
        f"Done! {n_pages} originals + {n_pages * n_vars} augmented = {total} images "
        f"({size_mb:.1f} MB)\nManifest: {manifest_path.name}\nSaved to: {output_dir}"
    )


# ---------------------------------------------------------------------------
# Batch mode — process category folders
# ---------------------------------------------------------------------------


def scan_categories():
    if SOURCE_DIR is None or not SOURCE_DIR.exists():
        return "No source directory configured. Use --source /path/to/categories"

    categories = []
    for d in sorted(SOURCE_DIR.iterdir()):
        if not d.is_dir():
            continue
        pdfs = list(d.glob("*.pdf"))
        if not pdfs:
            continue
        total_pages = 0
        for pdf in pdfs:
            doc = fitz.open(pdf)
            total_pages += len(doc)
            doc.close()
        categories.append({"name": d.name, "path": str(d), "pdfs": len(pdfs), "pages": total_pages})

    if not categories:
        return f"No category folders with PDFs found in {SOURCE_DIR}"

    target = 1500
    lines = ["Category          | PDFs | Pages | Vars needed | ~Total images"]
    lines.append("-" * 65)
    for c in categories:
        vars_needed = max(1, round(target / c["pages"])) if c["pages"] > 0 else 10
        c["variations"] = vars_needed
        total_est = c["pages"] * (1 + vars_needed)
        lines.append(
            f"{c['name']:18s}| {c['pdfs']:4d} | {c['pages']:5d} | {vars_needed:11d} | ~{total_est:,}"
        )

    lines.append(f"\nTotal categories: {len(categories)}")
    total_all = sum(c["pages"] * (1 + c["variations"]) for c in categories)
    lines.append(f"Total estimated images: ~{total_all:,}")
    return "\n".join(lines)


def run_batch(target_per_class, dpi, *args, progress=gr.Progress()):
    if SOURCE_DIR is None or not SOURCE_DIR.exists():
        return "No source directory configured. Use --source /path/to/categories"

    target = int(target_per_class)
    categories = []
    for d in sorted(SOURCE_DIR.iterdir()):
        if not d.is_dir():
            continue
        pdfs = sorted(d.glob("*.pdf"))
        if not pdfs:
            continue
        total_pages = 0
        for pdf in pdfs:
            doc = fitz.open(pdf)
            total_pages += len(doc)
            doc.close()
        vars_needed = max(1, round(target / total_pages)) if total_pages > 0 else 10
        categories.append(
            {"name": d.name, "pdfs": pdfs, "pages": total_pages, "variations": vars_needed}
        )

    if not categories:
        return "No categories found"

    total_work = sum(c["pages"] * (1 + c["variations"]) for c in categories)
    done = 0
    results = []

    for cat in categories:
        cat_dir = OUTPUT_DIR / cat["name"]
        cat_dir.mkdir(parents=True, exist_ok=True)
        cat_manifest = {
            "category": cat["name"],
            "dpi": int(dpi),
            "target_per_class": target,
            "variations_per_page": cat["variations"],
            "pdfs": [],
        }

        for pdf_path in cat["pdfs"]:
            pages = render_pdf_pages(pdf_path, int(dpi))
            stem = pdf_path.stem
            pdf_entry = {"file": pdf_path.name, "pages": len(pages), "images": []}

            for page_idx, page_img in enumerate(pages):
                orig_path = cat_dir / f"{stem}_p{page_idx:03d}_original.png"
                cv2.imwrite(str(orig_path), page_img)
                done += 1
                progress(done / total_work, desc=f"{cat['name']} — {stem} originals")
                pdf_entry["images"].append(
                    {
                        "file": orig_path.name,
                        "page": page_idx,
                        "type": "original",
                        "augmentations": [],
                    }
                )

            for var_idx in range(cat["variations"]):
                pipeline = _build_custom_pipeline(*args)
                for page_idx, page_img in enumerate(pages):
                    result = pipeline.augment(page_img, return_dict=1)
                    audit = extract_audit(result)
                    var_path = cat_dir / f"{stem}_p{page_idx:03d}_var{var_idx:03d}.png"
                    cv2.imwrite(str(var_path), result["output"])
                    done += 1
                    progress(
                        done / total_work,
                        desc=f"{cat['name']} — {stem} var {var_idx + 1}/{cat['variations']}",
                    )
                    pdf_entry["images"].append(
                        {
                            "file": var_path.name,
                            "page": page_idx,
                            "type": "augmented",
                            "variation": var_idx,
                            "augmentations": audit,
                        }
                    )

            cat_manifest["pdfs"].append(pdf_entry)

        manifest_path = cat_dir / f"{cat['name']}_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(cat_manifest, f, indent=2)

        img_count = sum(len(p["images"]) for p in cat_manifest["pdfs"])
        size_mb = sum(f.stat().st_size for f in cat_dir.glob("*.png")) / 1024 / 1024
        results.append(
            f"{cat['name']}: {img_count} images ({size_mb:.0f} MB) "
            f"[{len(cat['pdfs'])} PDFs x {cat['variations']} vars]"
        )

    return (
        f"BATCH COMPLETE — {len(categories)} categories, {done:,} images\n"
        f"Output: {OUTPUT_DIR}\n\n" + "\n".join(results)
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def build_ui():
    with gr.Blocks(
        title="Document Augmentor",
        theme=gr.themes.Soft(),
        css="""
        .main-title { text-align: center; margin-bottom: 0.2em; }
        .subtitle { text-align: center; color: #666; margin-bottom: 1em; font-size: 0.95em; }
        .save-path { background: #f0f7ff; padding: 6px 12px; border-radius: 6px;
                     border-left: 3px solid #2563eb; margin-bottom: 1em; font-size: 0.85em; }
        .full-preview img { max-height: none !important; }
        """,
    ) as app:
        gr.HTML("<h1 class='main-title'>Document Augmentor</h1>")
        gr.HTML(
            "<p class='subtitle'>"
            "Drop PDF &rarr; configure augmentations &rarr; generate training data for document intelligence"
            "</p>"
        )
        gr.HTML(f"<div class='save-path'>Output: <code>{OUTPUT_DIR.resolve()}</code></div>")

        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                file_input = gr.File(label="Drop PDF here", file_types=[".pdf"], type="filepath")
                page_select = gr.Dropdown(label="Page", choices=[], interactive=True)
                dpi = gr.Slider(minimum=72, maximum=600, value=200, step=10, label="Render DPI")
                n_variations = gr.Slider(
                    minimum=1, maximum=100, value=10, step=1, label="Variations per page"
                )

                with gr.Accordion("Ink Effects", open=False):
                    ink_bleed_on = gr.Checkbox(label="Ink Bleed", value=True)
                    ink_bleed_p = gr.Slider(0, 1, 0.5, step=0.05, label="probability")
                    ink_mottling_on = gr.Checkbox(label="Ink Mottling", value=True)
                    ink_mottling_p = gr.Slider(0, 1, 0.5, step=0.05, label="probability")
                    low_ink_random_on = gr.Checkbox(label="Low Ink Random Lines", value=True)
                    low_ink_random_p = gr.Slider(0, 1, 0.3, step=0.05, label="probability")
                    low_ink_periodic_on = gr.Checkbox(label="Low Ink Periodic Lines", value=True)
                    low_ink_periodic_p = gr.Slider(0, 1, 0.2, step=0.05, label="probability")
                    bleed_through_on = gr.Checkbox(label="Bleed Through", value=False)
                    bleed_through_p = gr.Slider(0, 1, 0.3, step=0.05, label="probability")

                with gr.Accordion("Paper Effects", open=False):
                    color_paper_on = gr.Checkbox(label="Colour Paper", value=True)
                    color_paper_p = gr.Slider(0, 1, 0.5, step=0.05, label="probability")
                    noise_texturize_on = gr.Checkbox(label="Noise Texturize", value=True)
                    noise_texturize_p = gr.Slider(0, 1, 0.5, step=0.05, label="probability")
                    brightness_texturize_on = gr.Checkbox(label="Brightness Texturize", value=True)
                    brightness_texturize_p = gr.Slider(0, 1, 0.4, step=0.05, label="probability")
                    stains_on = gr.Checkbox(label="Stains", value=False)
                    stains_p = gr.Slider(0, 1, 0.2, step=0.05, label="probability")
                    folding_on = gr.Checkbox(label="Folding", value=False)
                    folding_p = gr.Slider(0, 1, 0.2, step=0.05, label="probability")
                    dirty_drum_on = gr.Checkbox(label="Dirty Drum", value=False)
                    dirty_drum_p = gr.Slider(0, 1, 0.3, step=0.05, label="probability")
                    dirty_rollers_on = gr.Checkbox(label="Dirty Rollers", value=False)
                    dirty_rollers_p = gr.Slider(0, 1, 0.3, step=0.05, label="probability")

                with gr.Accordion("Scanning & Photo Effects", open=False):
                    brightness_on = gr.Checkbox(label="Brightness", value=True)
                    brightness_p = gr.Slider(0, 1, 0.5, step=0.05, label="probability")
                    gamma_on = gr.Checkbox(label="Gamma", value=True)
                    gamma_p = gr.Slider(0, 1, 0.4, step=0.05, label="probability")
                    gamma_range_lo = gr.Slider(0.1, 1.0, 0.5, step=0.05, label="gamma min")
                    gamma_range_hi = gr.Slider(1.0, 3.0, 1.5, step=0.05, label="gamma max")
                    subtle_noise_on = gr.Checkbox(label="Subtle Noise", value=True)
                    subtle_noise_p = gr.Slider(0, 1, 0.5, step=0.05, label="probability")
                    jpeg_on = gr.Checkbox(label="JPEG Compression", value=True)
                    jpeg_p = gr.Slider(0, 1, 0.4, step=0.05, label="probability")
                    jpeg_quality_lo = gr.Slider(5, 80, 25, step=5, label="quality min")
                    jpeg_quality_hi = gr.Slider(50, 100, 95, step=5, label="quality max")
                    lighting_gradient_on = gr.Checkbox(label="Lighting Gradient", value=False)
                    lighting_gradient_p = gr.Slider(0, 1, 0.3, step=0.05, label="probability")
                    color_shift_on = gr.Checkbox(label="Colour Shift", value=False)
                    color_shift_p = gr.Slider(0, 1, 0.3, step=0.05, label="probability")
                    shadow_cast_on = gr.Checkbox(label="Shadow Cast", value=False)
                    shadow_cast_p = gr.Slider(0, 1, 0.2, step=0.05, label="probability")

                with gr.Accordion("Degradation & Distortion", open=False):
                    bad_photocopy_on = gr.Checkbox(label="Bad Photocopy", value=False)
                    bad_photocopy_p = gr.Slider(0, 1, 0.3, step=0.05, label="probability")
                    faxify_on = gr.Checkbox(label="Faxify", value=False)
                    faxify_p = gr.Slider(0, 1, 0.2, step=0.05, label="probability")
                    reflected_light_on = gr.Checkbox(label="Reflected Light", value=False)
                    reflected_light_p = gr.Slider(0, 1, 0.2, step=0.05, label="probability")
                    squish_on = gr.Checkbox(label="Squish", value=False)
                    squish_p = gr.Slider(0, 1, 0.2, step=0.05, label="probability")
                    page_border_on = gr.Checkbox(label="Page Border", value=False)
                    page_border_p = gr.Slider(0, 1, 0.3, step=0.05, label="probability")
                    markup_on = gr.Checkbox(label="Markup / Highlight", value=False)
                    markup_p = gr.Slider(0, 1, 0.1, step=0.05, label="probability")

                with gr.Accordion("Output Scaling", open=False):
                    rescale_on = gr.Checkbox(label="Rescale output", value=False)
                    rescale_factor = gr.Slider(0.25, 2.0, 1.0, step=0.05, label="scale factor")

            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.TabItem("All Pages"):
                        gr.HTML(
                            "<p style='color:#666;font-size:0.9em;'>Click any thumbnail to enlarge.</p>"
                        )
                        with gr.Row():
                            originals_gallery = gr.Gallery(
                                label="Originals",
                                columns=4,
                                height=500,
                                object_fit="contain",
                                preview=True,
                            )
                            augmented_gallery = gr.Gallery(
                                label="Augmented",
                                columns=4,
                                height=500,
                                object_fit="contain",
                                preview=True,
                            )
                        augment_all_btn = gr.Button("Augment all pages", variant="secondary")
                        all_pages_audit = gr.Textbox(
                            label="Per-page summary", interactive=False, lines=8
                        )

                    with gr.TabItem("Single Page Inspect"):
                        with gr.Row():
                            original_img = gr.Image(
                                label="Original", type="numpy", elem_classes=["full-preview"]
                            )
                            augmented_img = gr.Image(
                                label="Augmented", type="numpy", elem_classes=["full-preview"]
                            )
                        reroll_btn = gr.Button("Reroll this page", variant="secondary")

                    with gr.TabItem("Full Size"):
                        original_full = gr.Image(
                            label="Original — full size", type="numpy", height=900
                        )
                        augmented_full = gr.Image(
                            label="Augmented — full size", type="numpy", height=900
                        )

                    with gr.TabItem("Batch — Categories"):
                        gr.HTML(
                            "<p style='color:#666;font-size:0.9em;'>"
                            "Process all category folders. Each subfolder = one class. "
                            "Auto-balances variation counts for equal dataset sizes."
                            "</p>"
                        )
                        scan_btn = gr.Button("Scan categories", variant="secondary")
                        category_info = gr.Textbox(
                            label="Category breakdown", interactive=False, lines=14
                        )
                        target_per_class = gr.Slider(
                            minimum=200,
                            maximum=5000,
                            value=1500,
                            step=100,
                            label="Target images per category",
                        )
                        batch_dpi = gr.Slider(
                            minimum=72, maximum=600, value=200, step=10, label="Render DPI"
                        )
                        run_batch_btn = gr.Button("Run batch", variant="primary", size="lg")
                        batch_status = gr.Textbox(label="Batch output", interactive=False, lines=12)

                preview_status = gr.Textbox(label="Status", interactive=False)
                audit_display = gr.Textbox(
                    label="Augmentation audit trail", interactive=False, lines=10
                )
                generate_btn = gr.Button("Generate & save", variant="primary", size="lg")
                output_status = gr.Textbox(label="Output", interactive=False, lines=4)

        aug_controls = [
            ink_bleed_on,
            ink_bleed_p,
            ink_mottling_on,
            ink_mottling_p,
            low_ink_random_on,
            low_ink_random_p,
            low_ink_periodic_on,
            low_ink_periodic_p,
            bleed_through_on,
            bleed_through_p,
            color_paper_on,
            color_paper_p,
            noise_texturize_on,
            noise_texturize_p,
            brightness_texturize_on,
            brightness_texturize_p,
            stains_on,
            stains_p,
            folding_on,
            folding_p,
            dirty_drum_on,
            dirty_drum_p,
            dirty_rollers_on,
            dirty_rollers_p,
            brightness_on,
            brightness_p,
            gamma_on,
            gamma_p,
            gamma_range_lo,
            gamma_range_hi,
            subtle_noise_on,
            subtle_noise_p,
            jpeg_on,
            jpeg_p,
            jpeg_quality_lo,
            jpeg_quality_hi,
            lighting_gradient_on,
            lighting_gradient_p,
            color_shift_on,
            color_shift_p,
            shadow_cast_on,
            shadow_cast_p,
            bad_photocopy_on,
            bad_photocopy_p,
            faxify_on,
            faxify_p,
            reflected_light_on,
            reflected_light_p,
            squish_on,
            squish_p,
            page_border_on,
            page_border_p,
            markup_on,
            markup_p,
            rescale_on,
            rescale_factor,
        ]

        preview_outputs = [original_img, augmented_img, preview_status, audit_display]

        def _sync_full(o, a):
            return o, a

        file_input.change(
            fn=load_pdf,
            inputs=[file_input, dpi],
            outputs=[page_select, preview_status, originals_gallery, augmented_gallery],
        ).then(
            fn=preview_page,
            inputs=[page_select] + aug_controls,
            outputs=preview_outputs,
        ).then(
            fn=_sync_full,
            inputs=[original_img, augmented_img],
            outputs=[original_full, augmented_full],
        )

        page_select.change(
            fn=preview_page,
            inputs=[page_select] + aug_controls,
            outputs=preview_outputs,
        ).then(
            fn=_sync_full,
            inputs=[original_img, augmented_img],
            outputs=[original_full, augmented_full],
        )

        reroll_btn.click(
            fn=preview_page,
            inputs=[page_select] + aug_controls,
            outputs=preview_outputs,
        ).then(
            fn=_sync_full,
            inputs=[original_img, augmented_img],
            outputs=[original_full, augmented_full],
        )

        augment_all_btn.click(
            fn=augment_all_pages, inputs=aug_controls, outputs=[augmented_gallery, all_pages_audit]
        )

        generate_btn.click(
            fn=run_augmentation,
            inputs=[file_input, n_variations, dpi] + aug_controls,
            outputs=[output_status],
        )

        scan_btn.click(fn=scan_categories, inputs=[], outputs=[category_info])
        run_batch_btn.click(
            fn=run_batch,
            inputs=[target_per_class, batch_dpi] + aug_controls,
            outputs=[batch_status],
        )

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Augmentor UI")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./augmented"),
        help="Output directory for augmented images (default: ./augmented)",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Source directory with category subfolders for batch mode",
    )
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    args = parser.parse_args()

    OUTPUT_DIR = args.output
    SOURCE_DIR = args.source

    print(f"\n  Output directory: {OUTPUT_DIR.resolve()}")
    if SOURCE_DIR:
        print(f"  Source directory: {SOURCE_DIR.resolve()}")
    print(f"  Open in browser:  http://127.0.0.1:{args.port}\n")

    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=args.port, show_error=True)

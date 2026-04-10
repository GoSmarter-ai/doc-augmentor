# Document Augmentor

Generate realistic augmented training data from PDF documents for document intelligence models.

Drop a PDF, configure 25 augmentation effects, and generate hundreds of realistic variations — ink bleed, scanner noise, paper aging, JPEG compression, coffee stains, and more. Built for training classification, extraction, and OCR models in manufacturing and industrial settings.

![UI Preview](docs/preview.png)

## Features

- **Drag & drop UI** — Gradio-based web interface with live preview
- **25 augmentation effects** — ink, paper, scanning, and degradation simulation via [augraphy](https://github.com/sparkfish/augraphy)
- **Full audit trail** — JSON manifest records exactly which augmentations fired on every image
- **Batch mode** — process entire category folders with auto-balanced variation counts
- **CLI mode** — scriptable for CI/CD pipelines
- **All pages** — renders and augments every page in multi-page PDFs

## Quick Start

```bash
git clone https://github.com/Seraphiel102/doc-augmentor.git
cd doc-augmentor
pip install -r requirements.txt

# Launch the UI
python app.py

# Or use the CLI
python doc_augmentor.py invoice.pdf -n 10 --preset medium
```

## UI Mode

```bash
# Basic — outputs to ./augmented/
python app.py

# Custom output directory
python app.py --output /path/to/training_data

# Batch mode — each subfolder is a category/class
python app.py --source /path/to/documents --output /path/to/training_data

# Custom port
python app.py --port 8080
```

### Batch Mode

Organise your source documents into folders by category:

```
documents/
├── supplier_a/
│   ├── cert_001.pdf
│   └── cert_002.pdf
├── supplier_b/
│   └── cert_003.pdf
└── supplier_c/
    ├── cert_004.pdf
    └── cert_005.pdf
```

Run with `--source`:

```bash
python app.py --source ./documents --output ./training_data
```

The batch tab auto-calculates how many variations each category needs so every class ends up with roughly equal training data. Categories with fewer source documents get more augmentation.

## CLI Mode

```bash
# Single PDF, 10 variations per page
python doc_augmentor.py certificate.pdf -n 10

# Directory of PDFs
python doc_augmentor.py ./invoices/ -n 5 --dpi 300

# Light augmentation for clean documents
python doc_augmentor.py report.pdf -n 20 --preset light

# Reproducible output
python doc_augmentor.py cert.pdf -n 10 --seed 42
```

### Presets

| Preset | Effects | Use case |
|--------|---------|----------|
| `light` | Subtle ink + brightness + noise | Clean scanned documents |
| `medium` | Ink bleed, paper texture, scanner effects | General purpose |
| `heavy` | Stains, folding, fax, photocopy degradation | Worst-case real-world docs |

## Augmentation Effects

### Ink Effects
- **Ink Bleed** — ink spreading into paper fibres
- **Ink Mottling** — uneven ink density
- **Low Ink Lines** — random and periodic low-ink streaks
- **Bleed Through** — ink visible from the reverse side

### Paper Effects
- **Colour Paper** — aged/tinted paper background
- **Noise Texturize** — paper grain and texture
- **Brightness Texturize** — uneven paper brightness
- **Stains** — coffee, water, grease marks
- **Folding** — fold lines and creases
- **Dirty Drum/Rollers** — printer drum artefacts

### Scanning & Photo Effects
- **Brightness/Gamma** — exposure variation
- **Subtle Noise** — sensor noise
- **JPEG Compression** — compression artefacts with configurable quality range
- **Lighting Gradient** — uneven scanner illumination
- **Colour Shift** — chromatic aberration
- **Shadow Cast** — document edge shadows

### Degradation & Distortion
- **Bad Photocopy** — multi-generation copy degradation
- **Faxify** — fax machine simulation
- **Reflected Light** — glossy surface reflections
- **Squish** — feed roller distortion
- **Page Border** — scanner edge effects
- **Markup** — highlighter and pen marks

## Output Structure

```
augmented/
├── certificate_001/
│   ├── certificate_001_p000_original.png
│   ├── certificate_001_p000_var000.png
│   ├── certificate_001_p000_var001.png
│   ├── ...
│   └── certificate_001_manifest.json
```

### Manifest

Each output folder includes a JSON manifest with full traceability:

```json
{
  "source": "certificate_001.pdf",
  "dpi": 200,
  "pages": 3,
  "variations_per_page": 10,
  "images": [
    {
      "file": "certificate_001_p000_var000.png",
      "page": 0,
      "type": "augmented",
      "variation": 0,
      "augmentations": [
        {"augmentation": "InkBleed", "applied": true, "parameters": {"p": 0.5}},
        {"augmentation": "ColorPaper", "applied": false, "parameters": {"p": 0.5}},
        ...
      ]
    }
  ]
}
```

## Use Cases

- **Document Classification** — train models to classify document types (invoices, certificates, reports) across varying scan quality
- **OCR Training** — improve OCR robustness by training on degraded document images
- **Azure AI Document Intelligence** — generate labelled training data for custom classifiers and extractors
- **Quality Assurance** — test document processing pipelines against realistic document variation

## Requirements

- Python 3.10+
- Dependencies: augraphy, pymupdf, opencv-python, numpy, gradio

## License

MIT License — Nightingale HQ Ltd

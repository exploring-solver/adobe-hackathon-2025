# PDF Heading Extractor - Hybrid Approach

An advanced PDF heading detection system that combines fast heuristic analysis with intelligent semantic verification to accurately extract document structure.

## What This Project Does

This tool automatically extracts headings (Title, H1, H2, H3) from PDF documents and outputs them in a structured JSON format. It's specifically designed for the **Round 1A Hackathon Challenge** with optimizations for speed, accuracy, and multilingual support.

### Key Features

  - **Hybrid Intelligence**: Combines Adobe-style structure detection with smart heuristics.
  - **Semantic Verification**: Uses lightweight AI models to verify heading candidates.
  - **Multilingual Support**: Handles English, Japanese, Hindi, Arabic, and Chinese documents.
  - **Fast Processing**: Processes 50-page PDFs in under 10 seconds.
  - **High Accuracy**: Multi-strategy approach for superior heading detection.
  - **Robust Fallbacks**: Multiple detection strategies ensure reliability.

-----

## How It Works

Our hybrid approach uses a **three-stage pipeline**:

1.  **Adobe-Style Detection**: First checks for native PDF structure/bookmarks.
2.  **Heuristic Analysis**: Fast font-based candidate generation using typography patterns.
3.  **Semantic Filtering**: AI-powered verification using contextual analysis.

This combination delivers both **speed** and **accuracy**—faster than pure AI approaches, more accurate than simple heuristics.

-----

## Quick Start (Local Installation)

### Prerequisites

  - Python 3.8+
  - 4GB RAM minimum
  - No GPU required (CPU optimized)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/exploring-solver/adobe-hackathon-2025/tree/main
cd Challenge_1a

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create required directories
mkdir -p data/models logs outputs

# 5. Download models (happens automatically on first run)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

### Basic Usage

```bash
# Extract headings from a PDF
python -m src.main document.pdf

# Save results to file
python -m src.main document.pdf --output results.json

# Use Round 1A hackathon format
python -m src.main document.pdf --round1a --output results.json

# Enable debug mode for detailed logging
python -m src.main document.pdf --debug

# Specify document language for better accuracy
python -m src.main document.pdf --language ja  # Japanese
python -m src.main document.pdf --language hi  # Hindi
```

-----

## Running with Docker 🐳

This project includes a Dockerized environment for consistent, isolated execution without local setup hassles. This is the recommended way to test and process multiple PDFs.

### Prerequisites

  - Docker Desktop installed and running.
  - Basic understanding of your terminal or command prompt.
  - Your PDFs stored in a local directory (e.g., a folder named `pdfs` in your project root).

### Step 1: Build the Docker Image

Open your terminal and navigate to the root of the repository. Run the following command to build the Docker image and name it `pdf-heading-extractor`.

```bash
docker build -t pdf-heading-extractor .
```

### Step 2: Create a Reusable Container

To process multiple PDFs or use the CLI interactively, it's best to create a container once and reuse it. This mounts your local PDF and output folders into the container for easy file access.

**For Linux/Mac:**

```bash
docker create -it --name pdf_heading_debug \
  -v "$(pwd)/pdfs":/app/input \
  -v "$(pwd)/outputs":/app/output \
  pdf-heading-extractor
```

**For Windows (Command Prompt):**

```bash
docker create -it --name pdf_heading_debug \
  -v "%cd%\pdfs":/app/input \
  -v "%cd%\outputs":/app/output \
  pdf-heading-extractor
```

*Note: Ensure you have local `pdfs` and `outputs` directories or change the paths to your desired locations.*

### Step 3: Start an Interactive Session

To enter the container's shell and run the tool:

```bash
docker start -ai pdf_heading_debug
```

You are now inside the Docker environment. You can run the extraction tool on files from your mounted `input` folder and save results to the `output` folder.

```bash
# Process a file and save the result
python -m src.main input/file01.pdf --round1a --output output/result01.json

# Process a second file with a specified language
python -m src.main input/file02.pdf --language ja --round1a --output output/result02.json
```

### Step 4: Re-run Without Recreating

Once the container is created, you can stop and restart it anytime without rebuilding or remounting. Simply use the same command to re-enter your interactive session:

```bash
docker start -ai pdf_heading_debug
```

### Optional: Delete and Recreate Container

If you need to reset the container, first remove the old one and then run the `docker create` command from Step 2 again.

```bash
docker rm pdf_heading_debug
```

-----

## Step-by-Step Usage Guide

### Step 1: Prepare Your PDF

  - Ensure your PDF file is readable (not password-protected).
  - Place it in the directory you will use for processing (e.g., the `pdfs` folder for Docker).
  - Text-based PDFs work best (scanned images require OCR, which is not included).

### Step 2: Run the Extraction

```bash
# Using local python installation
python -m src.main your_document.pdf --round1a --output result.json

# Or, inside the Docker container (see Docker section above)
python -m src.main input/your_document.pdf --round1a --output output/result.json
```

### Step 3: Check the Output

The system generates a JSON file in the **Round 1A format**:

```json
{
  "title": "Understanding AI Systems",
  "outline": [
    {
      "level": "H1",
      "text": "Introduction",
      "page": 1
    },
    {
      "level": "H2",
      "text": "What is AI?",
      "page": 2
    },
    {
      "level": "H3",
      "text": "History of AI",
      "page": 3
    }
  ]
}
```

### Step 4: Advanced Options

```bash
# Process multiple languages
python -m src.main japanese_doc.pdf --language ja --round1a --output result_ja.json

# Enable fast mode (skip semantic filtering for speed)
FAST_MODE=true python -m src.main document.pdf --round1a --output result.json

# Get detailed processing statistics
python -m src.main document.pdf --debug --round1a --output result.json
```

-----

## Configuration Options

### Language Support

  - `auto` - Automatic detection (default)
  - `en` / `english` - English documents
  - `ja` / `japanese` - Japanese documents
  - `hi` / `hindi` - Hindi documents
  - `ar` / `arabic` - Arabic documents
  - `zh` / `chinese` - Chinese documents

### Performance Modes

  - **Standard Mode**: Full hybrid pipeline with semantic verification.
  - **Fast Mode**: `FAST_MODE=true` - Skips semantic filtering for maximum speed.
  - **Debug Mode**: `--debug` - Detailed logging and processing statistics.

### Output Formats

  - **Round 1A Format**: `--round1a` - Competition-specific JSON format.
  - **Extended Format**: Default - Includes confidence scores and metadata.
  - **Multiple Formats**: Supports JSON, CSV, XML, HTML, Markdown.

-----

## Expected Performance

| Document Type     | Processing Time | Accuracy | Multilingual Bonus |
| ----------------- | --------------- | -------- | ------------------ |
| Academic Papers   | 2-5 seconds     | 90-95%   | ✅ Full Support    |
| Technical Manuals | 3-7 seconds     | 85-92%   | ✅ Full Support    |
| Books/Reports     | 4-8 seconds     | 88-94%   | ✅ Full Support    |
| Complex Layouts   | 6-10 seconds    | 82-90%   | ✅ Full Support    |

-----

## Hackathon Optimizations

### Round 1A Compliance

  - **≤10 seconds** processing time for 50-page PDFs.
  - **≤200MB** total model size (MiniLM is only \~80MB).
  - **CPU-only** operation (no GPU dependencies).
  - **Offline mode** (all models cached locally).
  - **AMD64 compatible** architecture.

### Competitive Advantages

  - **Multilingual Support**: Handles Japanese, Hindi, Arabic for bonus points.
  - **Hybrid Intelligence**: More accurate than pure heuristics, faster than pure AI.
  - **Robust Fallbacks**: Multiple strategies ensure reliability across document types.
  - **Cultural Intelligence**: Understands document patterns across languages.

-----

## Troubleshooting

### Common Issues

**Import Errors:**
If you get import errors when running locally, ensure you are running the script as a module from the project's root directory:

```bash
python -m src.main document.pdf
```

**Model Download Issues:**
If the automatic model download fails, you can trigger it manually:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

**Memory Issues:**
Use fast mode to reduce memory usage, as it skips the AI-powered semantic analysis step:

```bash
FAST_MODE=true python -m src.main document.pdf --round1a
```

### Debug Mode

Enable debug mode for detailed diagnostics:

```bash
python -m src.main document.pdf --debug --round1a --output result.json
```

This provides:

  - Processing stage timings
  - Font analysis details
  - Candidate generation statistics
  - Semantic filtering scores
  - Hierarchy assignment logic

-----

## Architecture Overview

```
PDF Input → Structure Check → Candidate Generation → Semantic Filtering → Hierarchy Assignment → JSON Output
     ↓              ↓                     ↓                      ↓                      ↓                 ↓
  Validate      Adobe TOC          Font Analysis          AI Verification        Level Assignment    Round1A Format
              Detection         Layout Patterns         Context Analysis         Cultural Rules
```

-----

## What Makes This System Unique

1.  **Adobe-Grade Intelligence**: First checks for native PDF structure like professional tools.
2.  **Cultural Awareness**: Understands document patterns across different languages and cultures.
3.  **Adaptive Thresholds**: Adjusts detection sensitivity based on document characteristics.
4.  **Multi-Strategy Ensemble**: Combines 5+ detection strategies for maximum reliability.
5.  **Production Ready**: Comprehensive error handling, validation, and logging.

-----

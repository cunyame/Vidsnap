# Vidsnap(IN DEVELOPMENT)
# Smart Lecture Notes Generator

A CLI-based tool for Windows that processes local lecture videos, detects slide changes, and generates structured notes. Built in Python with modular components, and designed for future GUI integration and advanced features like OCR and summarization.

## Features

* **Phase 1 (Completed)**

  * Detect slide transitions in video using SSIM-based frame comparison (OpenCV).
  * Extract unique slide images and compile them into a single PDF report (img2pdf / PyPDF2).
  * Modular code structure with robust CLI argument parsing and logging.

* **Phase 2 (In Progress)**

  * OCR text extraction from slides using Tesseract (pytesseract).
  * Export enriched Markdown notes with slide images and extracted text.

* **Phase 3 (In Progress)**

  * Automated extractive summarization of slide content using Gensim TextRank.

## Tech Stack

* **Language:** Python 3
* **Video & Image Processing:** OpenCV, scikit‑image
* **PDF Generation:** img2pdf, PyPDF2
* **OCR:** Tesseract OCR, pytesseract
* **Summarization:** Gensim
* **CLI:** argparse, logging

## Prerequisites

* Python 3.7 or higher
* [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and added to PATH (for Phase 2 features)
* FFmpeg (optional, for high-performance frame extraction)
* 
## Usage

### Phase 1: Slide Extraction & PDF Report

```bash
python main.py --input "path/to/lecture.mp4" --output "slides.pdf"
```

* `--input`: Path to local video file.
* `--output`: Path to generated PDF containing slides.

### Phase 2: OCR & Markdown Export (Coming Soon)

```bash
python main.py --input "lecture.mp4" --ocr --markdown --output-dir "notes/"
```

* `--ocr`: Enable Tesseract OCR on extracted slides.
* `--markdown`: Generate a Markdown file with images and extracted text.

### Phase 3: Summarization (Coming Soon)

```bash
python main.py --input "lecture.mp4" --ocr --summarize --output-dir "notes/"
```

* `--summarize`: Add extractive summary of slide text to output.

## Project Structure

```
SmartLectureNotes/
├── slide_detector.py      # Detects slide changes
├── pdf_generator.py       # Generates PDF from slide images
├── ocr_processor.py       # Performs OCR on slide images
├── summarizer.py          # Summarizes text using Gensim
├── main.py                # CLI entry point
├── requirements.txt       # Python dependencies
└── README.md              # Project overview and usage
```

## Roadmap

* **Phase 2**: Complete OCR integration and Markdown export.
* **Phase 3**: Implement summarization and improve output formatting.
* **GUI Integration**: Build a desktop GUI using PyQt or Tkinter wrapping the CLI core.
* **Packaging**: Bundle as a standalone Windows executable via PyInstaller.

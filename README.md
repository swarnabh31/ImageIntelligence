# Local Image Intelligence Portal

A fully local, GPU-accelerated tool that lets you point at a folder of images and search them by **meaning**, not filenames or folders. Type "dog," "invoice," "sunset over water," or "person holding a phone" and it finds matching images — whether or not the image contains any text.

Everything runs on-device. No image, query, or metadata ever leaves the machine.

---

## 1. What it does

1. **Ingest** — point it at a folder; it scans every image, generates a semantic vector for each one, runs OCR to extract any visible text, and stores everything in a local vector database.
2. **Search** — type a natural-language query; it finds the closest-matching images by meaning, with an optional boost from OCR/filename keyword matches.
3. **Dedupe** — visually identical/near-identical images (resizes, recompressions, minor edits) are detected via perceptual hashing and only embedded once, even across repeated runs.

---

## 2. Core idea: how the search actually works

The system relies on **CLIP** (Contrastive Language–Image Pretraining, originally from OpenAI), a model trained to place images and text descriptions into the *same* vector space. A photo of a dog and the word "dog" end up near each other geometrically, even though one is a JPEG and the other is four letters.

The pipeline follows one strict rule:

> **Every image is always embedded with CLIP's image encoder. Every search query is always embedded with CLIP's text encoder. Those two vector spaces are compatible by design — that's the entire trick.**

OCR (text extracted from inside images, like signs, labels, or screenshots) is captured separately and stored as **metadata only**. It gives a small relevance boost during search but never substitutes for the actual visual embedding. This separation matters — an earlier version of this code blurred that line and silently broke search for any image with incidental "found text" (see §6, Known Limitations).

---

## 3. Architecture / pipeline

### Ingestion (per folder, run on demand)

```
Folder of images
      │
      ▼
┌─────────────────────────────┐
│ Phase 1 — Parallel CPU work │   (ThreadPoolExecutor, 8 workers)
│  • Perceptual hash (phash)  │
│  • OCR via Tesseract        │
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│ Phase 2 — Deduplication     │
│  phash seen before? → reuse │
│  existing embedding         │
│  new? → queue for embedding │
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│ Phase 3 — GPU batch embed   │
│  CLIP image encoder         │
│  (batches of 64, fp16)      │
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│ Phase 4 — Write to ChromaDB │
│  vector + OCR text + phash  │
│  + filepath as metadata     │
└─────────────────────────────┘
```

### Search (per query)

```
User types a query (e.g. "dog")
      │
      ▼
CLIP text encoder → 768-dim vector
      │
      ▼
ChromaDB nearest-neighbor search (cosine distance)
      │
      ▼
Hybrid re-rank:
  semantic similarity (weighted, default 85%)
  + keyword match against OCR text/filename (default 15%)
      │
      ▼
Results above the relevance threshold, shown with thumbnails
```

---

## 4. Tools and libraries used

| Component | Role |
|---|---|
| **CLIP** (`openai/clip-vit-large-patch14`, via Hugging Face `transformers`) | Generates the 768-dim semantic vectors for both images and text queries |
| **PyTorch** | Runs CLIP on GPU, float16 precision for speed/VRAM efficiency |
| **ChromaDB** (persistent, local) | Vector database — stores embeddings + metadata, handles nearest-neighbor search |
| **Tesseract OCR** (via `pytesseract`) | Extracts any visible text inside images (signs, labels, screenshots) |
| **OpenCV** (`cv2`) | Image sharpness detection and adaptive denoising before OCR |
| **imagehash** | Perceptual hashing (phash) for duplicate/near-duplicate detection |
| **Streamlit** | The web UI — folder ingestion screen and search screen |
| **Pillow (PIL)** | Image loading/format handling |

---

## 5. Prerequisites

**Hardware**
- An NVIDIA GPU is strongly recommended. The app falls back to CPU but throughput drops by roughly two orders of magnitude.
- Built and tuned against an RTX 5090 (32 GB VRAM); batch size of 64 is sized for that. Lower `CLIP_BATCH_SIZE` on smaller GPUs.

**Software**
- Python 3.10+
- CUDA-enabled PyTorch matching your GPU/driver
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed system-wide (the code currently hardcodes a Windows path to `tesseract.exe` — needs adjusting for macOS/Linux)
- Python packages: `streamlit`, `torch`, `transformers`, `chromadb`, `opencv-python`, `imagehash`, `pytesseract`, `numpy`, `Pillow`

**First run**
- Downloads the CLIP model weights (~1.7 GB) from Hugging Face — one-time, then cached locally.
- Creates a `chroma_db/` folder in the working directory to persist the vector index between sessions.

---

## 6. Known limitations

- **Single-folder, single-machine.** No multi-user access, no remote/cloud storage support — paths must be local and absolute.
- **OCR quality varies.** Tesseract can misread stylized text, low-contrast text, or handwriting, and occasionally hallucinates spurious "text" out of pure visual texture (grass, fur, noise) on images with no text at all. This is harmless now — OCR is metadata-only — but it means the keyword boost is noisier on some images than others.
- **CLIP's vocabulary is general-purpose.** It's strong on common objects, scenes, and concepts, but weaker on narrow domain-specific terms (e.g., specific brand logos, technical diagrams, niche jargon) it wasn't trained to associate with imagery.
- **77-token query cap.** CLIP's text encoder truncates queries beyond 77 tokens — long, paragraph-style queries get cut off.
- **No incremental re-embedding.** If you upgrade the CLIP model or change the embedding logic, previously indexed images must be fully wiped (`chroma_db/` deleted) and reprocessed — there's no in-place migration.
- **Tesseract path is hardcoded** to a Windows install location; needs to be made configurable or auto-detected for portability.
- **No batching across folders.** Each "Process Folder" run is scoped to one directory; subfolders aren't recursed into.
- **No authentication/access control** — this is a local single-user tool, not designed for shared/networked deployment as-is.

---

## 7. Design decision worth flagging explicitly

OCR text is **never** used to choose how an image gets embedded, and never substitutes for the image embedding itself. Every image — regardless of how much or how little text it contains — goes through CLIP's image encoder. OCR output is stored purely as searchable metadata for a secondary keyword boost. This was the subject of a real bug fix during development: an earlier version routed images with >20 characters of detected OCR text through CLIP's *text* encoder instead, which silently discarded their actual visual content from the embedding and made unrelated photos un-findable by visual queries. Worth understanding if anyone modifies the ingestion logic later.

---

## 8. Possible future improvements

- Lightweight image captioning (not full OCR) for richer searchable metadata on text-free images, e.g. "a golden retriever running on grass."
- Recursive folder scanning.
- Configurable/auto-detected Tesseract path for cross-platform use.
- A "force re-embed" option to recover from model upgrades without manually deleting the database.
- Multi-GPU or queued batch processing for very large folders (10,000+ images).

import os
import sys
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chromadb
import cv2
import imagehash
import numpy as np
import pytesseract
import streamlit as st
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# ── Tesseract: auto-detect first, then allow UI override ────────────────────
def _detect_tesseract() -> str:
    """Try common install locations for tesseract.exe."""
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    # Check PATH as a fallback
    try:
        which = shutil.which("tesseract")
        if which:
            candidates.insert(0, which)
    except Exception:
        pass
    for c in candidates:
        if os.path.isfile(c):
            return c
    return ""

_DEFAULT_TESSERACT = _detect_tesseract()

if _DEFAULT_TESSERACT:
    pytesseract.pytesseract.tesseract_cmd = _DEFAULT_TESSERACT

DB_DIR = os.path.join(os.getcwd(), "chroma_db")
META_FILE = os.path.join(DB_DIR, ".image_intelligence_meta.json")

# ── CLIP model ──────────────────────────────────────────────────────────────
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
CLIP_BATCH_SIZE = 64

# ── Captioning model (Florence-2, already cached on this machine) ────────────
CAPTION_MODEL_NAME = "microsoft/Florence-2-large"

# ── Parallelism knobs ────────────────────────────────────────────────────────
OCR_WORKERS   = 8
CAPTION_WORKERS = 4  # lighter weight, GPU-bound per-call
DB_BATCH_SIZE = 100

# ── Smart preprocessing ──────────────────────────────────────────────────────
SHARPNESS_THRESHOLD = 500.0
SUPPORTED_EXT       = (".png", ".jpg", ".jpeg", ".webp")

HAS_TEXT_MIN_LEN = 3

# Model version tracking for force-re-embed detection
MODEL_VERSION = "2026-07-01-v2"

# ---------------------------------------------------------------------------
# MODEL VERSION TRACKING
# ---------------------------------------------------------------------------

def _read_meta() -> dict:
    if os.path.isfile(META_FILE):
        try:
            with open(META_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _write_meta(d: dict) -> None:
    os.makedirs(os.path.dirname(META_FILE), exist_ok=True)
    with open(META_FILE, "w") as f:
        json.dump(d, f, indent=2)

def check_model_version():
    """Return True if model version matches; False if stale."""
    meta = _read_meta()
    return meta.get("model_version") == MODEL_VERSION

def save_model_version():
    meta = _read_meta()
    meta["model_version"] = MODEL_VERSION
    _write_meta(meta)

# ---------------------------------------------------------------------------
# CLIENTS
# ---------------------------------------------------------------------------

@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=DB_DIR)


chroma_client = get_chroma_client()
collection    = chroma_client.get_or_create_collection(name="image_intelligence")


@st.cache_resource
def load_clip_model():
    """Load CLIP into VRAM once; Streamlit caches it across reruns."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
    if device == "cuda":
        model = model.half()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    return model, processor, device

@st.cache_resource
def load_caption_model():
    """Load Florence-2 for lightweight image captioning."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            CAPTION_MODEL_NAME, trust_remote_code=True
        ).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            CAPTION_MODEL_NAME, trust_remote_code=True
        )
        return model, tokenizer, device
    except Exception as exc:
        st.warning(f"Captioning model failed to load: {exc}")
        return None, None, None

# ---------------------------------------------------------------------------
# EMBEDDING FUNCTIONS
# ---------------------------------------------------------------------------

def embed_images_batch(image_paths, model, processor, device):
    """GPU batch image embedding via CLIP image encoder. Always the image encoder."""
    pil_images = []
    for path in image_paths:
        try:
            with Image.open(path) as img:
                pil_images.append(img.convert("RGB").copy())
        except Exception:
            pil_images.append(Image.new("RGB", (224, 224), (128, 128, 128)))

    inputs = processor(
        images=pil_images,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.inference_mode():
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().float().tolist()


def embed_text(text, model, processor, device):
    """Single text embedding via CLIP text encoder for search queries."""
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    ).to(device)

    with torch.inference_mode():
        features = model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)

    return features[0].cpu().float().tolist()

# ---------------------------------------------------------------------------
# CAPTIONING (Florence-2)
# ---------------------------------------------------------------------------

def generate_caption(filepath, caption_model, caption_tokenizer, caption_device):
    """Generate a short caption for text-free images using Florence-2."""
    if caption_model is None:
        return ""
    try:
        with Image.open(filepath) as img:
            image = img.convert("RGB")
        prompt = "<CAPTION>"
        inputs = caption_tokenizer(text=prompt, images=image, return_tensors="pt").to(
            caption_device
        )
        with torch.inference_mode():
            generated = caption_model.generate(
                **inputs, max_new_tokens=64, do_sample=False
            )
        caption = caption_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        # Florence-2 may include task tokens; clean up
        if ":" in caption:
            caption = caption.split(":", 1)[-1].strip()
        return caption
    except Exception:
        return ""

# ---------------------------------------------------------------------------
# HELPER UTILITIES
# ---------------------------------------------------------------------------

def smart_preprocess_for_ocr(img):
    arr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    if sharpness > SHARPNESS_THRESHOLD:
        return img
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    return Image.fromarray(denoised)


def batch_upsert(items):
    for i in range(0, len(items), DB_BATCH_SIZE):
        chunk = items[i : i + DB_BATCH_SIZE]
        collection.upsert(
            ids=[c["filepath"] for c in chunk],
            embeddings=[c["embedding"] for c in chunk],
            metadatas=[c["metadata"] for c in chunk],
            documents=[c["content"] for c in chunk],
        )


def recursive_image_scan(folder_path):
    """Recursively find all supported images in folder and subfolders."""
    all_files = []
    for root, _dirs, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith(SUPPORTED_EXT):
                all_files.append(os.path.join(root, f))
    return all_files

# ---------------------------------------------------------------------------
# PHASE WORKERS
# ---------------------------------------------------------------------------

def worker_ocr(filepath):
    """Phase 1 -- phash + OCR. CPU-bound."""
    try:
        with Image.open(filepath) as img:
            phash = str(imagehash.phash(img))
            ocr_img = smart_preprocess_for_ocr(img)
            ocr_text = pytesseract.image_to_string(ocr_img).strip()
        return {"filepath": filepath, "phash": phash, "ocr_text": ocr_text, "error": None}
    except Exception as exc:
        return {"filepath": filepath, "error": str(exc)}

# ---------------------------------------------------------------------------
# MODEL WARM-UP
# ---------------------------------------------------------------------------

def preload_local_models():
    with st.status("Initializing CLIP model...", expanded=True) as status:
        status.write(f"Loading `{CLIP_MODEL_NAME}` into VRAM...")
        status.write("*(First run: ~1.7 GB download if not cached)*")
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            status.write(
                f"PyTorch OK . device = `{device}` . CUDA = `{torch.version.cuda}`"
            )
            _, _, dev = load_clip_model()
            status.write(f"Cached on `{dev}` (float16).")

            # Try to load caption model in background (non-blocking for core search)
            try:
                cm, ct, cd = load_caption_model()
                if cm is not None:
                    status.write(f"Captioning model (`{CAPTION_MODEL_NAME}`) ready.")
                else:
                    status.write("Captioning model unavailable (fallback OK).")
            except Exception:
                status.write("Captioning model skipped (search still works).")

            status.update(label="Model ready!", state="complete", expanded=False)
            return True
        except Exception as exc:
            import traceback
            status.update(label="Initialization failed", state="error")
            st.error(f"CLIP failed to load: {exc}")
            st.code(traceback.format_exc())
            return False

# ---------------------------------------------------------------------------
# STREAMLIT APP
# ---------------------------------------------------------------------------

st.title("Local Image Intelligence Portal")
st.caption(
    "CLIP GPU batch embeddings . Parallel OCR metadata . "
    "Florence-2 captioning . Hybrid semantic search"
)

if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False

if not st.session_state.models_loaded:
    if preload_local_models():
        st.session_state.models_loaded = True
        st.rerun()
    else:
        st.stop()

# ── Tesseract path override ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    current_tesseract = getattr(pytesseract.pytesseract, "tesseract_cmd", "")
    new_tesseract = st.text_input(
        "Tesseract path (auto-detected)",
        value=current_tesseract,
        help="Leave blank to use auto-detected path.",
    )
    if new_tesseract and new_tesseract != current_tesseract:
        if os.path.isfile(new_tesseract):
            pytesseract.pytesseract.tesseract_cmd = new_tesseract
            st.success("Tesseract path updated.")
        else:
            st.error("File not found at that path.")

    st.divider()

    # Model version info
    meta = _read_meta()
    stored_version = meta.get("model_version", "never")
    if stored_version == MODEL_VERSION:
        st.success(f"Model version: `{MODEL_VERSION}` (up to date)")
    else:
        st.warning(
            f"Stored version: `{stored_version}`, current: `{MODEL_VERSION}`. "
            "Your index may need re-embedding."
        )

tab1, tab2 = st.tabs(["Process Folder", "Semantic Search"])

# ==========================================================================
# TAB 1 -- PROCESS FOLDER
# ==========================================================================
with tab1:
    st.header("Ingest Images")

    # Model version warning
    if not check_model_version():
        with st.warning("Model version mismatch detected."):
            st.write(
                "The embedding model or pipeline has been updated since your "
                "last index was created. Old embeddings may be incompatible."
            )
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Force Re-Embed (Delete DB)", use_container_width=True):
                    try:
                        chroma_client.delete_collection("image_intelligence")
                        shutil.rmtree(DB_DIR, ignore_errors=True)
                        os.makedirs(DB_DIR, exist_ok=True)
                        # Re-create collection
                        global collection
                        collection = chroma_client.get_or_create_collection(
                            name="image_intelligence"
                        )
                        st.session_state.needs_rerun = True
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Failed to reset DB: {exc}")

    folder_path = st.text_input("Absolute path to image folder:")
    recursive = st.checkbox(
        "Include subfolders (recursive scan)", value=True, help="Default: ON"
    )
    enable_captioning = st.checkbox(
        "Generate captions for text-free images",
        value=True,
        help="Uses Florence-2 to generate short descriptions of visual content. Adds ~0.5s per image.",
    )

    if st.button("Process Folder"):
        if not folder_path or not os.path.exists(folder_path):
            st.error("Please enter a valid directory path.")
        else:
            if recursive:
                all_files = recursive_image_scan(folder_path)
                scan_desc = "images (including subfolders)"
            else:
                all_files = [
                    os.path.join(folder_path, f)
                    for f in os.listdir(folder_path)
                    if f.lower().endswith(SUPPORTED_EXT)
                ]
                scan_desc = "images (top-level only)"

            if not all_files:
                st.warning("No supported images found in that directory.")
            else:
                # Pre-flight: load existing DB state
                with st.spinner("Scanning database for already-indexed files..."):
                    db_data = collection.get(include=["metadatas", "embeddings"])
                    existing_ids = set(db_data.get("ids", []))
                    phash_cache = {}

                    ids = db_data.get("ids") or []
                    metas = db_data.get("metadatas") or []
                    embs = db_data.get("embeddings")
                    embs = embs if embs is not None else []

                    for fid, meta, emb in zip(ids, metas, embs):
                        if meta and "phash" in meta:
                            phash_cache[meta["phash"]] = {
                                "content": meta.get("content"),
                                "ocr_text": meta.get("ocr_text", ""),
                                "caption": meta.get("caption", ""),
                                "has_text": meta.get("has_text", False),
                                "embedding": emb,
                            }

                new_files = [f for f in all_files if f not in existing_ids]
                skipped_count = len(all_files) - len(new_files)

                st.info(
                    f"**{len(all_files)}** {scan_desc} found -- "
                    f"**{skipped_count}** already indexed, "
                    f"**{len(new_files)}** to process."
                )

                if not new_files:
                    st.success("Index is already up-to-date. Nothing to do!")

                lock = threading.Lock()
                phash_lock = threading.Lock()
                counters = {"processed": 0, "duplicates": 0, "errors": 0, "captions": 0}

                overall_bar = st.progress(0.0)
                phase_label = st.empty()
                stats_display = st.empty()
                total = len(new_files)

                def update_stats():
                    with lock:
                        p = counters["processed"]
                        d = counters["duplicates"]
                        e = counters["errors"]
                        c = counters.get("captions", 0)
                    stats_display.markdown(
                        f"Indexed: **{p}** . "
                        f"Duplicates: **{d}** . "
                        f"Captions: **{c}** . "
                        f"Errors: **{e}**"
                    )

                # ==================================================================
                # PHASE 1: Parallel OCR + phash (CPU, 8 threads)
                # ==================================================================
                phase_label.markdown(
                    "**Phase 1 / 3 -- OCR . Perceptual Hashing** (parallel CPU)"
                )

                ocr_results = []
                phase1_done = 0

                with ThreadPoolExecutor(max_workers=OCR_WORKERS) as pool:
                    futures = {pool.submit(worker_ocr, fp): fp for fp in new_files}
                    for fut in as_completed(futures):
                        result = fut.result()
                        phase1_done += 1
                        overall_bar.progress(phase1_done / total * 0.25)

                        if result.get("error"):
                            with lock:
                                counters["errors"] += 1
                            update_stats()
                            st.warning(
                                f"OCR error . {os.path.basename(result['filepath'])} "
                                f"-- {result['error']}"
                            )
                            continue
                        ocr_results.append(result)

                # ==================================================================
                # PHASE 1.5: Captioning for text-free images (optional, parallel)
                # ==================================================================
                if enable_captioning and ocr_results:
                    cm, ct, cd = load_caption_model()
                    if cm is not None:
                        phase_label.markdown(
                            "**Phase 1.5 -- Generating captions for text-free images...**"
                        )
                        caption_done = 0

                        def caption_worker(r):
                            txt = r.get("ocr_text", "")
                            if len(txt) >= HAS_TEXT_MIN_LEN:
                                return {"filepath": r["filepath"], "caption": "", "error": None}
                            cap = generate_caption(
                                r["filepath"], cm, ct, cd
                            )
                            return {"filepath": r["filepath"], "caption": cap, "error": None}

                        with ThreadPoolExecutor(max_workers=CAPTION_WORKERS) as pool:
                            cfutures = {
                                pool.submit(caption_worker, r): r for r in ocr_results
                            }
                            for fut in as_completed(cfutures):
                                cr = fut.result()
                                caption_done += 1
                                pct = 0.25 + (caption_done / len(ocr_results)) * 0.10
                                overall_bar.progress(pct)
                                # Merge caption back into ocr_results
                                for r in ocr_results:
                                    if r["filepath"] == cr["filepath"]:
                                        r["caption"] = cr.get("caption", "")
                                        if cr.get("caption"):
                                            with lock:
                                                counters["captions"] += 1
                                        break
                    else:
                        # No caption model -- attach empty captions
                        for r in ocr_results:
                            r.setdefault("caption", "")
                else:
                    for r in ocr_results:
                        r.setdefault("caption", "")

                # ==================================================================
                # PHASE 2: Split into duplicates vs. needs-embedding
                # ==================================================================
                phase_label.markdown("**Phase 2 / 3 -- Deduplicating via perceptual hash...**")

                items_to_embed = []
                duplicate_items = []

                for r in ocr_results:
                    phash = r["phash"]
                    with phash_lock:
                        cached = phash_cache.get(phash)

                    if cached:
                        duplicate_items.append({
                            "filepath": r["filepath"],
                            "phash": phash,
                            "content": cached["content"],
                            "embedding": cached["embedding"],
                            "ocr_text": cached["ocr_text"],
                            "caption": r.get("caption", ""),
                            "has_text": cached["has_text"],
                        })
                    else:
                        items_to_embed.append(r)

                # ==================================================================
                # PHASE 3: CLIP GPU batch IMAGE embedding -- every non-duplicate
                # For large batches, queue processing in chunks to manage VRAM
                # ==================================================================
                clip_model, clip_processor, clip_device = load_clip_model()
                results_to_write = []
                total_phase3 = len(items_to_embed)
                phase3_done_count = 0

                def record_result(filepath, phash, ocr_text, caption, embedding):
                    nonlocal phase3_done_count
                    phase3_done_count += 1
                    pct = 0.35 + (phase3_done_count / max(total_phase3, 1)) * 0.55
                    overall_bar.progress(min(pct, 0.90))

                    content = os.path.basename(filepath)
                    has_text = len(ocr_text) >= HAS_TEXT_MIN_LEN

                    results_to_write.append({
                        "filepath": filepath,
                        "phash": phash,
                        "content": content,
                        "embedding": embedding,
                        "ocr_text": ocr_text,
                        "caption": caption,
                        "has_text": has_text,
                    })
                    with phash_lock:
                        phash_cache[phash] = {
                            "content": content,
                            "ocr_text": ocr_text,
                            "caption": caption,
                            "has_text": has_text,
                            "embedding": embedding,
                        }
                    with lock:
                        counters["processed"] += 1
                    update_stats()

                if items_to_embed:
                    n_batches = (len(items_to_embed) + CLIP_BATCH_SIZE - 1) // CLIP_BATCH_SIZE
                    phase_label.markdown(
                        f"**Phase 3 / 3 -- CLIP image embedding** "
                        f"({len(items_to_embed)} images . {n_batches} GPU batches)"
                    )
                    for batch_start in range(0, len(items_to_embed), CLIP_BATCH_SIZE):
                        batch = items_to_embed[batch_start : batch_start + CLIP_BATCH_SIZE]
                        paths = [item["filepath"] for item in batch]
                        try:
                            embeddings = embed_images_batch(
                                paths, clip_model, clip_processor, clip_device
                            )
                            for item, emb in zip(batch, embeddings):
                                record_result(
                                    item["filepath"],
                                    item["phash"],
                                    item.get("ocr_text", ""),
                                    item.get("caption", ""),
                                    emb,
                                )
                        except Exception as exc:
                            for _item in batch:
                                with lock:
                                    counters["errors"] += 1
                            st.warning(f"Image embed batch failed: {exc}")
                            update_stats()

                # ==================================================================
                # PHASE 4: Batch ChromaDB writes
                # ==================================================================
                phase_label.markdown("**Flushing results to database...**")

                for r in results_to_write:
                    r["metadata"] = {
                        "filepath": r["filepath"],
                        "content": r["content"],
                        "phash": r["phash"],
                        "ocr_text": r["ocr_text"],
                        "caption": r.get("caption", ""),
                        "has_text": r["has_text"],
                    }

                duplicate_write_list = [
                    {
                        "filepath": d["filepath"],
                        "content": d["content"],
                        "embedding": d["embedding"],
                        "metadata": {
                            "filepath": d["filepath"],
                            "content": d["content"],
                            "phash": d["phash"],
                            "ocr_text": d["ocr_text"],
                            "caption": d.get("caption", ""),
                            "has_text": d["has_text"],
                        },
                    }
                    for d in duplicate_items
                ]

                all_to_write = results_to_write + duplicate_write_list

                with lock:
                    counters["duplicates"] += len(duplicate_items)

                if all_to_write:
                    batch_upsert(all_to_write)

                # Save model version on successful run
                save_model_version()

                overall_bar.progress(1.0)
                phase_label.empty()
                update_stats()

                with lock:
                    p = counters["processed"]
                    d = counters["duplicates"]
                    e = counters["errors"]
                    c = counters.get("captions", 0)

                st.success(
                    f"Done!\n\n"
                    f"- Newly indexed: **{p}**\n"
                    f"- Duplicates linked: **{d}**\n"
                    f"- Captions generated: **{c}**\n"
                    f"- Already indexed (skipped): **{skipped_count}**\n"
                    f"- Errors: **{e}**"
                )

# ==========================================================================
# TAB 2 -- SEMANTIC SEARCH
# ==========================================================================
with tab2:
    st.header("Semantic Search")

    with st.form(key="search_form"):
        search_query = st.text_input(
            "What are you looking for?",
            placeholder=(
                "e.g. 'dog', 'invoice receipt', 'outdoor landscape', "
                "'cat on a chair'"
            ),
        )
        col_a, col_b, col_c = st.columns([2, 1, 1])
        with col_a:
            min_score = st.slider(
                "Minimum relevance score",
                min_value=0.0, max_value=1.0, value=0.20, step=0.05,
            )
        with col_b:
            semantic_weight = st.slider(
                "Semantic weight",
                min_value=0.0, max_value=1.0, value=0.85, step=0.05,
                help=(
                    "CLIP semantic search is very strong -- 0.85+ recommended. "
                    "Keyword score now only comes from OCR text/filename/caption."
                ),
            )
        with col_c:
            candidate_pool = st.slider(
                "Candidate pool",
                min_value=10, max_value=200, value=50, step=10,
            )
        submitted = st.form_submit_button("Search", use_container_width=True)

    def keyword_score(query, *texts):
        haystack = " ".join(t for t in texts if t).lower()
        if not haystack:
            return 0.0
        q_tokens = set(query.lower().split())
        hits = sum(1 for t in q_tokens if t in haystack)
        return hits / max(len(q_tokens), 1)

    def hybrid_rerank(query, ids, metadatas, distances, semantic_w):
        keyword_w = 1.0 - semantic_w
        max_dist = max(distances) if distances else 1.0
        min_dist = min(distances) if distances else 0.0
        dist_range = max_dist - min_dist or 1.0
        ranked = []
        for fid, meta, dist in zip(ids, metadatas, distances):
            norm_dist = (dist - min_dist) / dist_range
            sem_score = 1.0 - norm_dist
            kw_score = keyword_score(
                query,
                meta.get("ocr_text", ""),
                os.path.basename(meta.get("filepath", "")),
                meta.get("caption", ""),
            )
            combined = semantic_w * sem_score + keyword_w * kw_score
            ranked.append({
                "filepath": fid,
                "metadata": meta,
                "sem_score": round(sem_score, 3),
                "kw_score": round(kw_score, 3),
                "combined": round(combined, 3),
            })
        ranked.sort(key=lambda x: x["combined"], reverse=True)
        return ranked

    if submitted and search_query.strip():
        try:
            total_indexed = collection.count()
            if total_indexed == 0:
                st.info("The index is empty -- process a folder first.")
                st.stop()

            n_candidates = min(candidate_pool, total_indexed)

            with st.spinner("Searching..."):
                clip_model, clip_processor, clip_device = load_clip_model()
                query_embedding = embed_text(
                    search_query.strip(),
                    clip_model,
                    clip_processor,
                    clip_device,
                )
                raw = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_candidates,
                    include=["metadatas", "distances"],
                )

            ids = raw["ids"][0] if raw["ids"] else []
            metas = raw["metadatas"][0] if raw["metadatas"] else []
            distances = raw["distances"][0] if raw["distances"] else []

            if not ids:
                st.info("No results found.")
                st.stop()

            ranked = hybrid_rerank(
                search_query, ids, metas, distances, semantic_weight
            )
            visible = [r for r in ranked if r["combined"] >= min_score]

            if not visible:
                st.warning(
                    f"Found **{len(ranked)}** candidates but none passed the "
                    f"**{min_score:.0%}** threshold. Try lowering the minimum score."
                )
                st.stop()

            st.success(
                f"**{len(visible)}** result{'s' if len(visible) != 1 else ''} "
                f"out of {n_candidates} candidates"
            )

            for r in visible:
                filepath = r["filepath"]
                meta = r["metadata"]
                with st.container():
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        if os.path.exists(filepath):
                            st.image(filepath, use_container_width=True)
                        else:
                            st.warning("File not found at stored path.")
                    with col2:
                        st.markdown(f"**Path:** `{filepath}`")
                        caption = meta.get("caption", "")
                        if caption:
                            st.markdown(f"**Caption:** {caption}")
                        if meta.get("has_text"):
                            ocr_preview = (meta.get("ocr_text", "") or "")[:120]
                            st.markdown(f"**OCR text:** {ocr_preview}")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Combined", f"{r['combined']:.0%}")
                        c2.metric("Semantic", f"{r['sem_score']:.0%}")
                        c3.metric("Keyword", f"{r['kw_score']:.0%}")
                    st.divider()

        except Exception as exc:
            st.error(f"Search failed: {exc}")

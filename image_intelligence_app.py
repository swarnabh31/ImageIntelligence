import os

# See video_intelligence_app.py for the full explanation — this env has a
# leftover TensorFlow install that conflicts with newer protobuf versions
# pulled in by other packages. This app only uses PyTorch, so skip TF/Flax
# auto-detection in transformers entirely.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

DB_DIR = os.path.join(os.getcwd(), "chroma_db")

# ── CLIP model ──────────────────────────────────────────────────────────────
# CLIP encodes images AND text into the same vector space — no captions, no
# LLM inference, no flash_attn. Just pure GPU batch embedding.
#
# clip-vit-large-patch14 → 768-dim, best quality  (~1.7 GB download)
# clip-vit-base-patch32  → 512-dim, faster/smaller (~600 MB download)
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"

# RTX 5090 (32 GB): push to 128 if you want. Throughput ~500-1000 images/sec.
CLIP_BATCH_SIZE = 64

# ── Parallelism knobs ────────────────────────────────────────────────────────
OCR_WORKERS   = 8
DB_BATCH_SIZE = 100

# ── Smart preprocessing ──────────────────────────────────────────────────────
SHARPNESS_THRESHOLD = 500.0
SUPPORTED_EXT       = (".png", ".jpg", ".jpeg", ".webp")

# Minimum OCR character count before we bother flagging "has_text" / using it
# for the keyword boost. This is METADATA ONLY now — it never decides which
# CLIP encoder an image goes through. Every image always gets
# get_image_features(). Always. OCR text (real or hallucinated noise from
# Tesseract on textless photos) just rides along as searchable metadata.
HAS_TEXT_MIN_LEN = 3

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
    """
    Load CLIP into VRAM once; Streamlit caches it across reruns.

    RTX 5090 throughput (clip-vit-large-patch14, float16, batch=64):
        ~500–1000 images/second
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
    if device == "cuda":
        model = model.half()   # float16 — halves VRAM, same quality
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    return model, processor, device

# ---------------------------------------------------------------------------
# EMBEDDING FUNCTIONS
# ---------------------------------------------------------------------------

def embed_images_batch(
    image_paths: list[str],
    model,
    processor,
    device: str,
) -> list[list[float]]:
    """
    GPU batch image embedding via CLIP image encoder.
    Returns L2-normalised 768-dim vectors — one per image.
    Bad files get a zero vector so batch indices stay aligned.

    This is the ONLY embedding path for indexed images. There is no branch
    that routes an image through the text encoder instead — that was the
    bug. OCR text never determines which encoder runs.
    """
    pil_images: list[Image.Image] = []
    for path in image_paths:
        try:
            with Image.open(path) as img:
                pil_images.append(img.convert("RGB").copy())
        except Exception:
            pil_images.append(Image.new("RGB", (224, 224), (128, 128, 128)))

    inputs = processor(
        images         = pil_images,
        return_tensors = "pt",
        padding        = True,
    ).to(device)

    with torch.inference_mode():
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().float().tolist()


def embed_text(text: str, model, processor, device: str) -> list[float]:
    """
    Single text embedding via CLIP text encoder.
    Used ONLY for the user's search query — same vector space as
    embed_images_batch, so "a cat sitting" finds cat photos directly.
    CLIP text tokenizer caps at 77 tokens; long queries are truncated.
    """
    inputs = processor(
        text           = [text],
        return_tensors = "pt",
        padding        = True,
        truncation     = True,
        max_length     = 77,
    ).to(device)

    with torch.inference_mode():
        features = model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)

    return features[0].cpu().float().tolist()

# ---------------------------------------------------------------------------
# HELPER UTILITIES
# ---------------------------------------------------------------------------

def smart_preprocess_for_ocr(img: Image.Image) -> Image.Image:
    arr       = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray      = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    if sharpness > SHARPNESS_THRESHOLD:
        return img
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    return Image.fromarray(denoised)


def batch_upsert(items: list):
    for i in range(0, len(items), DB_BATCH_SIZE):
        chunk = items[i : i + DB_BATCH_SIZE]
        collection.upsert(
            ids        = [c["filepath"]  for c in chunk],
            embeddings = [c["embedding"] for c in chunk],
            metadatas  = [c["metadata"]  for c in chunk],
            documents  = [c["content"]   for c in chunk],
        )

# ---------------------------------------------------------------------------
# PHASE WORKERS
# ---------------------------------------------------------------------------

def worker_ocr(filepath: str) -> dict:
    """Phase 1 — phash + OCR. CPU-bound, runs in ThreadPoolExecutor.
    OCR output here is METADATA. It is never used to choose an embedding
    encoder and never becomes the thing that gets embedded."""
    try:
        with Image.open(filepath) as img:
            phash    = str(imagehash.phash(img))
            ocr_img  = smart_preprocess_for_ocr(img)
            ocr_text = pytesseract.image_to_string(ocr_img).strip()
        return {"filepath": filepath, "phash": phash, "ocr_text": ocr_text, "error": None}
    except Exception as exc:
        return {"filepath": filepath, "error": str(exc)}

# ---------------------------------------------------------------------------
# MODEL WARM-UP
# ---------------------------------------------------------------------------

def preload_local_models() -> bool:
    with st.status("Initializing CLIP model…", expanded=True) as status:
        status.write(f"Loading `{CLIP_MODEL_NAME}` into VRAM…")
        status.write("*(First run: ~1.7 GB download if not cached)*")
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            status.write(f"PyTorch OK · device = `{device}` · CUDA = `{torch.version.cuda}`")
            _, _, dev = load_clip_model()
            status.write(f"✓ CLIP loaded on `{dev}` (float16).")
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
st.caption("CLIP GPU batch embeddings (images only) · Parallel OCR metadata · Hybrid semantic search")

if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False

if not st.session_state.models_loaded:
    if preload_local_models():
        st.session_state.models_loaded = True
        st.rerun()
    else:
        st.stop()

tab1, tab2 = st.tabs(["Process Folder", "Semantic Search"])

# ==========================================================================
# TAB 1 — PROCESS FOLDER
# ==========================================================================
with tab1:
    st.header("Ingest Images")

    st.warning(
        "**Re-indexing required after this update.** The previous version "
        "embedded some images using the CLIP *text* encoder on OCR output "
        "instead of the image encoder, which silently discarded their "
        "visual content. Delete the `chroma_db` folder and re-process your "
        "folders so every image gets a real image embedding."
    )

    folder_path = st.text_input("Absolute path to image folder:")

    if st.button("Process Folder"):
        if not folder_path or not os.path.exists(folder_path):
            st.error("Please enter a valid directory path.")
        else:
            all_files = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(SUPPORTED_EXT)
            ]

            if not all_files:
                st.warning("No supported images found in that directory.")

            # Pre-flight: load existing DB state
            with st.spinner("Scanning database for already-indexed files…"):
                db_data      = collection.get(include=["metadatas", "embeddings"])
                existing_ids = set(db_data.get("ids", []))
                phash_cache  = {}

                ids   = db_data.get("ids") or []
                metas = db_data.get("metadatas") or []
                embs  = db_data.get("embeddings")
                embs  = embs if embs is not None else []

                for fid, meta, emb in zip(ids, metas, embs):
                    if meta and "phash" in meta:
                        phash_cache[meta["phash"]] = {
                            "content"  : meta.get("content"),
                            "ocr_text" : meta.get("ocr_text", ""),
                            "has_text" : meta.get("has_text", False),
                            "embedding": emb,
                        }

            new_files     = [f for f in all_files if f not in existing_ids]
            skipped_count = len(all_files) - len(new_files)

            st.info(
                f"**{len(all_files)}** images found — "
                f"**{skipped_count}** already indexed, "
                f"**{len(new_files)}** to process."
            )

            if not new_files:
                st.success("Index is already up-to-date. Nothing to do!")

            lock       = threading.Lock()
            phash_lock = threading.Lock()
            counters   = {"processed": 0, "duplicates": 0, "errors": 0}

            overall_bar   = st.progress(0.0)
            phase_label   = st.empty()
            stats_display = st.empty()
            total         = len(new_files)

            def update_stats():
                with lock:
                    p = counters["processed"]
                    d = counters["duplicates"]
                    e = counters["errors"]
                stats_display.markdown(
                    f"🆕 Newly indexed: **{p}** &nbsp;|&nbsp; "
                    f"🔁 Duplicates linked: **{d}** &nbsp;|&nbsp; "
                    f"❌ Errors: **{e}**"
                )

            # ==================================================================
            # PHASE 1: Parallel OCR + phash  (CPU, 8 threads)
            # ==================================================================
            phase_label.markdown(
                "**Phase 1 / 3 — OCR · Perceptual Hashing** (parallel CPU)"
            )

            ocr_results = []
            phase1_done = 0

            with ThreadPoolExecutor(max_workers=OCR_WORKERS) as pool:
                futures = {pool.submit(worker_ocr, fp): fp for fp in new_files}
                for fut in as_completed(futures):
                    result       = fut.result()
                    phase1_done += 1
                    overall_bar.progress(phase1_done / total * 0.30)

                    if result.get("error"):
                        with lock:
                            counters["errors"] += 1
                        update_stats()
                        st.warning(
                            f"OCR error · {os.path.basename(result['filepath'])} "
                            f"— {result['error']}"
                        )
                        continue
                    ocr_results.append(result)

            # ==================================================================
            # PHASE 2: Split into duplicates vs. needs-embedding
            # No text/visual branching here anymore — OCR length never
            # decides the embedding path. It only decides the phash dedup
            # lookup, exactly as before.
            # ==================================================================
            phase_label.markdown("**Phase 2 / 3 — Deduplicating via perceptual hash…**")

            items_to_embed  = []
            duplicate_items = []

            for r in ocr_results:
                phash = r["phash"]
                with phash_lock:
                    cached = phash_cache.get(phash)

                if cached:
                    duplicate_items.append({
                        "filepath" : r["filepath"],
                        "phash"    : phash,
                        "content"  : cached["content"],
                        "embedding": cached["embedding"],
                        "ocr_text" : cached["ocr_text"],
                        "has_text" : cached["has_text"],
                    })
                else:
                    items_to_embed.append(r)

            # ==================================================================
            # PHASE 3: CLIP GPU batch IMAGE embedding — every non-duplicate
            # image, no exceptions. OCR text is attached as metadata only.
            # ==================================================================
            clip_model, clip_processor, clip_device = load_clip_model()
            results_to_write = []
            total_phase3     = len(items_to_embed)
            phase3_done      = {"count": 0}

            def record_result(filepath, phash, ocr_text, embedding):
                phase3_done["count"] += 1
                pct = 0.30 + (phase3_done["count"] / max(total_phase3, 1)) * 0.60
                overall_bar.progress(min(pct, 0.90))

                content  = os.path.basename(filepath)
                has_text = len(ocr_text) >= HAS_TEXT_MIN_LEN

                results_to_write.append({
                    "filepath" : filepath,
                    "phash"    : phash,
                    "content"  : content,
                    "embedding": embedding,
                    "ocr_text" : ocr_text,
                    "has_text" : has_text,
                })
                with phash_lock:
                    phash_cache[phash] = {
                        "content"  : content,
                        "ocr_text" : ocr_text,
                        "has_text" : has_text,
                        "embedding": embedding,
                    }
                with lock:
                    counters["processed"] += 1
                update_stats()

            if items_to_embed:
                n_batches = (len(items_to_embed) + CLIP_BATCH_SIZE - 1) // CLIP_BATCH_SIZE
                phase_label.markdown(
                    f"**Phase 3 / 3 — CLIP image embedding** "
                    f"({len(items_to_embed)} images · {n_batches} GPU batches)"
                )
                for batch_start in range(0, len(items_to_embed), CLIP_BATCH_SIZE):
                    batch = items_to_embed[batch_start : batch_start + CLIP_BATCH_SIZE]
                    paths = [item["filepath"] for item in batch]
                    try:
                        embeddings = embed_images_batch(paths, clip_model, clip_processor, clip_device)
                        for item, emb in zip(batch, embeddings):
                            record_result(item["filepath"], item["phash"], item["ocr_text"], emb)
                    except Exception as exc:
                        for item in batch:
                            with lock:
                                counters["errors"] += 1
                        st.warning(f"Image embed batch failed: {exc}")
                        update_stats()

            # ==================================================================
            # PHASE 4: Batch ChromaDB writes
            # ==================================================================
            phase_label.markdown("**Flushing results to database…**")

            for r in results_to_write:
                r["metadata"] = {
                    "filepath": r["filepath"],
                    "content" : r["content"],
                    "phash"   : r["phash"],
                    "ocr_text": r["ocr_text"],
                    "has_text": r["has_text"],
                }

            duplicate_write_list = [
                {
                    "filepath" : d["filepath"],
                    "content"  : d["content"],
                    "embedding": d["embedding"],
                    "metadata" : {
                        "filepath": d["filepath"],
                        "content" : d["content"],
                        "phash"   : d["phash"],
                        "ocr_text": d["ocr_text"],
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

            overall_bar.progress(1.0)
            phase_label.empty()
            update_stats()

            with lock:
                p = counters["processed"]
                d = counters["duplicates"]
                e = counters["errors"]

            st.success(
                f"✅ Done!\n\n"
                f"- 🆕 Newly indexed: **{p}**\n"
                f"- 🔁 Duplicates linked: **{d}**\n"
                f"- ⏭️ Already indexed (skipped): **{skipped_count}**\n"
                f"- ❌ Errors: **{e}**"
            )

# ==========================================================================
# TAB 2 — SEMANTIC SEARCH
# Query → CLIP text encoder. Index → CLIP image encoder, always.
# Same space, genuine cross-modal search, no silent fallback to text-text.
# ==========================================================================
with tab2:
    st.header("Semantic Search")

    with st.form(key="search_form"):
        search_query = st.text_input(
            "What are you looking for?",
            placeholder="e.g. 'dog', 'invoice receipt', 'outdoor landscape', 'cat on a chair'",
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
                help="CLIP semantic search is very strong — 0.85+ recommended. "
                     "Keyword score now only comes from OCR text/filename, so "
                     "it's a light boost, not a primary signal.",
            )
        with col_c:
            candidate_pool = st.slider(
                "Candidate pool",
                min_value=10, max_value=200, value=50, step=10,
            )
        submitted = st.form_submit_button("🔍 Search", use_container_width=True)

    def keyword_score(query: str, *texts: str) -> float:
        haystack = " ".join(t for t in texts if t).lower()
        if not haystack:
            return 0.0
        q_tokens = set(query.lower().split())
        hits     = sum(1 for t in q_tokens if t in haystack)
        return hits / max(len(q_tokens), 1)

    def hybrid_rerank(query, ids, metadatas, distances, semantic_w):
        # All candidates are now CLIP image embeddings — one consistent
        # modality — so a single global min/max normalization is valid
        # again (no text-embedded items mixed into the pool to skew it).
        keyword_w  = 1.0 - semantic_w
        max_dist   = max(distances) if distances else 1.0
        min_dist   = min(distances) if distances else 0.0
        dist_range = max_dist - min_dist or 1.0
        ranked = []
        for fid, meta, dist in zip(ids, metadatas, distances):
            norm_dist = (dist - min_dist) / dist_range
            sem_score = 1.0 - norm_dist
            kw_score  = keyword_score(
                query,
                meta.get("ocr_text", ""),
                os.path.basename(meta.get("filepath", "")),
            )
            combined  = semantic_w * sem_score + keyword_w * kw_score
            ranked.append({
                "filepath" : fid,
                "metadata" : meta,
                "sem_score": round(sem_score, 3),
                "kw_score" : round(kw_score, 3),
                "combined" : round(combined, 3),
            })
        ranked.sort(key=lambda x: x["combined"], reverse=True)
        return ranked

    if submitted and search_query.strip():
        try:
            total_indexed = collection.count()
            if total_indexed == 0:
                st.info("The index is empty — process a folder first.")
                st.stop()

            n_candidates = min(candidate_pool, total_indexed)

            with st.spinner("Searching…"):
                clip_model, clip_processor, clip_device = load_clip_model()
                query_embedding = embed_text(
                    search_query.strip(), clip_model, clip_processor, clip_device
                )
                raw = collection.query(
                    query_embeddings = [query_embedding],
                    n_results        = n_candidates,
                    include          = ["metadatas", "distances"],
                )

            ids       = raw["ids"][0]       if raw["ids"]       else []
            metas     = raw["metadatas"][0] if raw["metadatas"] else []
            distances = raw["distances"][0] if raw["distances"] else []

            if not ids:
                st.info("No results found.")
                st.stop()

            ranked  = hybrid_rerank(search_query, ids, metas, distances, semantic_weight)
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
                meta     = r["metadata"]
                with st.container():
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        if os.path.exists(filepath):
                            st.image(filepath, use_container_width=True)
                        else:
                            st.warning("File not found at stored path.")
                    with col2:
                        st.markdown(f"**Path:** `{filepath}`")
                        if meta.get("has_text"):
                            ocr_preview = (meta.get("ocr_text", "") or "")[:120]
                            st.markdown(f"**OCR text:** {ocr_preview}")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Combined", f"{r['combined']:.0%}")
                        c2.metric("Semantic", f"{r['sem_score']:.0%}")
                        c3.metric("Keyword",  f"{r['kw_score']:.0%}")
                    st.divider()

        except Exception as exc:
            st.error(f"Search failed: {exc}")

#!/usr/bin/env python3
"""
Computes true 768-dimensional google/embeddinggemma-2 embeddings
for all 31,778 records (2,512 eResearch + 26,987 NeurIPS + 2,279 Supercomputing),
compatible 1-to-1 with in-browser Transformers.js v4 (`onnx-community/embeddinggemma-2-ONNX`).

Also refreshes `corpus-semantic/corpus_data.js` using the LLM-cleaned eResearch dataset
(`../data/corpus_cleaned.json`) alongside `./neurips_index.json` and `./sc_index.json`.

Quantizes the L2-normalized 768-D vectors to Int8 [-127, 127] and writes them to
`corpus-semantic/gemma_embeddings.js` so `index-gemma.html` loads all 31,778 vectors
and performs live in-browser neural query embedding via @huggingface/transformers.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ERES_CLEANED_PATH = os.path.join(HERE, "..", "data", "corpus_cleaned.json")
NEURIPS_PATH = os.path.join(HERE, "neurips_index.json")
SC_PATH = os.path.join(HERE, "sc_index.json")
CORPUS_DATA_JS_PATH = os.path.join(HERE, "corpus_data.js")
OUT_EMBEDDINGS_PATH = os.path.join(HERE, "gemma_embeddings.js")

MODEL_ID = "google/embeddinggemma-2"
ONNX_MODEL_ID = "onnx-community/embeddinggemma-2-ONNX"
GEMMA_DIM = 768
MAX_SEQ_LENGTH = 128


def format_presenter(rec: dict) -> str:
    """Format cleaned author and institution metadata into a concise presenter string."""
    authors = [a.strip() for a in (rec.get("authors_clean") or []) if a and a.strip()]
    insts = [i.strip() for i in (rec.get("institutions_clean") or []) if i and i.strip()]
    raw_pres = (rec.get("presenters") or "").strip()

    if authors and insts:
        return f"{', '.join(authors)} ({', '.join(insts)})"
    if authors:
        return ", ".join(authors)
    if insts and raw_pres:
        return f"{raw_pres} ({', '.join(insts)})"
    if insts:
        return ", ".join(insts)
    return raw_pres


def update_corpus_data_js(eres_cleaned: list, nips: list, sc: list) -> list:
    """Update SEARCH_INDEX, NEURIPS_INDEX, and SC_INDEX in corpus_data.js from cleaned sources."""
    bundle_obj = None
    existing_paradigms = {}

    if os.path.exists(CORPUS_DATA_JS_PATH):
        with open(CORPUS_DATA_JS_PATH, encoding="utf-8") as f:
            for line in f:
                if line.startswith("const BUNDLE = "):
                    bundle_obj = json.loads(line[len("const BUNDLE = "):].rstrip(";\n"))
                elif line.startswith("const SEARCH_INDEX = "):
                    old_si = json.loads(line[len("const SEARCH_INDEX = "):].rstrip(";\n"))
                    for idx, item in enumerate(old_si):
                        existing_paradigms[idx] = item.get("c", 0)
                        if item.get("t"):
                            existing_paradigms[(item.get("y"), item.get("t"))] = item.get("c", 0)

    search_index = []
    for idx, r in enumerate(eres_cleaned):
        title = (r.get("title") or "").strip()
        year = int(r.get("year") or 0)
        abstract = (r.get("abstract") or "").strip()
        text = (r.get("text") or f"{title}. {abstract}").strip()
        l250 = " ".join(text.split()[:250])
        p_str = format_presenter(r)
        authors_str = " ".join(r.get("authors_clean") or [])
        insts_str = " ".join(r.get("institutions_clean") or [])
        ft = f"{title} {abstract} {authors_str} {insts_str} {r.get('presenters', '')}".lower()
        c_idx = existing_paradigms.get((year, title), existing_paradigms.get(idx, 0))

        search_index.append({
            "y": year,
            "t": title,
            "p": p_str,
            "c": c_idx,
            "w": int(r.get("abstract_words") or len(abstract.split())),
            "k": r.get("kind") or "abstract",
            "s": r.get("source") or "",
            "url": r.get("url") or "",
            "l250": l250,
            "ft": ft,
        })

    print(f"Updating {CORPUS_DATA_JS_PATH} with {len(search_index)} cleaned eResearch records...")
    with open(CORPUS_DATA_JS_PATH, "w", encoding="utf-8") as f:
        if bundle_obj is not None:
            f.write("const BUNDLE = " + json.dumps(bundle_obj, ensure_ascii=False) + ";\n")
        f.write("const SEARCH_INDEX = " + json.dumps(search_index, ensure_ascii=False) + ";\n")
        f.write("const NEURIPS_INDEX = " + json.dumps(nips, ensure_ascii=False) + ";\n")
        f.write("const SC_INDEX = " + json.dumps(sc, ensure_ascii=False) + ";\n")

    return search_index


def encode_with_sentence_transformers(all_texts: list, batch_size: int = 256) -> np.ndarray:
    """Encode texts using SentenceTransformer('google/embeddinggemma-2') (270M text-only encoder)."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    if "embedding_gemma2" not in CONFIG_MAPPING:
        raise RuntimeError("Installed transformers version does not register 'embedding_gemma2' (requires >= 5.18.0)")

    from sentence_transformers import SentenceTransformer

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    model = SentenceTransformer(
        MODEL_ID,
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"torch_dtype": dtype},
        truncate_dim=GEMMA_DIM,
    )
    model.max_seq_length = MAX_SEQ_LENGTH
    return model.encode(
        all_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def encode_with_onnx(all_texts: list, batch_size: int = 32, onnx_file: str = "onnx/model_q4.onnx") -> np.ndarray:
    """
    Encode texts using onnx-community/embeddinggemma-2-ONNX via ONNX Runtime.
    Produces embeddings identical to the in-browser Transformers.js v4 model in index-gemma.html,
    and works even when transformers < 5.18.0 is installed.
    """
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    local_onnx_dir = os.path.expanduser("~/.cache/huggingface/embeddinggemma-2-onnx")
    tok_path = os.path.join(local_onnx_dir, "tokenizer.json")
    model_path = os.path.join(local_onnx_dir, onnx_file)
    data_path = os.path.join(local_onnx_dir, f"{onnx_file}_data")

    if not (os.path.exists(tok_path) and os.path.exists(model_path) and os.path.exists(data_path)):
        print(f"Downloading {ONNX_MODEL_ID} ({onnx_file}) to {local_onnx_dir}...")
        tok_path = hf_hub_download(ONNX_MODEL_ID, "tokenizer.json", local_dir=local_onnx_dir)
        hf_hub_download(ONNX_MODEL_ID, f"{onnx_file}_data", local_dir=local_onnx_dir)
        model_path = hf_hub_download(ONNX_MODEL_ID, onnx_file, local_dir=local_onnx_dir)
    else:
        print(f"Loading cached {ONNX_MODEL_ID} ({onnx_file}) from {local_onnx_dir}...")

    tokenizer = Tokenizer.from_file(tok_path)
    pad_id = tokenizer.token_to_id("<pad>") or 0
    tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>")
    tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)

    cpu_cores = os.cpu_count() or 8
    workers = min(8, max(1, cpu_cores // 2))
    threads_per_worker = max(1, cpu_cores // workers)

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = threads_per_worker
    sess_opts.inter_op_num_threads = 1
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])

    input_names = {inp.name for inp in session.get_inputs()}
    output_names = [out.name for out in session.get_outputs()]
    emb_out_name = "sentence_embedding" if "sentence_embedding" in output_names else output_names[-1]
    empty_mm = np.zeros((0, 512), dtype=np.float32)

    # Sort by text length to minimize padding overhead across 31,778 records, then restore original order
    n_total = len(all_texts)
    sorted_indices = np.argsort([len(t) for t in all_texts])
    all_vecs = np.zeros((n_total, GEMMA_DIM), dtype=np.float32)

    batches = [
        sorted_indices[start:start + batch_size]
        for start in range(0, n_total, batch_size)
    ]

    def run_batch(batch_idx):
        batch_texts = [all_texts[i] for i in batch_idx]
        encodings = tokenizer.encode_batch(batch_texts)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        feed = {"input_ids": input_ids}
        if "attention_mask" in input_names:
            feed["attention_mask"] = attention_mask
        if "image_features" in input_names:
            feed["image_features"] = empty_mm
        if "video_features" in input_names:
            feed["video_features"] = empty_mm
        if "audio_features" in input_names:
            feed["audio_features"] = empty_mm

        emb = session.run([emb_out_name], feed)[0]
        emb = emb[:, :GEMMA_DIM].astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return batch_idx, (emb / norms)

    t_start = time.time()
    completed_docs = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for step, (batch_idx, norm_emb) in enumerate(executor.map(run_batch, batches), 1):
            all_vecs[batch_idx] = norm_emb
            completed_docs += len(batch_idx)
            if (step % 50 == 0) or (completed_docs == n_total):
                elapsed = time.time() - t_start
                rate = completed_docs / max(elapsed, 1e-6)
                eta = (n_total - completed_docs) / max(rate, 1e-6)
                print(
                    f"  Encoded {completed_docs:,}/{n_total:,} ({100.0 * completed_docs / n_total:.1f}%) "
                    f"— {rate:.1f} docs/s, ETA {eta:.0f}s",
                    flush=True,
                )

    return all_vecs


def main():
    parser = argparse.ArgumentParser(description="Build 768-D EmbeddingGemma 2 embeddings for corpus-semantic.")
    parser.add_argument(
        "--backend",
        choices=["auto", "sentence-transformers", "onnx"],
        default="auto",
        help="Backend to run EmbeddingGemma 2 ('auto' tries SentenceTransformer then ONNX Runtime).",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Encoding batch size.")
    args = parser.parse_args()

    t0 = time.time()
    print("Loading corpus JSON datasets...")
    with open(ERES_CLEANED_PATH, encoding="utf-8") as f:
        eres_cleaned = json.load(f)
    with open(NEURIPS_PATH, encoding="utf-8") as f:
        nips = json.load(f)
    with open(SC_PATH, encoding="utf-8") as f:
        sc = json.load(f)

    print(f"Loaded {len(eres_cleaned)} cleaned eResearch, {len(nips)} NeurIPS, {len(sc)} SC records.")

    # Update corpus_data.js with cleaned eResearch metadata + local NeurIPS & SC indices
    eres_index = update_corpus_data_js(eres_cleaned, nips, sc)

    # Prepare texts using EmbeddingGemma 2's canonical document prompt format:
    # "title: {title} | text: {content}"
    eres_texts = []
    for r in eres_cleaned:
        title = (r.get("title") or "none").strip()
        abstract_lead = " ".join((r.get("abstract") or "").split()[:100]).strip()
        eres_texts.append(f"title: {title} | text: {abstract_lead}")

    # NeurIPS: Paper Title formatted as EmbeddingGemma 2 document
    nips_texts = [f"title: {item[1]} | text: " for item in nips]
    # Supercomputing: Paper Title formatted as EmbeddingGemma 2 document
    sc_texts = [f"title: {item[1]} | text: " for item in sc]

    all_texts = eres_texts + nips_texts + sc_texts
    print(f"Total documents to embed with {MODEL_ID}: {len(all_texts)}")

    t_enc = time.time()
    if args.backend == "onnx":
        all_vecs = encode_with_onnx(all_texts, batch_size=args.batch_size)
    elif args.backend == "sentence-transformers":
        all_vecs = encode_with_sentence_transformers(all_texts, batch_size=args.batch_size)
    else:
        try:
            all_vecs = encode_with_sentence_transformers(all_texts, batch_size=args.batch_size)
        except Exception as exc:
            print(f"SentenceTransformer({MODEL_ID!r}) unavailable ({exc}); using ONNX Runtime ({ONNX_MODEL_ID})...")
            all_vecs = encode_with_onnx(all_texts, batch_size=args.batch_size)

    print(f"Encoded all {len(all_texts)} documents into {all_vecs.shape} in {time.time() - t_enc:.1f}s!")

    # Quantize 768-D L2-normalized vectors to Int8 [-127, 127]
    n_eres = len(eres_index)
    n_nips = len(nips)
    n_sc = len(sc)

    doc_int8 = np.clip(np.round(all_vecs * 127.0), -127, 127).astype(np.int8)
    eres_b64 = base64.b64encode(doc_int8[:n_eres].tobytes()).decode("ascii")
    nips_b64 = base64.b64encode(doc_int8[n_eres:n_eres + n_nips].tobytes()).decode("ascii")
    sc_b64 = base64.b64encode(doc_int8[n_eres + n_nips:].tobytes()).decode("ascii")

    print(f"Writing {GEMMA_DIM}-D {MODEL_ID} embeddings to {OUT_EMBEDDINGS_PATH}...")
    with open(OUT_EMBEDDINGS_PATH, "w", encoding="utf-8") as f:
        f.write(f"// Pre-computed {GEMMA_DIM}-D {MODEL_ID} Document Embeddings (31,778 Records)\n")
        f.write("// Paired with in-browser @huggingface/transformers (onnx-community/embeddinggemma-2-ONNX) live query inference\n")
        f.write(f"const GEMMA_DIM = {GEMMA_DIM};\n")
        f.write(f"const ERES_GEMMA_B64 = \"{eres_b64}\";\n")
        f.write(f"const NIPS_GEMMA_B64 = \"{nips_b64}\";\n")
        f.write(f"const SC_GEMMA_B64 = \"{sc_b64}\";\n")

    sz_mb = os.path.getsize(OUT_EMBEDDINGS_PATH) / (1024 * 1024)
    print(f"Done in {time.time() - t0:.1f}s! {os.path.basename(OUT_EMBEDDINGS_PATH)} size: {sz_mb:.2f} MB")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Computes true 384-dimensional sentence-transformers/all-MiniLM-L6-v2 embeddings
for all 31,778 records (2,512 eResearch + 26,987 NeurIPS + 2,279 Supercomputing),
compatible 1-to-1 with in-browser Transformers.js (`Xenova/all-MiniLM-L6-v2`,
pooling='mean', normalize=True), just like SemanticFinder (https://github.com/do-me/SemanticFinder).

Quantizes the L2-normalized 384-D vectors to Int8 [-127, 127] and writes them to
corpus-semantic/minilm_embeddings.js so the browser loads all 31,778 vectors in <40ms
and performs live in-browser neural query embedding via @xenova/transformers.
"""
import base64
import json
import os
import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus")
OUT_DIR = os.path.dirname(__file__)


def main():
    t0 = time.time()
    print("Loading corpus JSON datasets...")
    with open(os.path.join(SRC_DIR, "search_index.json"), encoding="utf-8") as f:
        eres = json.load(f)
    with open(os.path.join(SRC_DIR, "neurips_index.json"), encoding="utf-8") as f:
        nips = json.load(f)
    with open(os.path.join(SRC_DIR, "sc_index.json"), encoding="utf-8") as f:
        sc = json.load(f)

    print(f"Loaded {len(eres)} eResearch, {len(nips)} NeurIPS, {len(sc)} SC records.")

    # Prepare texts for all-MiniLM-L6-v2
    # eResearch: Title + Lead-250 words (all-MiniLM-L6-v2 max_seq_length is 256 tokens)
    eres_texts = [f"{r.get('t', '')}. {r.get('l250', '')}" for r in eres]
    # NeurIPS: Paper Title
    nips_texts = [item[1] for item in nips]
    # Supercomputing: Paper Title
    sc_texts = [item[1] for item in sc]

    all_texts = eres_texts + nips_texts + sc_texts
    print(f"Total documents to embed with sentence-transformers/all-MiniLM-L6-v2: {len(all_texts)}")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
    model.max_seq_length = 128

    t_enc = time.time()
    all_vecs = model.encode(
        all_texts,
        batch_size=512,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    print(f"Encoded all {len(all_texts)} documents into {all_vecs.shape} in {time.time() - t_enc:.1f}s!")

    # Also encode a compact fallback dictionary of common query phrases & words
    # so initial page render before WebAssembly finishes initializing has instant 384-D vectors
    # common_queries = [
    #     "data mining", "machine learning", "deep learning", "artificial intelligence",
    #     "large language model", "llm", "generative ai", "agentic workflow", "rag",
    #     "transformer", "diffusion model", "reinforcement learning", "neural network",
    #     "graph neural network", "computer vision", "optimization", "bayesian",
    #     "cloud computing", "nectar", "research cloud", "virtual machine", "kubernetes",
    #     "high performance computing", "hpc", "supercomputing", "exascale", "gpu", "cuda",
    #     "mpi", "parallel storage", "lustre", "file system", "workflow", "pipeline",
    #     "container", "docker", "singularity", "jupyter", "notebook", "reproducibility",
    #     "research software engineer", "rse", "carpentries", "training", "skills",
    #     "fair", "fair data", "metadata", "persistent identifier", "pid", "orcid", "doi", "raid",
    #     "ands", "ardc", "data commons", "research data", "repository", "open science",
    #     "indigenous", "indigenous data sovereignty", "care principles",
    #     "trust", "trusted research environment", "five safes", "sensitive data", "privacy", "cybersecurity",
    #     "quantum", "quantum computing", "climate", "weather", "ocean", "earth science",
    #     "genomics", "bioinformatics", "alphafold", "astronomy", "radio astronomy", "digital twin",
    #     "visualisation", "virtual reality"
    # ]
    # q_vecs = model.encode(common_queries, normalize_embeddings=True)
    # fallback_q_map = {}
    # for q_str, q_v in zip(common_queries, q_vecs):
    #     q_int8 = np.clip(np.round(q_v * 127.0), -127, 127).astype(np.int8)
    #     fallback_q_map[q_str] = base64.b64encode(q_int8.tobytes()).decode("ascii")

    # Quantize 384-D L2-normalized vectors to Int8 [-127, 127]
    n_eres = len(eres)
    n_nips = len(nips)
    n_sc = len(sc)

    doc_int8 = np.clip(np.round(all_vecs * 127.0), -127, 127).astype(np.int8)
    eres_b64 = base64.b64encode(doc_int8[:n_eres].tobytes()).decode("ascii")
    nips_b64 = base64.b64encode(doc_int8[n_eres:n_eres + n_nips].tobytes()).decode("ascii")
    sc_b64 = base64.b64encode(doc_int8[n_eres + n_nips:].tobytes()).decode("ascii")

    out_path = os.path.join(OUT_DIR, "semantic_engine.js")
    print(f"Writing 384-D all-MiniLM-L6-v2 embeddings to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("// Pre-computed 384-D sentence-transformers/all-MiniLM-L6-v2 Embeddings (31,778 Records)\n")
        f.write("// Paired with in-browser @xenova/transformers (Xenova/all-MiniLM-L6-v2) live query inference\n")
        f.write("const MINILM_DIM = 384;\n")
        # f.write(f"const MINILM_FALLBACK_QUERIES = {json.dumps(fallback_q_map)};\n")
        f.write(f"const ERES_MINILM_B64 = \"{eres_b64}\";\n")
        f.write(f"const NIPS_MINILM_B64 = \"{nips_b64}\";\n")
        f.write(f"const SC_MINILM_B64 = \"{sc_b64}\";\n")

    sz_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Done in {time.time() - t0:.1f}s! semantic_engine.js size: {sz_mb:.2f} MB")


if __name__ == "__main__":
    main()

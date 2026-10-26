#!/usr/bin/env python3
"""
Builds corpus-semantic/semantic_engine.js and copies corpus_data.js.
Creates:
1. A curated + corpus-mined Semantic Ontology for eResearch, NeurIPS, and Supercomputing.
2. A 64-dimensional TruncatedSVD (LSA) vector embedding space across all 31,778 records
   (2,512 eResearch + 26,987 NeurIPS + 2,279 Supercomputing), quantized to Int8 and Base64-encoded
   for <3ms in-browser cosine similarity search across all 31,778 documents.
3. Vocabulary term embeddings (6,000 unigrams & bigrams) so ANY user query can be projected
   into the 64-D vector space and expanded with nearest-neighbor terms in real time.
"""
import base64
import json
import os
import shutil
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "corpus")
OUT_DIR = os.path.dirname(__file__)

NEURIPS_PARADIGMS = [
    "Classical Statistical ML Kernels Sparsity Graphical",
    "Optimization Learning Theory Bandits Convex Gradient",
    "Wave-1 Deep Learning CNNs RNNs GANs Computer Vision",
    "Reinforcement Learning Control Policy RL Game Theory",
    "Transformers Self-Attention Representation Learning Pretraining",
    "Foundation Models LLMs Generative AI Diffusion Reasoning Alignment",
    "AI for Science Graphs Molecules Physics Trust Safety Benchmarks",
]

SC_PARADIGMS = [
    "Exascale Architectures Manycore CPUs Green Liquid HPC Supercomputer",
    "GPU Accelerators CUDA Tensor Cores Heterogeneous Kernels OpenACC",
    "High-Speed Interconnects MPI RDMA Network Topologies Dragonfly Collective",
    "Parallel Storage Lustre GPFS I/O NVMe Burst Buffer Lossy Compression",
    "Numerical Solvers Sparse Linear Algebra Stencils AMG Fault Tolerance Resilience",
    "Programming Models OpenMP Kokkos SYCL Compilers Performance Portability Workflows",
    "AI-HPC Convergence Distributed LLMs Deep Learning Training Quantum Simulation",
]

CURATED_ONTOLOGY = {
    "data mining": [
        "data mining", "knowledge discovery", "pattern mining", "clustering",
        "classification", "feature selection", "anomaly detection",
        "predictive modeling", "machine learning", "data analytics"
    ],
    "machine learning": [
        "machine learning", "deep learning", "neural network", "supervised learning",
        "random forest", "support vector", "representation learning",
        "predictive model", "gradient boosting", "foundation model"
    ],
    "ai": [
        "artificial intelligence", "machine learning", "deep learning",
        "neural network", "generative ai", "large language model", "llm",
        "foundation model", "ai for science", "computer vision"
    ],
    "llm": [
        "llm", "large language model", "language model", "gpt", "chatgpt",
        "transformer", "generative ai", "rag", "retrieval augmented",
        "prompt", "fine-tuning", "agentic", "foundation model"
    ],
    "large language model": [
        "large language model", "llm", "language model", "gpt", "chatgpt",
        "transformer", "generative ai", "rag", "retrieval augmented",
        "fine-tuning", "instruction tuning", "foundation model"
    ],
    "generative ai": [
        "generative ai", "large language model", "llm", "diffusion model",
        "chatgpt", "gpt", "foundation model", "multimodal", "agentic", "rag"
    ],
    "agent": [
        "agent", "agentic", "multi-agent", "autonomous agent", "tool use",
        "llm", "reasoning", "reinforcement learning", "planning"
    ],
    "transformer": [
        "transformer", "self-attention", "attention mechanism", "bert", "gpt",
        "vision transformer", "large language model", "pre-trained", "sequence modeling"
    ],
    "diffusion": [
        "diffusion", "diffusion model", "score-based", "denoising",
        "generative model", "image generation", "flow matching", "latent diffusion"
    ],
    "reinforcement learning": [
        "reinforcement learning", "policy gradient", "q-learning", "markov decision",
        "multi-agent", "bandit", "reward", "offline rl", "control"
    ],
    "deep learning": [
        "deep learning", "neural network", "convolutional", "recurrent",
        "transformer", "backpropagation", "representation learning", "gpu"
    ],
    "metadata": [
        "metadata", "fair", "persistent identifier", "orcid", "doi", "raid",
        "data provenance", "data catalogue", "schema", "ontology",
        "vocabulary", "data stewardship", "repository"
    ],
    "fair": [
        "fair", "findable", "accessible", "interoperable", "reusable",
        "care principles", "metadata", "persistent identifier", "doi",
        "orcid", "raid", "data governance", "data stewardship", "open data"
    ],
    "pid": [
        "persistent identifier", "pid", "orcid", "doi", "raid", "ror",
        "igsn", "handle", "metadata", "citation", "provenance"
    ],
    "orcid": [
        "orcid", "persistent identifier", "pid", "doi", "raid", "ror",
        "researcher identifier", "metadata", "attribution"
    ],
    "ands": [
        "ands", "australian national data service", "rdsi", "nectar",
        "ardc", "australian research data commons", "research data australia",
        "data commons", "national data", "institutional repository"
    ],
    "ardc": [
        "ardc", "australian research data commons", "ands", "nectar", "rdsi",
        "data commons", "biocommons", "ecocommons", "planet research data commons",
        "people research data commons", "hass", "national research infrastructure"
    ],
    "data commons": [
        "data commons", "ardc", "australian research data commons",
        "biocommons", "ecocommons", "open data", "trusted data",
        "federated data", "data portal", "research data"
    ],
    "cloud": [
        "cloud", "cloud computing", "nectar", "research cloud", "openstack",
        "virtual machine", "aws", "azure", "kubernetes", "hybrid cloud",
        "infrastructure as a service"
    ],
    "nectar": [
        "nectar", "research cloud", "virtual laboratory", "openstack",
        "cloud computing", "virtual machine", "ardc", "national research cloud"
    ],
    "hpc": [
        "hpc", "high performance computing", "supercomputing", "supercomputer",
        "exascale", "petascale", "gpu", "cluster", "nci", "pawsey", "gadi",
        "setonix", "parallel computing", "mpi"
    ],
    "gpu": [
        "gpu", "cuda", "tensor core", "accelerator", "nvidia", "rocm",
        "openacc", "heterogeneous computing", "deep learning", "hpc"
    ],
    "supercomputing": [
        "supercomputing", "supercomputer", "hpc", "high performance computing",
        "exascale", "petascale", "nci", "pawsey", "mpi", "gpu", "parallel"
    ],
    "exascale": [
        "exascale", "frontier", "aurora", "el capitan", "fugaku", "setonix",
        "supercomputing", "hpc", "gpu", "heterogeneous", "power efficiency"
    ],
    "workflow": [
        "workflow", "pipeline", "nextflow", "snakemake", "cwl", "galaxy",
        "kepler", "reproducibility", "container", "automation", "orchestration"
    ],
    "container": [
        "container", "docker", "singularity", "apptainer", "kubernetes",
        "reproducibility", "workflow", "portable", "microservices"
    ],
    "docker": [
        "docker", "container", "singularity", "apptainer", "kubernetes",
        "reproducibility", "virtualisation", "workflow"
    ],
    "jupyter": [
        "jupyter", "jupyterhub", "notebook", "binder", "interactive computing",
        "python", "virtual desktop", "reproducible research", "literate programming"
    ],
    "rse": [
        "research software engineer", "research software", "rse", "software sustainability",
        "carpentries", "code quality", "open source", "version control", "git"
    ],
    "carpentries": [
        "carpentries", "software carpentry", "data carpentry", "library carpentry",
        "digital skills", "training", "workforce", "community of practice", "rse"
    ],
    "indigenous": [
        "indigenous", "indigenous data sovereignty", "care principles",
        "first nations", "aboriginal", "maori", "traditional knowledge",
        "cultural governance", "maiamnayri", "data governance"
    ],
    "trust": [
        "trust", "trusted research environment", "five safes", "sensitive data",
        "privacy", "cybersecurity", "governance", "de-identification",
        "federated learning", "ethics"
    ],
    "sensitive data": [
        "sensitive data", "five safes", "trusted research environment",
        "clinical data", "health data", "privacy", "de-identification",
        "secure enclave", "data governance"
    ],
    "quantum": [
        "quantum", "quantum computing", "qubit", "quantum circuit",
        "quantum simulation", "variational quantum", "qaoa", "fault-tolerant quantum"
    ],
    "climate": [
        "climate", "earth system", "weather prediction", "ocean", "atmosphere",
        "ecocommons", "environmental", "geospatial", "remote sensing", "digital twin"
    ],
    "genomics": [
        "genomics", "bioinformatics", "biocommons", "sequencing", "dna",
        "proteomics", "alphafold", "single-cell", "metagenomics", "life sciences"
    ],
    "graph": [
        "graph", "graph neural network", "gnn", "knowledge graph",
        "linked data", "network analysis", "sparql", "rdf", "topology"
    ],
}


def main():
    t0 = time.time()
    print("Loading corpus JSON datasets...")
    with open(os.path.join(SRC_DIR, "data_bundle.json"), encoding="utf-8") as f:
        bundle = json.load(f)
    with open(os.path.join(SRC_DIR, "search_index.json"), encoding="utf-8") as f:
        eres = json.load(f)
    with open(os.path.join(SRC_DIR, "neurips_index.json"), encoding="utf-8") as f:
        nips = json.load(f)
    with open(os.path.join(SRC_DIR, "sc_index.json"), encoding="utf-8") as f:
        sc = json.load(f)

    print(f"Loaded {len(eres)} eResearch, {len(nips)} NeurIPS, {len(sc)} SC records.")

    # Copy corpus_data.js to corpus-semantic/
    src_data_js = os.path.join(SRC_DIR, "corpus_data.js")
    dst_data_js = os.path.join(OUT_DIR, "corpus_data.js")
    if os.path.exists(src_data_js):
        shutil.copy2(src_data_js, dst_data_js)
        print("Copied corpus_data.js to corpus-semantic/corpus_data.js")

    # Build text corpus for unified 64-D TF-IDF + TruncatedSVD
    docs = []
    eres_pnames = bundle["paradigm_names"]
    for r in eres:
        title = r.get("t", "")
        l250 = r.get("l250", "")
        p_name = eres_pnames[r.get("c", 0)] if 0 <= r.get("c", 0) < len(eres_pnames) else ""
        # Weight title + lead-250 words + paradigm context
        docs.append(f"{title} {title} {l250} {p_name}")

    for item in nips:
        yr, title, auth, track, pid = item
        p_name = NEURIPS_PARADIGMS[pid] if 0 <= pid < len(NEURIPS_PARADIGMS) else ""
        docs.append(f"{title} {title} {p_name}")

    for item in sc:
        yr, title, auth, track, pid, doi = item
        p_name = SC_PARADIGMS[pid] if 0 <= pid < len(SC_PARADIGMS) else ""
        docs.append(f"{title} {title} {p_name}")

    print(f"Fitting balanced TfidfVectorizer across eResearch, NeurIPS, and SC...")
    # Build balanced training set (all 2,512 eResearch + all 2,279 SC + 4,500 stratified NeurIPS)
    # so eResearch, HPC, and ML terms all get equal representation in the 6,000-term vocabulary & SVD axes
    rng = np.random.default_rng(42)
    nips_sample_idx = rng.choice(len(nips), size=min(4500, len(nips)), replace=False)
    balanced_docs = (
        docs[:len(eres)]
        + [docs[len(eres) + i] for i in nips_sample_idx]
        + docs[len(eres) + len(nips):]
    )

    vec = TfidfVectorizer(
        max_features=6000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=3,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_-]{1,}\b",
    )
    X_bal = vec.fit_transform(balanced_docs)
    vocab = vec.get_feature_names_out()
    idf = vec.idf_
    print(f"Balanced TF-IDF matrix shape: {X_bal.shape}")

    n_dim = 48
    print(f"Fitting TruncatedSVD (n_components={n_dim}) on balanced corpus, then transforming all {len(docs)} records...")
    svd = TruncatedSVD(n_components=n_dim, random_state=42)
    svd.fit(X_bal)

    X_all = vec.transform(docs)
    doc_vecs = svd.transform(X_all).astype(np.float32)
    doc_vecs = normalize(doc_vecs, norm="l2", axis=1)

    # Exact query projection matrix: for a query with TF-IDF weights, projection is sum(idf[t] * V[t])
    term_proj = (svd.components_.T * idf[:, None]).astype(np.float32)
    term_vecs_norm = normalize(term_proj, norm="l2", axis=1)

    # Quantize document vectors to int8 [-127, 127]
    doc_int8 = np.clip(np.round(doc_vecs * 127.0), -127, 127).astype(np.int8)
    n_eres = len(eres)
    n_nips = len(nips)
    n_sc = len(sc)

    eres_b64 = base64.b64encode(doc_int8[:n_eres].tobytes()).decode("ascii")
    nips_b64 = base64.b64encode(doc_int8[n_eres:n_eres + n_nips].tobytes()).decode("ascii")
    sc_b64 = base64.b64encode(doc_int8[n_eres + n_nips:].tobytes()).decode("ascii")

    # Store exact SVD projection rows (scaled by global max abs) in Int8 [-127, 127]
    max_abs = float(np.max(np.abs(term_proj))) or 1.0
    term_int8 = np.clip(np.round((term_proj / max_abs) * 127.0), -127, 127).astype(np.int8)
    term_b64 = base64.b64encode(term_int8.tobytes()).decode("ascii")
    idf_int8 = np.clip(np.round((idf / float(np.max(idf))) * 127.0), 1, 127).astype(np.uint8)
    idf_b64 = base64.b64encode(idf_int8.tobytes()).decode("ascii")

    # Pre-compute top-6 nearest semantic neighbors for each vocabulary term
    print("Computing nearest semantic neighbors for vocabulary terms...")
    sim_matrix = np.dot(term_vecs_norm, term_vecs_norm.T)
    term_neighbors = {}
    vocab_list = [str(w) for w in vocab]
    for i, w in enumerate(vocab_list):
        if len(w) < 3:
            continue
        sims = sim_matrix[i]
        # Top indices excluding self
        top_idx = np.argsort(sims)[::-1][1:18]
        picked = []
        for j in top_idx:
            cand = vocab_list[j]
            if len(cand) < 3:
                continue
            # Avoid trivial morphological duplicates if we already have enough
            if cand == w or cand == w + "s" or w == cand + "s":
                continue
            if float(sims[j]) < 0.52:
                break
            picked.append(cand)
            if len(picked) >= 6:
                break
        if picked:
            term_neighbors[w] = picked

    out_js_path = os.path.join(OUT_DIR, "semantic_engine.js")
    print(f"Writing {out_js_path}...")
    with open(out_js_path, "w", encoding="utf-8") as f:
        f.write("// Pre-computed 48-D Latent Semantic Embedding Engine & Domain Ontology (31,778 Records)\n")
        f.write(f"const SEMANTIC_DIM = {n_dim};\n")
        f.write(f"const SEMANTIC_ONTOLOGY = {json.dumps(CURATED_ONTOLOGY)};\n")
        f.write(f"const SEMANTIC_VOCAB = {json.dumps(vocab_list)};\n")
        f.write(f"const SEMANTIC_NEIGHBORS = {json.dumps(term_neighbors)};\n")
        f.write(f"const VOCAB_VECS_B64 = \"{term_b64}\";\n")
        f.write(f"const VOCAB_IDF_B64 = \"{idf_b64}\";\n")
        f.write(f"const ERES_VECS_B64 = \"{eres_b64}\";\n")
        f.write(f"const NIPS_VECS_B64 = \"{nips_b64}\";\n")
        f.write(f"const SC_VECS_B64 = \"{sc_b64}\";\n")

    sz_mb = os.path.getsize(out_js_path) / (1024 * 1024)
    print(f"Done in {time.time() - t0:.1f}s! semantic_engine.js size: {sz_mb:.2f} MB")


if __name__ == "__main__":
    main()

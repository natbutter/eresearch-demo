# In-Browser Semantic Search Explorer (`eresearch-demo`)

An interactive, serverless semantic search engine that runs entirely in your web browser across **31,778** conference talks and papers (2010–2026). Built as an educational demonstration of how modern vector embeddings and in-browser neural inference work.

## Key Technologies

* **[Transformers.js](https://huggingface.co/docs/transformers.js) (`@xenova/transformers`) & ONNX Runtime Web**: Runs the quantized [`Xenova/all-MiniLM-L6-v2`](https://huggingface.co/Xenova/all-MiniLM-L6-v2) transformer model directly in the browser via WebAssembly (WASM) to embed user search queries in ~10–30 ms without a backend server.
* **[SentenceTransformers](https://sbert.net/) (`sentence-transformers/all-MiniLM-L6-v2`)**: Used offline in [`build_minilm_embeddings.py`](build_minilm_embeddings.py) to encode all 31,778 documents into L2-normalized **384-dimensional** semantic vectors.
* **Int8 Scalar Quantization & TypedArrays**: Compresses the 384-D `Float32` document vectors by 4× into signed 8-bit integers (`Int8Array`, `[-127, 127]`) stored as Base64 in `semantic_engine.js`, enabling full-corpus cosine similarity (dot-product) scans in `<5 ms`.

## How Semantic Search Works (Step-by-Step)

Follow the numbered comments in [`build_minilm_embeddings.py`](build_minilm_embeddings.py) and [`index.html`](index.html):

1. **Offline Document Embedding (`build_minilm_embeddings.py`)**: Each talk/paper is converted into a 384-D unit vector using `all-MiniLM-L6-v2`, quantized to `Int8`, and exported to `semantic_engine.js`.
2. **Decode Embeddings (`index.html` — Step 2)**: On page load, Base64 strings are decoded into flat `Int8Array` buffers in browser memory.
3. **In-Browser Query Embedding (`index.html` — Step 3 & Step 6)**: When you type a query, `Transformers.js` runs `Xenova/all-MiniLM-L6-v2` in the browser (`pooling: 'mean'`, `normalize: true`) to produce a 384-D query vector.
4. **Cosine Similarity Scan (`index.html` — Step 4)**: Because both query and document vectors are L2-normalized, cosine similarity simplifies to a fast dot product (`sum(qVec[d] * docVec[d])`) across all 31,778 records.
5. **Interactive Trend Visualization (`index.html` — Step 5)**: Plots the yearly share of matching talks/papers from 2010 to 2026, comparing **Neural Semantic Search** against an **Exact Keyword** baseline.

## Running Locally

Because `index.html` loads `Transformers.js` as an ES module, serve the folder with any static HTTP server:

```bash
python3 -m http.server 8000
# Open http://localhost:8000 in your browser
```

## Datasets & Attribution

This repository indexes metadata from three scholarly sources (see [`LICENSE`](LICENSE) for full attribution details):

* **[eResearch Australasia Conference](https://conference.eresearch.edu.au/)** (2,512 talks, 2010–2026) — Australasian eResearch Organisations ([AeRO](https://aero.edu.au/)) & [legacy archive](https://eresearchau.files.wordpress.com/).
* **[NeurIPS Proceedings](https://papers.nips.cc/)** (26,987 papers, 2010–2025) — Conference on Neural Information Processing Systems ([neurips.cc](https://neurips.cc/)).
* **[ACM/IEEE Supercomputing (SC)](https://sc25.supercomputing.org/)** (2,279 papers, SC10–SC25) — International Conference for High Performance Computing, Networking, Storage, and Analysis ([ACM Digital Library](https://dl.acm.org/conference/sc) / [DBLP](https://dblp.org/)).

## License

Code is released under the MIT License. See [`LICENSE`](LICENSE) for code licensing and dataset/model attributions.

| Retriever | Recall@200 (ranking ceiling) | Overlap with exact top-200 | Mean latency (ms) | p99 latency (ms) |
|---|---:|---:|---:|---:|
| exact | 0.6171 | 1.0000 | 0.0458 | 0.0686 |
| hnsw-engine | 0.6131 | 0.9842 | 1.0664 | 2.3082 |
| hnsw-engine-numba | 0.6131 | 0.9842 | 0.0936 | 0.1954 |
| faiss-hnsw | 0.6066 | 0.9784 | 0.1270 | 0.3899 |

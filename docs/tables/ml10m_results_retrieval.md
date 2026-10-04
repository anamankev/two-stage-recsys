| Retriever | Recall@200 (ranking ceiling) | Overlap with exact top-200 | Mean latency (ms) | p99 latency (ms) |
|---|---:|---:|---:|---:|
| exact | 0.6545 | 1.0000 | 0.2029 | 0.2913 |
| hnsw-engine-numba | 0.6408 | 0.9768 | 0.2005 | 0.4719 |
| faiss-hnsw | 0.6381 | 0.9748 | 0.2609 | 0.8788 |

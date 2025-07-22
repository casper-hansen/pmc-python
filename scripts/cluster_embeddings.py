import faiss, numpy as np, sys
from datasets import load_dataset
from tqdm.auto import tqdm

DATASET_ID = "casperhansen/pmc-oa-markdown-embeddings"
EMBED_COL  = "embed"
TEXT_COL   = "text"
K, TAU     = 5, 0.70
BATCH_Q    = 10_000

# 1. Load & normalise -------------------------------------------------------
ds   = load_dataset(DATASET_ID, split="train").take(10000)
emb  = np.asarray(ds[EMBED_COL], dtype='float32')
faiss.normalize_L2(emb)
n, d = emb.shape
print(f"{n:,} vectors, dim={d}")

# 2. ANN index with cosine similarity --------------------------------------
index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 200
index.add(emb)
index.hnsw.efSearch = 64

# 3. k-NN search -----------------------------------------------------------
D = np.empty((n, K + 1), dtype='float32')   # similarities
I = np.empty((n, K + 1), dtype='int64')
for s in tqdm(range(0, n, BATCH_Q), desc="k-NN search"):
    e = min(s + BATCH_Q, n)
    D[s:e], I[s:e] = index.search(emb[s:e], K + 1)

# 4. Form tight 5-item clusters -------------------------------------------
used, clusters = np.zeros(n, bool), []

def all_pairs_above_threshold(vecs, tau):
    M = vecs @ vecs.T
    return np.all(M[~np.eye(len(vecs), dtype=bool)] >= tau)

for seed in np.argsort(-D[:, 1]):          # best 1-NN similarity first
    if used[seed]:
        continue
    neigh = I[seed, 1:K + 1]
    if (D[seed, 1:K + 1] < TAU).any():     # centre must like all 4
        continue
    if used[neigh].any():
        continue
    cand = np.concatenate(([seed], neigh))
    if all_pairs_above_threshold(emb[cand], TAU):
        clusters.append(cand.tolist())
        used[cand] = True

print(f"Clusters kept: {len(clusters):,}")

# 5. Inspect a cluster -----------------------------------------------------
if clusters:
    cid = 1
    print("\nSample cluster:")
    for idx in clusters[cid]:
        print("‣", ds[TEXT_COL][idx][:500].replace("\n", " ") + "…")
else:
    print("Try lowering TAU or increasing K to find clusters.")
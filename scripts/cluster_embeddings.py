"""
1. Load a 10k slice of the PMC-OA markdown-embedding dataset
2. Remove near-duplicate documents with SemHash
3. Build & normalise embeddings
4. Greedy clique clustering (size 5, all-pairs ≥ TAU)
5. Push resulting clusters to the Hub
"""

import faiss
import numpy as np
from datasets import (load_dataset, Dataset, DatasetDict,
                      Features, Sequence, Value)
from tqdm.auto import tqdm
from semhash import SemHash

# ----------------------------- config ---------------------------------
DATASET_ID = "casperhansen/pmc-oa-markdown-embeddings"
TEXT_COL, EMBED_COL = "text", "embed"

DEDUP_THRESHOLD = 0.90     # SemHash cut-off
K = 5                      # seed + 4 neighbours
TAU = 0.70                 # min cosine sim inside a clique
BATCH_Q = 10_000           # ANN query batch size
# ----------------------------------------------------------------------

# 1. Load a small slice
ds_raw = load_dataset(DATASET_ID, split="train")

# 2. Near-duplicate removal ----------------------------------------------------
records = [{"id": i, TEXT_COL: txt} for i, txt in enumerate(ds_raw[TEXT_COL])]
semhash   = SemHash.from_records(records, columns=[TEXT_COL])
dedup_res = semhash.self_deduplicate(threshold=DEDUP_THRESHOLD)

keep_ids = [r["id"] for r in dedup_res.selected]
print(f"Kept {len(keep_ids):,} / {len(records):,} after SemHash (τ={DEDUP_THRESHOLD})")

ds = ds_raw.select(keep_ids)            # keep only non-duplicates

# 3. Build & normalise embeddings ---------------------------------------------
emb = np.asarray(ds[EMBED_COL], dtype="float32")
faiss.normalize_L2(emb)                 # cosine → inner-product
n, d = emb.shape
print(f"{n:,} vectors, dim={d}")

# 4. ANN search + greedy 5-clique clustering -----------------------------------
index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 200
index.add(emb)
index.hnsw.efSearch = 64

D = np.empty((n, K + 1), dtype="float32")   # distances
I = np.empty((n, K + 1), dtype="int64")     # indices

for s in tqdm(range(0, n, BATCH_Q), desc="k-NN search"):
    e = min(s + BATCH_Q, n)
    D[s:e], I[s:e] = index.search(emb[s:e], K + 1)   # self + K NNs

used, clusters = np.zeros(n, bool), []               # bookkeeping

def clique_ok_and_mean(vecs: np.ndarray, tau: float):
    """
    Return (is_clique, mean_pairwise_similarity).

    We build the full similarity matrix *once* and reuse it for both the
    ≥tau test and the mean computation, so there's no extra work.
    """
    S = vecs @ vecs.T                                            # (k,k)
    off_diag = S[np.triu_indices(len(vecs), k=1)]                # k·(k-1)/2
    return np.all(off_diag >= tau), float(off_diag.mean())


used, clusters, avg_sims = np.zeros(n, bool), [], []

for seed in np.argsort(-D[:, 1]):                               # strong 1-NN first
    if used[seed]:
        continue

    neigh = I[seed, 1 : K + 1]

    if (D[seed, 1 : K + 1] < TAU).any():                        # seed must like all 4
        continue
    if used[neigh].any():
        continue

    cand = np.concatenate(([seed], neigh))
    ok, mu = clique_ok_and_mean(emb[cand], TAU)
    if ok:
        clusters.append(cand.tolist())
        avg_sims.append(mu)
        used[cand] = True

print(f"Clusters kept: {len(clusters):,}")

# 5. Build HF dataset with clusters & push to Hub ------------------------------
cluster_texts  = [[ds[TEXT_COL][i]  for i in c] for c in clusters]
cluster_embeds = [[ds[EMBED_COL][i] for i in c] for c in clusters]

features = Features({
    "texts" : Sequence(Value("string")),
    "embeds": Sequence(Sequence(Value("float32"))),
    "avg_similarity": Value("float32"),
})

ds_clusters = Dataset.from_dict(
    {
        "texts"         : cluster_texts,
        "embeds"        : cluster_embeds,
        "avg_similarity": avg_sims,
    },
    features=features
)

DatasetDict({"train": ds_clusters}).push_to_hub(
    "casperhansen/pmc-oa-markdown-clustering-test"
)
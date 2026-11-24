import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import pickle
from pathlib import Path
from time import perf_counter_ns

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import torch
from torchvision.datasets import FashionMNIST
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.transforms.functional import resize

ROOT = Path(__file__).resolve().parent
SIZES = (1000, 5000, 12000)
CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def encoder(device):
    net = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).to(device).eval()
    return net


@torch.inference_mode()
def encode(model, images, device):
    images = resize(images[:, None].to(device).float() / 255, [96, 96], antialias=True).repeat(1, 3, 1, 1)
    mean = images.new_tensor([.485, .456, .406])[None, :, None, None]
    std = images.new_tensor([.229, .224, .225])[None, :, None, None]
    features = model.avgpool(model.features((images - mean) / std)).flatten(1)
    return torch.nn.functional.normalize(features).cpu().numpy()


def top10(scores):
    ids = np.argpartition(scores, -10)[-10:]
    return ids[np.argsort(scores[ids])[::-1]]


def datasets():
    FashionMNIST.mirrors = ["https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"]
    return (FashionMNIST(ROOT / "cache", train=True, download=True),
            FashionMNIST(ROOT / "cache", train=False, download=True))


def run():
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(4)
    training, testing = datasets()
    rng = np.random.default_rng(731)
    train = rng.permutation(len(training))[:max(SIZES)]
    test = rng.permutation(len(testing))[:300]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = ROOT / "cache" / "features.npz"
    if cache.exists():
        arrays = np.load(cache)
        vectors, queries = arrays["vectors"], arrays["queries"]
    else:
        model = encoder(device)
        def batches(images):
            return np.concatenate([encode(model, chunk, device) for chunk in images.split(96)])
        vectors, queries = batches(training.data[train]), batches(testing.data[test])
        np.savez(cache, vectors=vectors, queries=queries)
    projection = rng.normal(size=(vectors.shape[1], 16)).astype("float32") / 4
    projected_queries = queries @ projection
    rows = []
    with threadpool_limits(limits=1, user_api="blas"):
        for size in SIZES:
            data = vectors[:size]
            tree = cKDTree(data @ projection)
            exact_bytes = data.nbytes
            approx_bytes = exact_bytes + projection.nbytes + len(pickle.dumps(tree, protocol=5))
            for candidates in (0, 32, 128, 512):
                def search(query, projected):
                    if not candidates:
                        return top10(data @ query)
                    ids = tree.query(projected, k=candidates)[1]
                    return ids[top10(data[ids] @ query)]
                search(queries[0], projected_queries[0])
                times, hits, relevance, results = [], [], [], []
                for i, query in enumerate(queries):
                    start = perf_counter_ns()
                    ids = search(query, query @ projection if candidates else None)
                    times.append((perf_counter_ns() - start) / 1e6)
                    results.append(ids)
                    if candidates:
                        hits.append(len(set(ids).intersection(exact[i])) / 10)
                    relevance.append(float(np.mean(training.targets[train[ids]].numpy() == int(testing.targets[test[i]]))))
                if not candidates:
                    exact = results
                rows.append(dict(size=size, candidates=candidates, recall=round(float(np.mean(hits)) if hits else 1, 4),
                                 precision=round(float(np.mean(relevance)), 4), p95_ms=round(float(np.percentile(times, 95)), 4),
                                 index_bytes=approx_bytes if candidates else exact_bytes))
            print(size, rows[-4:], flush=True)
        # Curated replay includes a measured class mismatch, selected algorithmically.
        relevance = [int((training.targets[train[ids]] == testing.targets[test[i]]).sum()) for i, ids in enumerate(exact)]
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    public = ROOT / "site"
    public.mkdir(exist_ok=True)
    def thumbnail(images, index):
        import base64
        from io import BytesIO
        buffer = BytesIO()
        Image.fromarray(images[index].numpy()).save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    order = np.argsort(relevance, kind="stable")
    selected = [int(order[-1]), int(order[0]), 1, 2, 3, 4, 5, 6]
    samples = [
        dict(query=thumbnail(testing.data, test[i]), label=CLASSES[int(testing.targets[test[i]])],
             matches=[dict(image=thumbnail(training.data, train[j]), label=CLASSES[int(training.targets[train[j]])]) for j in exact[i]],
             approximate=[dict(image=thumbnail(training.data, train[j]), label=CLASSES[int(training.targets[train[j]])]) for j in results[i]])
        for i in dict.fromkeys(selected)
    ]
    (public / "results.js").write_text("window.report = " + json.dumps(dict(rows=rows, samples=samples)) + ";\n", encoding="utf8")
    (ROOT / "results.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf8")
    fig, ax = plt.subplots(figsize=(7.1, 4.0), layout="constrained")
    for size, color in zip(SIZES, ("#315c8b", "#0a766e", "#9b591f")):
        approximate = [row for row in rows if row["size"] == size and row["candidates"]]
        baseline = next(row for row in rows if row["size"] == size and not row["candidates"])
        ax.plot([r["p95_ms"] for r in approximate], [r["recall"] for r in approximate], "o-", color=color, label=f"{size:,} images")
        ax.scatter(baseline["p95_ms"], 1, marker="D", color=color)
    ax.set(xlabel="Per-query p95 latency (ms)", ylabel="Recall@10 vs exact", ylim=(0, 1.06))
    ax.grid(alpha=.18)
    ax.legend(title="Diamond = exact", frameon=False)
    fig.savefig(figures / "recall-latency.png", dpi=180)
    (public / "recall-latency.png").write_bytes((figures / "recall-latency.png").read_bytes())
    assert rows[-1]["recall"] >= .88, rows[-1]
    assert rows[-1]["precision"] >= rows[-4]["precision"] - .03, rows[-1]


if __name__ == "__main__":
    run()

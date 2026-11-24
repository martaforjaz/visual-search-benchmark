"""Compare pretrained encoders and search indexes on held-out COIL-100 viewpoints."""
import io
import json
from pathlib import Path
import re
from time import perf_counter
from zipfile import ZipFile

import faiss
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
import torch
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"


def load_catalog():
    archive = ZipFile(CACHE / "coil100.zip")
    entries = []
    for name in archive.namelist():
        match = re.fullmatch(r"obj(\d+)__(\d+)\.png", Path(name).name)
        if match:
            entries.append((tuple(map(int, match.groups())), name))
    entries.sort()
    assert len(entries) == 7200
    images = [Image.open(io.BytesIO(archive.read(name))).convert("RGB") for _, name in entries]
    archive.close()
    return images, np.array([key for key, _ in entries])


def load_encoder(name, device):
    if name == "mobilenet":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = mobilenet_v3_small(weights=weights).eval().to(device)
        return lambda x: model.avgpool(model.features(x)).flatten(1), weights.transforms()
    import open_clip
    model, _, transform = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai", device=device)
    return model.eval().encode_image, transform


@torch.inference_mode()
def embed(images, encode_image, transform, device):
    return np.concatenate([
        torch.nn.functional.normalize(encode_image(torch.stack([transform(img) for img in images[i:i+64]]).to(device)).float())
        .cpu().numpy()
        for i in range(0, len(images), 64)
    ])


def measure(search, queries, truth):
    search(queries[0])
    ids, latency = [], []
    for query in queries:
        start = perf_counter()
        ids.append(search(query))
        latency.append((perf_counter()-start)*1000)
    ids = np.asarray(ids)
    row = {"p95_ms": round(float(np.percentile(latency, 95)), 4)}
    if truth is not None:
        row["neighbor_recall"] = round(float(np.mean([len(set(a)&set(b))/10 for a,b in zip(ids, truth)])), 4)
    return ids, row


def run():
    images, keys = load_catalog()
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gallery = np.flatnonzero(keys[:, 1] % 30 == 0)
    queries = np.flatnonzero(keys[:, 1] % 60 == 15)
    results = {"dataset":"COIL-100", "queries":len(queries), "encoders":{}, "indexes":[]}
    for name in ("mobilenet", "clip"):
        path = CACHE / f"coil-{name}{'-quickgelu' if name == 'clip' else ''}.npy"
        if not path.exists():
            method, transform = load_encoder(name, device)
            np.save(path, embed(images, method, transform, device))
        vectors = np.load(path)
        database, probe = vectors[gallery], vectors[queries]
        exact = faiss.IndexFlatIP(database.shape[1])
        exact.add(database)
        ids = exact.search(probe,10)[1]
        matches = keys[gallery[ids],0] == keys[queries,0,None]
        results["encoders"][name] = {"hit_at_1":float(matches[:,0].mean()), "hit_at_10":float(matches.any(axis=1).mean())}
        near = np.flatnonzero(keys[:, 1] < 60)
        far = np.flatnonzero((keys[:, 1] >= 150) & (keys[:, 1] <= 210))
        exact.reset()
        exact.add(vectors[near])
        ids = exact.search(vectors[far], 10)[1]
        matches = keys[near[ids], 0] == keys[far, 0, None]
        results["encoders"][name]["opposite_view_hit_at_1"] = float(matches[:, 0].mean())
        results["encoders"][name]["opposite_view_hit_at_10"] = float(matches.any(axis=1).mean())
        print(name, results["encoders"][name], flush=True)
    # Hold the MobileNet representation fixed. Every gallery includes all 100 objects.
    vectors = np.load(CACHE / "coil-mobilenet.npy")
    probe = vectors[queries]
    for spacing in (120,60,30):
        ids = np.flatnonzero(keys[:,1] % spacing == 0)
        data = vectors[ids]
        dim = data.shape[1]
        flat = faiss.IndexFlatIP(dim)
        flat.add(data)
        graph = faiss.IndexHNSWFlat(dim, 24, faiss.METRIC_INNER_PRODUCT)
        graph.hnsw.efConstruction = 80
        graph.add(data)
        projection = np.random.default_rng(731).normal(size=(dim,16)).astype("float32") / 4
        tree = cKDTree(data @ projection)

        def projected(q):
            shortlist = tree.query(q @ projection, k=128)[1]
            return shortlist[np.argsort(-(data[shortlist] @ q))[:10]]

        for name, fn in [("exact", lambda q: flat.search(q[None],10)[1][0]), ("kd-tree", projected),
                         ("hnsw", lambda q: graph.search(q[None],10)[1][0])]:
            output, metrics = measure(fn, probe, None if name=="exact" else truth)
            if name=="exact":
                truth = output
            hit = (keys[ids[output],0] == keys[queries,0,None]).any(axis=1).mean()
            results["indexes"].append(dict(size=len(ids), method=name, hit_at_10=float(hit), **metrics))
    (ROOT / "catalog-results.json").write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    with threadpool_limits(1, user_api="blas"):
        run()

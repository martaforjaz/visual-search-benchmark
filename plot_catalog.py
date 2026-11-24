"""Render the catalog science figure from measured results."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
data = json.loads((root / "catalog-results.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), layout="constrained")
names = ["MobileNetV3", "CLIP ViT-B/32"]
for i, (method, metrics) in enumerate(data["encoders"].items()):
    axes[0].bar([i-.17, i+.17], [100*metrics["hit_at_1"], 100*metrics["opposite_view_hit_at_1"]],
                color=["#244e79","#439387"], width=.32)
axes[0].set(xticks=[0,1], xticklabels=names, ylabel="Same-object Hit@1 (%)", ylim=(0,110), title="A   Viewpoint shift")
axes[0].legend(["Interleaved", "Opposite sector"], frameon=False)
for method, color in [("exact","#244e79"),("kd-tree","#ab6433"),("hnsw","#439387")]:
    rows = [r for r in data["indexes"] if r["method"] == method]
    axes[1].plot([r["size"] for r in rows], [r["p95_ms"] for r in rows], "o-", color=color, label=method)
axes[1].set(xlabel="Gallery images", ylabel="Search p95 (ms)", title="B   Fixed MobileNet embeddings")
axes[1].legend(frameon=False)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.savefig(root / "product-site" / "benchmark.png", dpi=180)

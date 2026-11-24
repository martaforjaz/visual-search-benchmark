"""Build a reproducible, illustrated comparison using real encoder rankings."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torchvision.transforms.functional import to_tensor, normalize

from build_product_demo import Encoder, SITE
from catalog_benchmark import embed, load_encoder

ROOT = Path(__file__).resolve().parent


def mobile(images):
    model = Encoder().eval()
    with torch.inference_mode():
        inputs = torch.stack([normalize(to_tensor(image), [.485,.456,.406], [.229,.224,.225])
                              for image in images])
        return torch.nn.functional.normalize(model(inputs)).numpy()


def main():
    torch.set_num_threads(2)
    catalog = json.loads((SITE / "catalog.json").read_text())
    gallery = catalog["gallery"]
    candidates = []
    for query in catalog["queries"]:
        image = Image.open(SITE / query["image"]).convert("RGB")
        candidates.append((query, image, "Held-out illustration"))
        for saturation in (.5, .15, 0):
            degraded = ImageEnhance.Color(image).enhance(saturation).rotate(12, fillcolor="#eaf0f7")
            candidates.append((query, degraded.filter(ImageFilter.GaussianBlur(1)), "Rotated, blurred, reduced saturation"))
    images = [image for _, image, _ in candidates]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, transform = load_encoder("clip", device)
    clip_gallery = embed([Image.open(SITE / row["image"]).convert("RGB") for row in gallery],
                         encoder, transform, device)
    scores = [mobile(images) @ np.asarray([row["vector"] for row in gallery]).T,
              embed(images, encoder, transform, device) @ clip_gallery.T]
    ranks = [np.argsort(-matrix, axis=1, kind="stable")[:, :3] for matrix in scores]

    def difficulty(i):
        query = candidates[i][0]["id"]
        top = [gallery[order[i, 0]]["id"] for order in ranks]
        return (sum(item != query and item // 2 == query // 2 for item in top),
                (top[0] == query) != (top[1] == query))

    chosen = max(range(len(candidates)), key=difficulty)
    if difficulty(chosen)[0] == 0:
        raise RuntimeError("No same-category wrong product was found")
    query, image, condition = candidates[chosen]
    image.save(SITE / "comparison-query.png")
    methods = []
    for name, order in zip(("MobileNetV3-Small", "CLIP ViT-B/32"), ranks):
        matches = [{**{key: gallery[j][key] for key in ("label", "image")},
                    "correct": gallery[j]["id"] == query["id"]} for j in order[chosen]]
        methods.append({"name": name, "matches": matches})
    report = dict(query=dict(label=query["label"], image="comparison-query.png", condition=condition),
                  methods=methods)
    (SITE / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(report)
    fig, axes = plt.subplots(2, 4, figsize=(8.6, 4.5), layout="constrained")
    for row, method in enumerate(methods):
        items = [(image, "Query · " + query["label"], False)] + [
            (Image.open(SITE / item["image"]), f"{i+1}. " + item["label"], item["correct"])
            for i, item in enumerate(method["matches"])
        ]
        for ax, (img, label, correct) in zip(axes[row], items):
            ax.imshow(img)
            ax.set(xticks=[], yticks=[], xlabel=label)
            for spine in ax.spines.values():
                spine.set_color("#287868" if correct else "#c5d3e3")
                spine.set_linewidth(2 if correct else .7)
        axes[row, 0].set_title(method["name"], loc="left", color="#142d50")
    fig.savefig(ROOT / "figures" / "product-comparison.png", dpi=170)


if __name__ == "__main__":
    main()

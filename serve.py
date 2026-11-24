"""Local image upload demo, backed by the same exact retrieval baseline."""
import base64
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch

from benchmark import CLASSES, ROOT, SIZES, datasets, encode, encoder, top10


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT / "site", **kwargs)

    def do_POST(self):
        if self.path != "/search":
            return self.send_error(404)
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 2_000_000:
                return self.send_error(413)
            with Image.open(BytesIO(self.rfile.read(size))) as image:
                if image.width * image.height > 4_000_000:
                    return self.send_error(413)
                pixels = np.array(image.convert("L").resize((28, 28)))
        except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError):
            return self.send_error(400, "Expected a valid image")
        vector = encode(model, torch.from_numpy(pixels[None]), "cpu")[0]
        matches = []
        for idx in train[top10(vectors @ vector)]:
            buffer = BytesIO()
            Image.fromarray(training.data[idx].numpy()).save(buffer, format="PNG")
            matches.append(dict(image="data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(),
                                label=CLASSES[int(training.targets[idx])]))
        body = json.dumps(matches).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    torch.set_num_threads(4)
    vectors = np.load(ROOT / "cache" / "features.npz")["vectors"]
    training, _ = datasets()
    train = np.random.default_rng(731).permutation(len(training))[:max(SIZES)]
    model = encoder("cpu")
    print("Open http://127.0.0.1:8777", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8777), Handler).serve_forever()

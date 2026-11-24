"""Loopback catalog API with durable, idempotent insertions."""
from contextlib import closing
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import re
import sqlite3
from threading import Lock
from time import perf_counter
from urllib.parse import urlsplit

import numpy as np
from PIL import Image
import torch
from catalog_benchmark import CACHE, load_encoder


class Catalog:
    def __init__(self, db, embed, seed=()):
        self.db, self.embed, self.lock = db, embed, Lock()
        with closing(sqlite3.connect(db)) as con, con:
            con.execute("CREATE TABLE IF NOT EXISTS views (id TEXT PRIMARY KEY, product TEXT, digest TEXT, vector BLOB)")
            con.executemany("INSERT OR IGNORE INTO views VALUES (?, ?, ?, ?)", seed)
            self.reload(con)

    def reload(self, con):
        rows = con.execute("SELECT id, product, vector FROM views ORDER BY id").fetchall()
        self.snapshot = ([(r[0], r[1]) for r in rows],
                         np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows]) if rows else None)

    def insert(self, key, product, body):
        digest = sha256(body).hexdigest()
        vector = self.embed(body)
        with self.lock, closing(sqlite3.connect(self.db)) as con:
            row = con.execute("SELECT product, digest FROM views WHERE id=?", (key,)).fetchone()
            if row:
                return 200 if row == (product, digest) else 409
            with con:
                con.execute("INSERT INTO views VALUES (?, ?, ?, ?)", (key, product, digest, vector.tobytes()))
            self.reload(con)
        return 201

    def search(self, body):
        start = perf_counter()
        vector = self.embed(body)
        encoded = perf_counter()
        labels, vectors = self.snapshot
        ids = np.argsort(-(vectors @ vector))[:10] if vectors is not None else []
        matches = [{"view": labels[i][0], "product": labels[i][1]} for i in ids]
        return {"matches": matches, "embedding_ms": (encoded-start)*1000, "index_ms": (perf_counter()-encoded)*1000}


def server(catalog, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            path = urlsplit(self.path).path
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 2_000_000:
                    return self.send_error(413)
                body = self.rfile.read(length)
                if path == "/search":
                    status, result = 200, catalog.search(body)
                elif match := re.fullmatch(r"/products/([\w-]{1,64})/views/([\w-]{1,64})", path, re.ASCII):
                    product, view = match.groups()
                    status = catalog.insert(f"{product}/{view}", product, body)
                    result = {"status": status}
                else:
                    return self.send_error(404)
            except (ValueError, OSError, Image.DecompressionBombError):
                return self.send_error(400)
            response = json.dumps(result).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def production_catalog(db, seed=True):
    torch.set_num_threads(2)
    method, transform = load_encoder("mobilenet", "cpu")

    @torch.inference_mode()
    def embedding(body):
        with Image.open(io.BytesIO(body)) as image:
            if image.width * image.height > 4_000_000:
                raise ValueError("Image too large")
            return torch.nn.functional.normalize(method(transform(image.convert("RGB"))[None]))[0].numpy()

    rows = []
    if seed:
        vectors = np.load(CACHE / "coil-mobilenet.npy")
        rows = [(f"object-{i//72+1}/angle-{i%72*5}", f"object-{i//72+1}", "dataset", vector.tobytes())
                for i, vector in enumerate(vectors) if i % 72 % 6 == 0]
    return Catalog(db, embedding, rows)


if __name__ == "__main__":
    with server(production_catalog(CACHE / "products.sqlite"), 8778) as api:
        print("Catalog API: http://127.0.0.1:8778", flush=True)
        api.serve_forever()

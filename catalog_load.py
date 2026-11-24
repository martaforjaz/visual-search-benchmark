"""Measure concurrent HTTP search during catalog refresh and verify restart durability."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from time import perf_counter
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from catalog_api import production_catalog, server


@contextmanager
def running(catalog):
    with server(catalog, 0) as api:
        thread = Thread(target=api.serve_forever)
        thread.start()
        try:
            yield f"http://127.0.0.1:{api.server_port}"
        finally:
            api.shutdown()
            thread.join()


def call(url, image):
    start = perf_counter()
    try:
        with urlopen(Request(url, data=image), timeout=30) as r:
            return r.status, json.load(r), (perf_counter()-start)*1000
    except HTTPError as e:
        return e.code, {}, (perf_counter()-start)*1000


def image(color):
    buffer = io.BytesIO()
    Image.new("RGB", (64,64), color).save(buffer, format="PNG")
    return buffer.getvalue()


def main():
    source, different = image("red"), image("blue")
    with TemporaryDirectory() as tmp:
        db = Path(tmp) / "catalog.db"
        catalog = production_catalog(db)
        with running(catalog) as base, ThreadPoolExecutor(max_workers=5) as pool:
            target = base + "/products/test/views/front"
            assert call(base+"/search", source)[0] == 200
            search = lambda _: call(base+"/search", source)
            baseline = list(pool.map(search, range(40)))
            writer = pool.submit(lambda: [call(base+f"/products/update/views/{i}", different) for i in range(10)])
            searches = list(pool.map(search, range(40)))
            assert all(status==201 for status,_,_ in writer.result())
            duplicates = list(pool.map(lambda _:call(target,source)[0], range(12)))
            assert sorted(duplicates) == [200]*11+[201]
            assert call(target,different)[0] == 409
        with running(production_catalog(db, seed=False)) as base:
            status, result, _ = call(base+"/search",source)
            assert status==200 and sum(x["view"]=="test/front" for x in result["matches"])==1
        report = {}
        for key, batch in [("search_only",baseline),("during_updates",searches)]:
            assert all(code==200 and len(res["matches"])==10 for code,res,_ in batch)
            report[key] = {"requests":len(batch), **{
                field+"_p95_ms":round(float(np.percentile(values,95)),3)
                for field,values in [
                    ("http",[x[2] for x in batch]),("embedding",[x[1]["embedding_ms"] for x in batch]),
                    ("index",[x[1]["index_ms"] for x in batch])]}}
        report["duplicate_requests"] = len(duplicates)
        report["restart_verified"] = True
    Path("catalog-load.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report)


if __name__ == "__main__":
    main()

"""Download the UCI "Diabetes 130-US Hospitals for Years 1999-2008" dataset.

Source: UCI Machine Learning Repository, dataset id 296
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
Licence: CC BY 4.0. Citation: Strack B. et al. (2014), BioMed Research International,
"Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000
Clinical Database Patient Records", doi:10.1155/2014/781670.

Writes (deterministically, gzip mtime=0 so re-runs do not create new commits):
  data/raw/diabetic_data.csv.gz
  data/raw/IDs_mapping.csv
  data/raw/MANIFEST.json  (source URL + sha256 of each file)

Run with:  python scripts/fetch_data.py
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
OUT = Path(__file__).resolve().parents[1] / "data" / "raw"


def download(url: str, retries: int = 4) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "readmission-risk-api/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"download failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def find_member(zf: zipfile.ZipFile, name: str) -> bytes:
    for info in zf.infolist():
        if Path(info.filename).name.lower() == name.lower():
            return zf.read(info)
        if info.filename.endswith(".zip"):  # nested archive
            inner = zipfile.ZipFile(io.BytesIO(zf.read(info)))
            try:
                return find_member(inner, name)
            except KeyError:
                pass
    raise KeyError(name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    blob = download(URL)
    zf = zipfile.ZipFile(io.BytesIO(blob))
    print("archive members:", [i.filename for i in zf.infolist()])
    data = find_member(zf, "diabetic_data.csv")
    mapping = find_member(zf, "IDs_mapping.csv")

    buf = io.BytesIO()
    with gzip.GzipFile(filename="diabetic_data.csv", mode="wb", fileobj=buf, mtime=0) as gz:
        gz.write(data)
    (OUT / "diabetic_data.csv.gz").write_bytes(buf.getvalue())
    (OUT / "IDs_mapping.csv").write_bytes(mapping)

    manifest = {
        "source": URL,
        "dataset": "Diabetes 130-US Hospitals for Years 1999-2008 (UCI id 296)",
        "licence": "CC BY 4.0",
        "files": {
            "diabetic_data.csv": {
                "sha256_uncompressed": hashlib.sha256(data).hexdigest(),
                "rows_including_header": data.count(b"\n"),
            },
            "IDs_mapping.csv": {"sha256": hashlib.sha256(mapping).hexdigest()},
        },
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

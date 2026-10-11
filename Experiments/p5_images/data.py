"""
Fetch 28 x 28 image sets and store them as uint8 arrays in data/<name>.npz.

    python data.py                      mnist only (10 MB), enough for noise -> digits
    python data.py mnist kmnist         any of: mnist, fashion, kmnist, emnist_letters

Each file holds images x of shape (N, 28, 28) and labels y. The .npz files are
all that dsbm.py reads, so they can be copied to a machine without internet
access; data/raw can be deleted afterwards. emnist_letters needs a 560 MB
archive, the other sets are under 30 MB each.
"""

from __future__ import annotations

import gzip
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"
IDX_URLS = {
    "mnist": "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "fashion": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
    "kmnist": "http://codh.rois.ac.jp/kmnist/dataset/kmnist/",
}
EMNIST_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"


def fetch(url: str, path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(path)
    return path


def read_idx(raw: bytes) -> np.ndarray:
    """Parse an IDX file: images (N, 28, 28) or labels (N,)."""
    ndim = raw[3]
    shape = np.frombuffer(raw, dtype=">u4", count=ndim, offset=4)
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def idx_pair(name: str) -> tuple[np.ndarray, np.ndarray]:
    parts = []
    for file in ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"):
        path = fetch(IDX_URLS[name] + file, DATA / "raw" / f"{name}-{file}")
        parts.append(read_idx(gzip.decompress(path.read_bytes())))
    return parts[0], parts[1]


def emnist_letters() -> tuple[np.ndarray, np.ndarray]:
    """Letters labelled 0..25, upper and lower case merged.

    EMNIST stores images column-major, hence the transpose.
    """
    path = fetch(EMNIST_URL, DATA / "raw" / "emnist-gzip.zip")
    parts = []
    with zipfile.ZipFile(path) as archive:
        for kind in ("images-idx3", "labels-idx1"):
            raw = archive.read(f"gzip/emnist-letters-train-{kind}-ubyte.gz")
            parts.append(read_idx(gzip.decompress(raw)))
    return parts[0].transpose(0, 2, 1), parts[1] - 1


def main() -> None:
    for name in sys.argv[1:] or ["mnist"]:
        out = DATA / f"{name}.npz"
        if out.exists():
            print(f"{out} exists, skipping")
            continue
        x, y = emnist_letters() if name == "emnist_letters" else idx_pair(name)
        np.savez_compressed(out, x=x, y=y)
        print(f"{out}: x {x.shape}, labels {y.min()}..{y.max()}")


if __name__ == "__main__":
    main()

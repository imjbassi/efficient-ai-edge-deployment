"""Download and verify the Imagenette 160 px dataset used by the paper."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path


URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
SHA256 = "64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=Path("data"))
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    archive = args.destination / "imagenette2-160.tgz"
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    actual = digest(archive)
    if actual != SHA256:
        raise RuntimeError(f"Checksum mismatch: {actual}")
    destination = args.destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
            if member.issym() or member.islnk() or member.isdev():
                raise RuntimeError(f"Unsupported archive member: {member.name}")
        bundle.extractall(destination)
    print(args.destination / "imagenette2-160")


if __name__ == "__main__":
    main()

"""Fetch a contiguous, pose-qualified Replica room0 return segment."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import threading

from remotezip import RemoteZip

SOURCE = "https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip"
DEST = Path("datasets/replica/room0_revisit380")
COUNT = 380
LOCAL = threading.local()


def fetch(index):
    if not hasattr(LOCAL, "archive"):
        LOCAL.archive = RemoteZip(SOURCE)
    result = {}
    for kind, suffix in (("frame", "jpg"), ("depth", "png")):
        relative = f"results/{kind}{index:06d}.{suffix}"
        path = DEST / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(LOCAL.archive.read(f"Replica/room0/{relative}"))
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    with RemoteZip(SOURCE) as archive:
        trajectory = archive.read("Replica/room0/traj.txt")
    (DEST / "traj.txt").write_bytes(trajectory)
    manifest = {"source": SOURCE, "scene": "room0", "frames": COUNT,
                "source_frame_ids": [0, COUNT - 1],
                "files": {"traj.txt": hashlib.sha256(trajectory).hexdigest()}}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for index, hashes in enumerate(pool.map(fetch, range(COUNT))):
            manifest["files"].update(hashes)
            if index % 50 == 49:
                print(f"Fetched {index + 1}/{COUNT}", flush=True)
    (DEST / "slice_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

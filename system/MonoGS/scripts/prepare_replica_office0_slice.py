"""Download a reproducible 100-frame Replica office0 RGB-D slice from NICE-SLAM."""

import hashlib
import json
from pathlib import Path

from remotezip import RemoteZip


SOURCE = "https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip"
DEST = Path("datasets/replica/office0_slice100")
COUNT = 100


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = {"source": SOURCE, "scene": "office0", "frames": COUNT, "files": {}}
    with RemoteZip(SOURCE) as archive:
        names = ["Replica/office0/traj.txt"]
        names += [f"Replica/office0/results/frame{i:06d}.jpg" for i in range(COUNT)]
        names += [f"Replica/office0/results/depth{i:06d}.png" for i in range(COUNT)]
        for name in names:
            relative = Path(name).relative_to("Replica/office0")
            output = DEST / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            if not output.exists():
                output.write_bytes(archive.read(name))
            manifest["files"][str(relative)] = hashlib.sha256(output.read_bytes()).hexdigest()
    (DEST / "slice_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

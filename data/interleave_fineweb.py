import argparse
import glob
import os
from pathlib import Path


parser = argparse.ArgumentParser(description="Create interleaved FineWeb/FineWeb-Edu shard symlinks")
parser.add_argument("--fineweb", default="data/fineweb10B/fineweb_train_*.bin")
parser.add_argument("--fineweb_edu", default="data/fineweb_edu100B/fineweb_edu_train_*.bin")
parser.add_argument("--out", default="data/fineweb_mix110B")
parser.add_argument("--edu_per_fineweb", type=int, default=10)
parser.add_argument("--force", action="store_true")
args = parser.parse_args()

fineweb = sorted(Path(p) for p in glob.glob(args.fineweb))
fineweb_edu = sorted(Path(p) for p in glob.glob(args.fineweb_edu))
assert fineweb, f"no files matched {args.fineweb}"
assert fineweb_edu, f"no files matched {args.fineweb_edu}"

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

mixed = []
i = j = 0
while i < len(fineweb) or j < len(fineweb_edu):
    if i < len(fineweb):
        mixed.append(("fineweb", fineweb[i]))
        i += 1
    for _ in range(args.edu_per_fineweb):
        if j < len(fineweb_edu):
            mixed.append(("fineweb_edu", fineweb_edu[j]))
            j += 1

for k, (name, src) in enumerate(mixed):
    dst = out / f"mix_train_{k:06d}_{name}.bin"
    if dst.exists() or dst.is_symlink():
        if not args.force:
            raise FileExistsError(f"{dst} exists; pass --force to overwrite")
        dst.unlink()
    os.symlink(os.path.relpath(src.resolve(), dst.parent), dst)

print(f"linked {len(mixed)} shards in {out}")

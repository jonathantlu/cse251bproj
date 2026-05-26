import argparse
import glob
import os
from pathlib import Path


parser = argparse.ArgumentParser(description="Create interleaved FineWeb/FineWeb-Edu shard symlinks")
parser.add_argument("--fineweb", default="data/fineweb10B/fineweb_train_*.bin")
parser.add_argument("--fineweb_edu", default="data/fineweb_edu100B/fineweb_edu_train_*.bin")
parser.add_argument("--out", default="data/fineweb_mix10B")
parser.add_argument("--edu_per_fineweb", type=int, default=10)
parser.add_argument("--fineweb_offset", type=int, default=-1, help="-1 places FineWeb in the middle of each Edu block")
parser.add_argument("--tokens_per_shard", type=int, default=10**8)
parser.add_argument("--target_tokens", type=int, default=10**10, help="0 links all shards")
parser.add_argument("--force", action="store_true")
args = parser.parse_args()
fineweb_offset = args.edu_per_fineweb // 2 if args.fineweb_offset < 0 else args.fineweb_offset
assert 0 <= fineweb_offset <= args.edu_per_fineweb

fineweb = sorted(Path(p) for p in glob.glob(args.fineweb))
fineweb_edu = sorted(Path(p) for p in glob.glob(args.fineweb_edu))
assert fineweb, f"no files matched {args.fineweb}"
assert fineweb_edu, f"no files matched {args.fineweb_edu}"

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
if args.force:
    for old in out.glob("mix_train_*.bin"):
        old.unlink()

target_shards = None if args.target_tokens == 0 else max(1, args.target_tokens // args.tokens_per_shard)
fineweb_needed = len(fineweb) if target_shards is None else min(len(fineweb), target_shards // (args.edu_per_fineweb + 1))
edu_needed = len(fineweb_edu) if target_shards is None else max(0, target_shards - fineweb_needed)
edu_start = 0 if target_shards is None else max(0, (len(fineweb_edu) - edu_needed) // 2)
fineweb = fineweb[:fineweb_needed]
fineweb_edu = fineweb_edu[edu_start:edu_start + edu_needed]

mixed = []
i = j = 0
while i < len(fineweb) or j < len(fineweb_edu):
    for _ in range(fineweb_offset):
        if j < len(fineweb_edu):
            mixed.append(("fineweb_edu", fineweb_edu[j]))
            j += 1
    if i < len(fineweb):
        mixed.append(("fineweb", fineweb[i]))
        i += 1
    for _ in range(args.edu_per_fineweb - fineweb_offset):
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

print(f"linked {len(mixed)} shards in {out} ({len(fineweb)} fineweb, {len(fineweb_edu)} fineweb_edu starting at edu shard {edu_start})")

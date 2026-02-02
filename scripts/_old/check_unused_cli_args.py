# scripts/check_unused_cli_args.py
from __future__ import annotations

import re
from pathlib import Path

ARG_RE = re.compile(r"add_argument\(\s*['\"](--[a-zA-Z0-9_-]+)['\"]")
DEST_RE = re.compile(r"add_argument\([^)]*dest\s*=\s*['\"]([a-zA-Z0-9_]+)['\"]", re.DOTALL)


def main():
    p = Path("scripts/train_pix2pix.py")
    if not p.exists():
        raise FileNotFoundError(f"Can't find {p}")

    txt = p.read_text(encoding="utf-8", errors="ignore")

    # collect flags
    flags = sorted(set(ARG_RE.findall(txt)))

    # map to arg dest names (argparse default: --foo-bar => args.foo_bar)
    def flag_to_dest(f: str) -> str:
        return f[2:].replace("-", "_")

    dests = [flag_to_dest(f) for f in flags]

    # heuristic: search for "args.<dest>" occurrences
    unused = []
    for f, d in zip(flags, dests):
        if re.search(rf"\bargs\.{re.escape(d)}\b", txt) is None:
            unused.append((f, d))

    print("Checked file:", p)
    print("Total CLI flags:", len(flags))
    if not unused:
        print("[OK] No obvious unused args found (by args.<name> scan).")
    else:
        print("[WARN] Possibly unused args (not found as args.<name>):")
        for f, d in unused:
            print(f"  {f}  (dest: {d})")
        print("\nNote: this is a heuristic; args accessed via getattr() won't be detected.")


if __name__ == "__main__":
    main()

"""Command-line entry points.

Usage:
    python -m sec.cli stage0 --video V.mp4 --reference ref.png --out result.json
"""

from __future__ import annotations

import argparse
import json
import sys

from .stage0_reference import run_stage0


def _stage0(args: argparse.Namespace) -> None:
    result = run_stage0(
        video_path=args.video,
        reference_path=args.reference,
        max_frames=args.max_frames,
    )
    payload = result.to_dict()
    text = json.dumps(payload, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"Wrote {args.out}")
    print(text)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sec")
    sub = parser.add_subparsers(dest="command", required=True)

    p0 = sub.add_parser("stage0", help="Run the Stage 0 zero-bit reference test")
    p0.add_argument("--video", required=True, help="Path to the clip V")
    p0.add_argument("--reference", required=True, help="Path to the still reference image (D)")
    p0.add_argument("--out", default=None, help="Optional path to write JSON results")
    p0.add_argument("--max-frames", dest="max_frames", type=int, default=None)
    p0.set_defaults(func=_stage0)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])

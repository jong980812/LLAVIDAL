#!/usr/bin/env python3
"""
Compare Charades-STA video IDs between a JSON file and a video directory.

- Extract IDs from the JSON's "id" field.
- Extract IDs from the video directory by filename stem (e.g., QQGU3 from QQGU3.mp4).
- Print and optionally save:
    * IDs in JSON but missing in the video dir
    * IDs present in the video dir but missing in JSON
    * Intersections and counts
Usage:
    python compare_charades_ids.py \
        --json /data/jongseo/project/vlm/LLAVIDAL/evaluation/ADLMCQ-AR-Charades.json \
        --video_dir /data/dataset/Charades-STA/raw_videos \
        --exts .mp4 .avi .mov \
        --save_dir ./charades_check_reports
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Set

def load_json_ids(json_path: Path) -> List[str]:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[ERROR] Failed to read JSON: {json_path}\n{e}", file=sys.stderr)
        sys.exit(1)

    ids = []
    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict) and "id" in item:
                ids.append(str(item["id"]).strip())
            else:
                print(f"[WARN] Skipping item #{i} (no 'id' field)", file=sys.stderr)
    else:
        print("[ERROR] JSON root is not a list. Expected a list of objects with 'id'.", file=sys.stderr)
        sys.exit(1)
    return ids

def load_video_ids(video_dir: Path, exts: List[str]) -> List[str]:
    if not video_dir.exists():
        print(f"[ERROR] Video dir not found: {video_dir}", file=sys.stderr)
        sys.exit(1)

    exts = {e.lower() for e in exts}
    ids = []
    for p in video_dir.iterdir():
        if p.is_file():
            suffix = p.suffix.lower()
            if suffix in exts:
                ids.append(p.stem)
    return ids

def write_list(save_dir: Path, name: str, items: List[str]):
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{name}.txt"
    out_path.write_text("\n".join(sorted(items)), encoding="utf-8")
    return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True, help="Path to ADLMCQ-AR-Charades.json")
    parser.add_argument("--video_dir", type=Path, required=True, help="Directory containing videos (e.g., QQGU3.mp4)")
    parser.add_argument("--exts", type=str, nargs="*", default=[".mp4"], help="Video extensions to include")
    parser.add_argument("--save_dir", type=Path, default=None, help="If given, save report .txt files here")
    parser.add_argument("--case_insensitive", action="store_true", help="Case-insensitive ID comparison")
    args = parser.parse_args()

    json_ids = load_json_ids(args.json)
    video_ids = load_video_ids(args.video_dir, args.exts)

    if args.case_insensitive:
        to_norm = lambda s: s.lower()
    else:
        to_norm = lambda s: s

    json_set: Set[str] = {to_norm(x) for x in json_ids}
    video_set: Set[str] = {to_norm(x) for x in video_ids}

    only_in_json = sorted(json_set - video_set)
    only_in_videos = sorted(video_set - json_set)
    intersection = sorted(json_set & video_set)

    print("=== Charades-STA ID Consistency Check ===")
    print(f"JSON file:   {args.json}")
    print(f"Video dir:   {args.video_dir}")
    print(f"Extensions:  {', '.join(args.exts)}")
    print(f"Case-insensitive: {args.case_insensitive}")
    print("\n--- Counts ---")
    print(f"Total JSON IDs:          {len(json_set)} (raw {len(json_ids)})")
    print(f"Total Video IDs:         {len(video_set)} (raw {len(video_ids)})")
    print(f"Present in both:         {len(intersection)}")
    print(f"In JSON only (missing videos): {len(only_in_json)}")
    print(f"In Videos only (missing in JSON): {len(only_in_videos)}")

    def preview(name: str, items: List[str], k: int = 20):
        print(f"\n--- {name} (showing up to {k}) ---")
        for x in items[:k]:
            print(x)
        if len(items) > k:
            print(f"... (+{len(items)-k} more)")

    preview("IDs in JSON but not in videos", only_in_json)
    preview("IDs in videos but not in JSON", only_in_videos)

    if args.save_dir:
        p1 = write_list(args.save_dir, "only_in_json_missing_videos", only_in_json)
        p2 = write_list(args.save_dir, "only_in_videos_missing_json", only_in_videos)
        p3 = write_list(args.save_dir, "intersection_ids", intersection)
        print(f"\nReports saved to:\n- {p1}\n- {p2}\n- {p3}")

if __name__ == "__main__":
    main()

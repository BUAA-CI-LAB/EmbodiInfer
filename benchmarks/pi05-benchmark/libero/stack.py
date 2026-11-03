#!/usr/bin/env python3
"""Join films of one height side by side (shorter ones hold their last frame), or one after another."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("films", nargs="+", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--sequence", action="store_true", help="play the films one after another")
    parser.add_argument("--hold-s", type=float, default=0.5, help="pause after each film in a sequence")
    args = parser.parse_args()

    clips = []
    for path in args.films:
        reader = imageio.get_reader(path)
        fps = reader.get_meta_data().get("fps", 50)
        clips.append([np.asarray(frame) for frame in reader])
        reader.close()
    writer = imageio.get_writer(args.output, fps=fps, codec="libx264", quality=8, macro_block_size=8)
    if args.sequence:
        for frames in clips:
            for frame in frames + [frames[-1]] * int(args.hold_s * fps):
                writer.append_data(frame)
    else:
        length = max(len(frames) for frames in clips)
        for index in range(length):
            writer.append_data(np.concatenate([frames[min(index, len(frames) - 1)] for frames in clips], axis=1))
    writer.close()
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()

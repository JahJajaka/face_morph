"""Re-build the final video from already-generated morph sequence frames,
applying effects.json. Skips the slow morph computation step.

Usage (inside the docker container):
    python rebuild_video.py --sequence morph_results/sequence \
        --output morph_results --pause 120 --tail 59 --fps 30 \
        --effects effects.json
"""
import argparse
import glob
import json
import logging
import os
import subprocess

import main

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("rebuild_video")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pause", type=int, default=0)
    p.add_argument("--tail", type=int, default=0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--effects", default=None)
    p.add_argument("--music", default="music.mp3")
    p.add_argument("--cut", default="00:01:12")
    return p.parse_args()


def main_entry():
    args = parse_args()

    frame_paths = sorted(glob.glob(os.path.join(args.sequence, "frame_*.jpg")))
    if not frame_paths:
        logger.error(f"No frames in {args.sequence}")
        return
    logger.info(f"Found {len(frame_paths)} sequence frames")

    effects = None
    if args.effects:
        with open(args.effects) as f:
            effects = json.load(f).get("effects", [])
        logger.info(f"Loaded {len(effects)} effects from {args.effects}")

    # Clean leftover writer artefacts
    for fname in ("morph_video.mp4", "test_morph_video.mp4"):
        p = os.path.join(args.output, fname)
        if os.path.exists(p):
            os.remove(p)

    main.create_morph_video_from_paths(
        frame_paths,
        args.output,
        fps=args.fps,
        pause_frames=args.pause,
        loops=1,
        interpolation_factor=0,
        apply_smoothing=False,
        temporal_window=3,
        tail_pause=args.tail,
        effects=effects,
    )

    # Mux audio
    morph_video = os.path.join(args.output, "morph_video.mp4")
    final_video = os.path.join(args.output, "morph_final.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", morph_video,
        "-i", args.music,
        "-t", args.cut,
        "-c:v", "copy",
        "-c:a", "aac",
        final_video,
    ]
    logger.info(" ".join(cmd))
    subprocess.run(cmd, check=True)
    logger.info(f"Wrote {final_video}")


if __name__ == "__main__":
    main_entry()

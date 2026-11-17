"""Build the possession file:  python solution/run.py --team <team_name>"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, source_frames  # noqa: E402
import decode  # noqa: E402
import features  # noqa: E402
import tags  # noqa: E402

# Every constant of the method. Reasons for each value: TECHNICAL.md, "Parameters".
PARAMS = {
    # who has the ball (metres)
    "control_radius_m": 2.0,
    "contest_gap_m": 1.0,
    "dist_clip_m": 5.0,
    "w_control": 1.0,
    "w_margin": 0.2,
    # team shape when the ball is not seen
    "shape_window_s": 1.0,
    "shape_unit_m": 2.5,
    "shape_cap": 2.0,
    "w_shape": 0.15,
    # is the ball alive (seconds, m/s)
    "w_ball_in_play": 0.3,
    "w_ball_out": 1.5,
    "out_window_s": 0.5,
    "w_invisible": 0.6,
    "invisible_window_s": 2.0,
    "speed_window_s": 1.0,
    "slow_speed": 1.0,
    "fast_speed": 1.6,
    "speed_band": 0.5,
    "w_slow": 0.8,
    "w_fast": 0.5,
    # restarts the tags miss
    "still_speed": 1.0,
    "still_window_s": 1.0,
    "w_still": 0.8,
    "taker_window_s": 0.5,
    "w_taker": 1.0,
    "w_exit": 1.2,
    # smart tagging
    "tag_phase": 0.15,
    "tag_dead": 0.6,
    "tag_setpiece_team": 0.2,
    "setpiece_max_wait_s": 15.0,
    "kick_speed": 5.0,
    "rest_before_kick_s": 0.5,
    "setpiece_follow_s": 5.0,
    "goal_dead_s": 30.0,
    # viterbi
    "switch_cost": 12.0,
    "dead_cost": 20.0,
}


def write_submission(path, labels):
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))


def build_labels(game=GAME, params=PARAMS):
    """The whole pipeline: tracking + tagging -> per-frame evidence -> Viterbi -> letters."""
    fps = GAMES[game]["fps"]
    n_frames = source_frames("tactical", game)
    feats = features.build(game)
    tag_team, tag_dead = tags.evidence(n_frames, feats, params, fps)
    score = decode.emissions(feats, tag_team, tag_dead, params, fps)
    return decode.decode(score, half_video_blocks(game), n_frames, params)


def main():
    parser = argparse.ArgumentParser(description="Build and validate the possession CSV.")
    parser.add_argument("--team", default="submission", help="output file is <team>.csv")
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results"))
    args = parser.parse_args()

    started = time.time()
    labels = build_labels()

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "{}.csv".format(args.team))
    write_submission(path, labels)
    counts = {k: int(np.sum(labels == k)) for k in "HADX"}
    print("wrote {}  {}  ({:.0f} s)".format(os.path.relpath(path, ROOT), counts, time.time() - started))

    validator = os.path.join(ROOT, "evaluation", "validate_submission.py")
    return subprocess.run([sys.executable, validator, path], check=False).returncode


if __name__ == "__main__":
    sys.exit(main())

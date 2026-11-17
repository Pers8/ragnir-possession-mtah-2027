"""Build the possession file: python solution/run.py --team <team_name>"""

import argparse
import os
import subprocess
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks, source_frames  # noqa: E402
import decode  # noqa: E402
import features  # noqa: E402
import tags  # noqa: E402

PARAMS = {
    # tracking: who has the ball
    "control_radius_m": 2.0,
    "contest_gap_m": 1.0,
    "dist_clip_m": 5.0,
    "w_control": 1.0,
    "w_margin": 0.2,
    # tracking: is the ball alive
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
    # smart tagging
    "tag_phase": 0.15,
    "tag_dead": 0.6,
    "tag_setpiece_team": 0.2,
    "setpiece_lead_s": 3.5,
    "kickoff_lead_s": 3.5,
    "setpiece_follow_s": 5.0,
    # viterbi
    "switch_cost": 12.0,
    "dead_cost": 20.0,
}


def write_submission(path, labels):
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", default="submission")
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results"))
    args = parser.parse_args()
    warnings.simplefilter("ignore")
    started = time.time()

    n_frames = source_frames("tactical", GAME)
    f = features.build(GAME)
    tag_team, tag_dead = tags.evidence(n_frames, f["teams"], PARAMS)
    score = decode.emissions(f, tag_team, tag_dead, PARAMS)
    labels = decode.decode(score, half_video_blocks(GAME), n_frames, PARAMS)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "{}.csv".format(args.team))
    write_submission(path, labels)
    counts = {k: int(np.sum(labels == k)) for k in "HADX"}
    print("wrote {}  {}  ({:.0f} s)".format(os.path.relpath(path, ROOT), counts, time.time() - started))

    subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"), path],
                   check=False)


if __name__ == "__main__":
    main()

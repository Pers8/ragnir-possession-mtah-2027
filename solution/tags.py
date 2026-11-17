"""Smart Tagging intervals turned into weak per-frame evidence for H, A and dead ball."""

import difflib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
import load_smart_tagging  # noqa: E402

ATTACKING = {
    "build up", "progression", "creating chances", "attacking transition", "long ball",
    "ball in final third", "ball in the box", "players in the box",
}
DEFENDING = {
    "defending in attacking third", "defending in middle third", "defending in defensive third",
    "defensive transition",
}
SET_PIECE = "set pieces"
KICKOFF_TYPES = {"kick off (start)", "kick off (after goal)"}


def side_of(team_tag, teams):
    if not team_tag or team_tag.strip().upper() == "N/A":
        return None
    scores = [difflib.SequenceMatcher(None, team_tag.lower(), name.lower()).ratio() for name in teams]
    return int(np.argmax(scores))


def evidence(n_frames, teams, params):
    """(team_score[n, 2], dead_score[n]) built from the tagged intervals."""
    team_score = np.zeros((n_frames, 2), dtype=np.float32)
    dead_score = np.zeros(n_frames, dtype=np.float32)
    fps = 25
    for event in load_smart_tagging.load():
        code = (event["code"] or "").strip().lower()
        side = side_of(event["tags"].get("Team"), teams)
        start = max(0, event["start_frame"])
        end = min(n_frames, event["end_frame"])
        if end <= start:
            continue

        if code in ATTACKING and side is not None:
            team_score[start:end, side] += params["tag_phase"]
        elif code in DEFENDING and side is not None:
            team_score[start:end, 1 - side] += params["tag_phase"]
        elif code == SET_PIECE:
            kind = (event["tags"].get("Type") or "").strip().lower()
            lead = params["kickoff_lead_s"] if kind in KICKOFF_TYPES else params["setpiece_lead_s"]
            restart = min(end, start + int(lead * fps))
            dead_score[start:restart] += params["tag_dead"]
            if side is not None:
                after = min(end, restart + int(params["setpiece_follow_s"] * fps))
                team_score[restart:after, side] += params["tag_setpiece_team"]
    return team_score, dead_score


if __name__ == "__main__":
    from collections import Counter
    events = load_smart_tagging.load()
    teams = ("Barcelona", "Atletic Club")
    print(Counter((e["tags"].get("Team"), side_of(e["tags"].get("Team"), teams)) for e in events))

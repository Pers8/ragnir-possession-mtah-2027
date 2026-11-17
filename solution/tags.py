"""Smart Tagging intervals turned into weak per-frame evidence for H, A and dead ball."""

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
AFTER_GOAL = "kick off (after goal)"
KICKOFF_TYPES = {"kick off (start)", AFTER_GOAL}
NO_TEAM = {"", "n/a"}


def _clean(text):
    return (text or "").strip().lower()


def side_of(team_tag, teams):
    """0 for home, 1 for away, None for an untagged team. Unknown names are an error, not a guess."""
    tag = _clean(team_tag)
    if tag in NO_TEAM:
        return None
    names = [_clean(name) for name in teams]
    if tag not in names:
        raise ValueError("Smart Tagging team {!r} matches neither {}".format(team_tag, teams))
    return names.index(tag)


def evidence(n_frames, teams, params, fps):
    """(team_score[n, 2], dead_score[n]) accumulated over every tagged interval."""
    team_score = np.zeros((n_frames, 2), dtype=np.float32)
    dead_score = np.zeros(n_frames, dtype=np.float32)

    for event in load_smart_tagging.load():
        code = _clean(event["code"])
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
            kind = _clean(event["tags"].get("Type"))
            lead_s = params["kickoff_lead_s"] if kind in KICKOFF_TYPES else params["setpiece_lead_s"]
            restart = min(end, start + int(lead_s * fps))
            dead_score[start:restart] += params["tag_dead"]
            if kind == AFTER_GOAL:
                celebration = max(0, start - int(params["goal_dead_s"] * fps))
                dead_score[celebration:start] += params["tag_dead"]
            if side is not None:
                after = min(end, restart + int(params["setpiece_follow_s"] * fps))
                team_score[restart:after, side] += params["tag_setpiece_team"]

    return team_score, dead_score

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


def find_restart(feats, start, stop, params, fps):
    """First frame in [start, stop) where the ball is struck after resting. Resting means half a
    second with the ball unseen or slower than walking pace, ending at most half a second before
    the kick (possibly before the tag opens); struck means faster than kick_speed for two frames.
    Returns start when there is no such kick, so the tag then adds no dead-ball time."""
    rest = int(params["rest_before_kick_s"] * fps)
    first = max(0, start - 2 * rest)
    speed = feats["ball_speed"][first:stop]
    quiet = ~(feats["visible"][first:stop] & (speed >= params["still_speed"]))
    moving = feats["visible"][first:stop] & (speed > params["kick_speed"])
    rested = np.convolve(quiet, np.ones(rest), "full")[:len(quiet)] >= rest
    for k in range(start - first, len(quiet) - 1):
        if moving[k] and moving[k + 1] and rested[max(0, k - rest):k].any():
            return first + k
    return start


def evidence(n_frames, feats, params, fps):
    """(team_score[n, 2], dead_score[n]) accumulated over every tagged interval."""
    teams = feats["teams"]
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
            stop = min(end, start + int(params["setpiece_max_wait_s"] * fps))
            restart = find_restart(feats, start, stop, params, fps)
            dead_score[start:restart] += params["tag_dead"]
            if kind == AFTER_GOAL:
                celebration = max(0, start - int(params["goal_dead_s"] * fps))
                dead_score[celebration:start] += params["tag_dead"]
            if side is not None:
                after = min(end, restart + int(params["setpiece_follow_s"] * fps))
                team_score[restart:after, side] += params["tag_setpiece_team"]

    return team_score, dead_score

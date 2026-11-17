"""Per-frame tracking features: where the ball is, how close each team is to it, how fast play is."""

import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, require, source_frames  # noqa: E402
from baseline import team_names, team_of_each_track  # noqa: E402

CACHE = os.path.join(HERE, "cache", "atd.npz")
BOUNDS_MARGIN = 0.005  # normalised units, ~0.5 m: the line itself is in play
HOME = 0


# --- parsing ---

def _xys(group):
    """'x,y,speed' -> three floats, NaN when the tracker has no observation."""
    if group[0] == "N":
        return np.nan, np.nan, np.nan
    x, y, speed = group.split(",")
    return float(x), float(y), float(speed)


def parse_atd(raw_path, n_frames, n_slots):
    """Raw ATD text -> ball[n, 3], players[n, slots, 3], recorded[n], indexed by video frame."""
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    recorded = np.zeros(n_frames, dtype=bool)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1
            if not 0 <= frame < n_frames:
                continue
            recorded[frame] = True
            ball[frame] = _xys(parts[2].strip())
            for slot, group in enumerate(parts[1].split(";")[:n_slots]):
                if group[0] != "N":
                    players[frame, slot] = _xys(group)
    return ball, players, recorded


def load_tracking(game=GAME):
    """Parsed tracking, cached in solution/cache/ and rebuilt if the raw file changes."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    source_size = os.path.getsize(raw_path)
    if os.path.exists(CACHE):
        cached = np.load(CACHE)
        if "source_size" in cached and int(cached["source_size"]) == source_size:
            return cached["ball"], cached["players"], cached["recorded"], cached["team"]

    team = team_of_each_track(meta_path)
    ball, players, recorded = parse_atd(raw_path, source_frames("tactical", game), len(team))
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, ball=ball, players=players, recorded=recorded, team=team,
             source_size=source_size)
    return ball, players, recorded, team


# --- features ---

def build(game=GAME):
    """Dict of per-frame arrays used by decode.emissions, plus the (home, away) club names."""
    ball, players, recorded, team = load_tracking(game)
    scale = np.array(GAMES[game]["pitch"], dtype=np.float32)

    ball_xy = ball[:, :2] * scale
    player_xy = players[:, :, :2] * scale
    visible = ~np.isnan(ball[:, 0])
    on_pitch = (ball[:, :2] >= -BOUNDS_MARGIN) & (ball[:, :2] <= 1 + BOUNDS_MARGIN)
    in_bounds = visible & np.all(on_pitch, axis=1)

    dist = np.linalg.norm(player_xy - ball_xy[:, None, :], axis=2)
    home = (team == HOME)[None, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN frames -> NaN, handled downstream
        d_home = np.nanmin(np.where(home, dist, np.nan), axis=1)
        d_away = np.nanmin(np.where(~home, dist, np.nan), axis=1)
        player_speed = np.nanmean(players[:, :, 2], axis=1)
        taker_outside = np.nanmax(np.where(dist <= TAKER_RADIUS_M,
                                           _outside_m(player_xy, scale), np.nan), axis=1)

    fps = GAMES[game]["fps"]
    return {
        "recorded": recorded,
        "visible": visible,
        "in_bounds": in_bounds,
        "d_home": d_home,
        "d_away": d_away,
        "player_speed": player_speed,
        "ball_speed": ball_speed(ball_xy, fps),
        "taker_outside": np.nan_to_num(taker_outside, nan=0.0) >= TAKER_OUTSIDE_M,
        "exit_runs": exit_runs(ball_xy, visible, scale, fps),
        "teams": team_names(require(["atd_meta"], game)[0]),
    }


# --- dead-ball cues ---

TAKER_RADIUS_M = 2.0     # a throw-in / corner taker is touching the ball
TAKER_OUTSIDE_M = 0.5    # and standing clearly beyond the line
EXIT_LINE_M = 3.0        # ball last seen this close to a line ...
EXIT_SPEED = 2.0         # ... moving outward at least this fast (m/s)


def _outside_m(xy, scale):
    """How far each point is beyond the nearest pitch line, in metres (<= 0 inside)."""
    return np.maximum.reduce([-xy[..., 0], xy[..., 0] - scale[0], -xy[..., 1], xy[..., 1] - scale[1]])


def ball_speed(ball_xy, fps):
    """Ball speed in m/s from positions, NaN where either neighbouring frame is missing."""
    speed = np.full(len(ball_xy), np.nan)
    speed[1:] = np.linalg.norm(np.diff(ball_xy, axis=0), axis=1) * fps
    return speed


def exit_runs(ball_xy, visible, scale, fps):
    """True on every frame of a gap in the ball track that starts with the ball leaving the pitch."""
    out = np.zeros(len(visible), dtype=bool)
    seen = np.flatnonzero(visible)
    gaps = np.flatnonzero(np.diff(seen) > 1)
    for g in gaps:
        last, back = seen[g], seen[g] - 2
        if back < 0 or not visible[back]:
            continue
        velocity = (ball_xy[last] - ball_xy[back]) * fps / 2
        pos = ball_xy[last]
        to_line = np.array([pos[0], scale[0] - pos[0], pos[1], scale[1] - pos[1]])
        outward = np.array([-velocity[0], velocity[0], -velocity[1], velocity[1]])
        line = int(np.argmin(to_line))
        if to_line[line] <= EXIT_LINE_M and outward[line] >= EXIT_SPEED:
            out[last + 1:seen[g + 1]] = True
    return out


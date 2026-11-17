"""Per-frame tracking features: ball position, distance from the ball to each team, movement."""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, require, source_frames  # noqa: E402
from baseline import team_of_each_track, team_names  # noqa: E402

CACHE = os.path.join(HERE, "cache", "atd.npz")
BOUNDS_MARGIN = 0.005


# --- parsing ---

def _triple(group):
    if group[0] == "N":
        return np.nan, np.nan, np.nan
    x, y, s = group.split(",")
    return float(x), float(y), float(s)


def parse_atd(raw_path, n_frames, n_slots):
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
            ball[frame] = _triple(parts[2].strip())
            for slot, group in enumerate(parts[1].split(";")[:n_slots]):
                if group[0] != "N":
                    players[frame, slot] = _triple(group)
    return ball, players, recorded


def load_raw(game=GAME, use_cache=True):
    if use_cache and os.path.exists(CACHE):
        data = np.load(CACHE)
        return data["ball"], data["players"], data["recorded"], data["team"]
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    team = team_of_each_track(meta_path)
    ball, players, recorded = parse_atd(raw_path, source_frames("tactical", game), len(team))
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, ball=ball, players=players, recorded=recorded, team=team)
    return ball, players, recorded, team


# --- features ---

def build(game=GAME):
    ball, players, recorded, team = load_raw(game)
    length, width = GAMES[game]["pitch"]
    scale = np.array([length, width], dtype=np.float32)

    ball_xy = ball[:, :2] * scale
    player_xy = players[:, :, :2] * scale
    visible = ~np.isnan(ball[:, 0])
    in_bounds = visible & np.all((ball[:, :2] >= -BOUNDS_MARGIN) & (ball[:, :2] <= 1 + BOUNDS_MARGIN), axis=1)

    dist = np.linalg.norm(player_xy - ball_xy[:, None, :], axis=2)
    home = team == 0
    with np.errstate(invalid="ignore"):
        d_home = np.nanmin(np.where(home[None, :], dist, np.nan), axis=1)
        d_away = np.nanmin(np.where(~home[None, :], dist, np.nan), axis=1)
        player_speed = np.nanmean(players[:, :, 2], axis=1)
        n_players = np.sum(~np.isnan(players[:, :, 0]), axis=1)

    return {
        "recorded": recorded,
        "visible": visible,
        "in_bounds": in_bounds,
        "ball_xy": ball_xy,
        "ball_speed": ball[:, 2],
        "d_home": d_home,
        "d_away": d_away,
        "player_speed": player_speed,
        "n_players": n_players,
        "teams": team_names(require(["atd_meta"], game)[0]),
    }


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")
    f = build()
    q = [5, 25, 50, 75, 95]
    print("recorded", int(f["recorded"].sum()), "visible", int(f["visible"].sum()), "in_bounds", int(f["in_bounds"].sum()))
    print("ball speed q", np.nanpercentile(f["ball_speed"], q).round(2))
    print("player speed q", np.nanpercentile(f["player_speed"], q).round(2))
    print("nearest dist q", np.nanpercentile(np.fmin(f["d_home"], f["d_away"]), q).round(2))
    print("n players q", np.percentile(f["n_players"][f["recorded"]], q))

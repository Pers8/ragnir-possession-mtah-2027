"""Per-frame evidence for H / A / D, then a Viterbi pass that turns it into whole possessions."""

import numpy as np

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])


# --- helpers ---

def rolling_mean(values, half_width):
    values = np.nan_to_num(values.astype(np.float64), nan=0.0)
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    index = np.arange(len(values))
    low = np.maximum(0, index - half_width)
    high = np.minimum(len(values), index + half_width + 1)
    return (cumulative[high] - cumulative[low]) / (high - low)


# --- emissions ---

def emissions(f, tag_team, tag_dead, p):
    n = len(f["visible"])
    score = np.zeros((n, 3), dtype=np.float64)
    fps = 25

    playable = f["in_bounds"] & ~np.isnan(f["d_home"]) & ~np.isnan(f["d_away"])
    d_home = np.where(playable, f["d_home"], np.inf)
    d_away = np.where(playable, f["d_away"], np.inf)
    nearest = np.fmin(d_home, d_away)

    margin = np.clip(np.nan_to_num(d_away - d_home, nan=0.0, posinf=0.0, neginf=0.0),
                     -p["dist_clip_m"], p["dist_clip_m"]) / p["dist_clip_m"]
    margin[~playable] = 0.0
    score[:, H] += p["w_margin"] * margin
    score[:, A] -= p["w_margin"] * margin

    control = playable & (nearest <= p["control_radius_m"])
    contested = control & (np.abs(d_home - d_away) <= p["contest_gap_m"])
    clear = control & ~contested
    score[clear & (d_home < d_away), H] += p["w_control"]
    score[clear & (d_away < d_home), A] += p["w_control"]

    alive_vis = f["in_bounds"].astype(np.float64)
    score[:, D] -= p["w_ball_in_play"] * alive_vis

    out = f["visible"] & ~f["in_bounds"]
    out_share = rolling_mean(out, int(p["out_window_s"] * fps))
    score[:, D] += p["w_ball_out"] * out_share

    invisible_share = rolling_mean(~f["visible"], int(p["invisible_window_s"] * fps))
    score[:, D] += p["w_invisible"] * invisible_share ** 2

    speed = rolling_mean(f["player_speed"], int(p["speed_window_s"] * fps))
    slow = np.clip((p["slow_speed"] - speed) / p["speed_band"], 0, 1)
    fast = np.clip((speed - p["fast_speed"]) / p["speed_band"], 0, 1)
    score[:, D] += p["w_slow"] * slow
    score[:, D] -= p["w_fast"] * fast

    score[:, H] += tag_team[:, 0] - tag_team[:, 1]
    score[:, A] += tag_team[:, 1] - tag_team[:, 0]
    score[:, D] += tag_dead
    return score


# --- viterbi ---

def viterbi(score, switch_cost, dead_cost):
    n = len(score)
    trans = np.array([
        [0.0, -switch_cost, -dead_cost],
        [-switch_cost, 0.0, -dead_cost],
        [-dead_cost, -dead_cost, 0.0],
    ])
    back = np.zeros((n, 3), dtype=np.int8)
    total = score[0].copy()
    for t in range(1, n):
        cand = total[:, None] + trans
        back[t] = np.argmax(cand, axis=0)
        total = cand[back[t], (0, 1, 2)] + score[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(total))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(score, blocks, n_frames, p):
    labels = np.full(n_frames, "X", dtype="<U1")
    for _, first, last in blocks:
        path = viterbi(score[first:last + 1], p["switch_cost"], p["dead_cost"])
        labels[first:last + 1] = LETTERS[path]
    return labels

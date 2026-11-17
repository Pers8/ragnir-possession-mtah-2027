"""Per-frame evidence for H / A / D, then a Viterbi pass that turns it into whole possessions."""

import numpy as np

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])


# --- helpers ---

def rolling_mean(values, half_width):
    """Centred mean over +/- half_width frames, NaN counted as 0, shrinking at the edges."""
    values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0)
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    index = np.arange(len(values))
    low = np.maximum(0, index - half_width)
    high = np.minimum(len(values), index + half_width + 1)
    return (cumulative[high] - cumulative[low]) / (high - low)


def fit_logistic(x, y, iterations=25):
    """1-D logistic regression P(y | x) = sigmoid(slope * x + bias), Newton's method."""
    X = np.column_stack([x, np.ones(len(x))])
    w = np.zeros(2)
    for _ in range(iterations):
        prob = 1 / (1 + np.exp(-X @ w))
        gradient = X.T @ (y - prob)
        hessian = (X * (prob * (1 - prob))[:, None]).T @ X
        w += np.linalg.solve(hessian + 1e-9 * np.eye(2), gradient)
    return w[0], w[1]


# --- emissions ---

def emissions(feats, tag_team, tag_dead, params, fps):
    """score[n, 3]: how much each frame's evidence favours H, A and D. Higher is more likely."""
    p = params
    score = np.zeros((len(feats["visible"]), 3), dtype=np.float64)

    with np.errstate(invalid="ignore"):
        # who has the ball
        playable = feats["in_bounds"] & ~np.isnan(feats["d_home"]) & ~np.isnan(feats["d_away"])
        d_home = np.where(playable, feats["d_home"], np.inf)
        d_away = np.where(playable, feats["d_away"], np.inf)
        nearest = np.fmin(d_home, d_away)

        gap = np.nan_to_num(d_away - d_home, nan=0.0, posinf=0.0, neginf=0.0)
        margin = np.clip(gap, -p["dist_clip_m"], p["dist_clip_m"]) / p["dist_clip_m"]
        margin[~playable] = 0.0
        score[:, H] += p["w_margin"] * margin
        score[:, A] -= p["w_margin"] * margin

        control = playable & (nearest <= p["control_radius_m"])
        contested = control & (np.abs(d_home - d_away) <= p["contest_gap_m"])
        clear = control & ~contested
        score[clear & (d_home < d_away), H] += p["w_control"]
        score[clear & (d_away < d_home), A] += p["w_control"]

        # team shape where the ball cannot tell us, calibrated on the clear-control frames
        shape = rolling_mean(feats["shape_diff"], int(p["shape_window_s"] * fps))
        usable = feats["shape_ok"] & ~np.isnan(feats["shape_diff"])
        known = clear & usable
        slope, bias = fit_logistic(shape[known], (d_home < d_away)[known])
        log_odds = np.clip(slope * shape + bias, -p["shape_cap"], p["shape_cap"])
        blind = usable & ~playable
        half = 0.5 * p["w_shape"] * log_odds[blind]
        score[blind, H] += half
        score[blind, A] -= half
        score[blind, D] += np.abs(half)  # shape says which team, not whether the ball is alive

        # is the ball alive
        score[:, D] -= p["w_ball_in_play"] * feats["in_bounds"]

        out = feats["visible"] & ~feats["in_bounds"]
        score[:, D] += p["w_ball_out"] * rolling_mean(out, int(p["out_window_s"] * fps))

        unseen = rolling_mean(~feats["visible"], int(p["invisible_window_s"] * fps))
        score[:, D] += p["w_invisible"] * unseen ** 2

        speed = rolling_mean(feats["player_speed"], int(p["speed_window_s"] * fps))
        slow = np.clip((p["slow_speed"] - speed) / p["speed_band"], 0, 1)
        fast = np.clip((speed - p["fast_speed"]) / p["speed_band"], 0, 1)
        score[:, D] += p["w_slow"] * slow
        score[:, D] -= p["w_fast"] * fast

        # restarts the tags miss
        still = feats["visible"] & (feats["ball_speed"] < p["still_speed"])
        score[:, D] += p["w_still"] * rolling_mean(still, int(p["still_window_s"] * fps)) ** 2
        taker = rolling_mean(feats["taker_outside"], int(p["taker_window_s"] * fps))
        score[:, D] += p["w_taker"] * taker
        score[:, D] += p["w_exit"] * feats["exit_runs"]

    # smart tagging
    score[:, H] += tag_team[:, 0] - tag_team[:, 1]
    score[:, A] += tag_team[:, 1] - tag_team[:, 0]
    score[:, D] += tag_dead
    return score


# --- viterbi ---

def viterbi(score, switch_cost, dead_cost):
    """Best H/A/D path through score[n, 3] when every change of label pays a cost."""
    n = len(score)
    trans = np.array([
        [0.0, -switch_cost, -dead_cost],
        [-switch_cost, 0.0, -dead_cost],
        [-dead_cost, -dead_cost, 0.0],
    ])
    states = np.arange(3)
    back = np.zeros((n, 3), dtype=np.int8)
    total = score[0].copy()
    for t in range(1, n):
        candidates = total[:, None] + trans
        back[t] = np.argmax(candidates, axis=0)
        total = candidates[back[t], states] + score[t]

    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(total))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(score, halves, n_frames, params):
    """Letters for every video frame: Viterbi inside each half, X outside them."""
    labels = np.full(n_frames, "X", dtype="<U1")
    for _, first, last in halves:
        path = viterbi(score[first:last + 1], params["switch_cost"], params["dead_cost"])
        labels[first:last + 1] = LETTERS[path]
    return labels

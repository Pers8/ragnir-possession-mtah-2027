"""Unit tests on synthetic inputs, no match data needed:  python -m unittest discover tests"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "solution"))
import decode  # noqa: E402
import tags  # noqa: E402

H, A, D = decode.H, decode.A, decode.D


def evidence(*runs):
    """runs of (state, n_frames, strength) -> score[n, 3]."""
    rows = []
    for state, n, strength in runs:
        row = np.zeros(3)
        row[state] = strength
        rows += [row] * n
    return np.array(rows)


class ViterbiTest(unittest.TestCase):

    def test_short_contrary_blip_does_not_switch(self):
        score = evidence((H, 50, 1.0), (A, 5, 1.0), (H, 50, 1.0))
        path = decode.viterbi(score, switch_cost=12.0, dead_cost=20.0)
        self.assertTrue(np.all(path == H))

    def test_sustained_evidence_switches_once(self):
        score = evidence((H, 50, 1.0), (A, 50, 1.0))
        path = decode.viterbi(score, switch_cost=12.0, dead_cost=20.0)
        self.assertEqual(int(np.sum(np.diff(path) != 0)), 1)
        self.assertEqual(path[0], H)
        self.assertEqual(path[-1], A)

    def test_no_evidence_keeps_possession(self):
        score = evidence((A, 30, 1.0), (H, 200, 0.0))
        path = decode.viterbi(score, switch_cost=12.0, dead_cost=20.0)
        self.assertTrue(np.all(path == A))

    def test_stoppage_becomes_dead_ball(self):
        score = evidence((H, 50, 1.0), (D, 100, 1.0), (A, 50, 1.0))
        path = decode.viterbi(score, switch_cost=12.0, dead_cost=20.0)
        self.assertEqual(list(decode.LETTERS[path[[0, 100, -1]]]), ["H", "D", "A"])


class HelpersTest(unittest.TestCase):

    def test_rolling_mean_shrinks_at_edges(self):
        out = decode.rolling_mean(np.array([1.0, 0.0, 0.0, 0.0]), 1)
        np.testing.assert_allclose(out, [0.5, 1 / 3, 0.0, 0.0])

    def test_team_tags_map_exactly(self):
        teams = ("Barcelona", "Atletic Club")
        self.assertEqual(tags.side_of("Barcelona", teams), 0)
        self.assertEqual(tags.side_of(" atletic club ", teams), 1)
        self.assertIsNone(tags.side_of("N/A", teams))
        with self.assertRaises(ValueError):
            tags.side_of("Real Madrid", teams)


if __name__ == "__main__":
    unittest.main()

# Team Ragnir — More than a Hack 2027, Phase 1: Ball Possession

Frame-level ball possession (`H` / `A` / `D` / `X`) for FC Barcelona vs Athletic Club, built from
Metrica Sports' automatic tracking and Smart Tagging.

**Method in one line:** per-frame evidence for *home / away / dead ball* from the tracking and the
Smart Tagging, then a Viterbi pass with a cost for every change of label, so possession only switches when
the other team clearly controls the ball.

- Technical document: [docs/Ragnir.pdf](docs/Ragnir.pdf) (LaTeX source in `docs/`)
- Original challenge README (scorer, starter code, rules): [CHALLENGE_README.md](CHALLENGE_README.md)

## Team

| | |
|---|---|
| Peres | [@Pers8](https://github.com/Pers8) |
| Breal | [@Bral9](https://github.com/Bral9) |
| | [@mahouboy23](https://github.com/mahouboy23) |

## Install

Python 3.11 (tested), 3.8+ should work.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS / Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## Data

The match data is licensed for the competition and is **not** in this repository. Put the tactical
files in `data/tactical/`:

```
data/tactical/BAR-ATH_tactical_atd.xml
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_smart_tagging.xml
```

## Reproduce the submission

```bash
python solution/run.py --team Ragnir
```

Writes `evaluation/results/Ragnir.csv` and runs the official validator on it. About 1 minute the
first time (the tracking is parsed and cached in `solution/cache/`), ~15 s after. No randomness:
the same data always gives the same file.

Optional, score it on the public 5 minutes:

```bash
python evaluation/run_evaluation.py
```

## Tests

```bash
python -m unittest discover tests
```

Unit tests on synthetic inputs (no match data needed): the Viterbi step ignores short contrary
blips, switches once on sustained evidence, keeps possession through missing evidence and marks
stoppages as dead ball; the restart and kick detectors and the team-shape measure behave as
expected; team names in the tagging must match the tracking exactly.

## Code

| file | what it does |
|---|---|
| `solution/features.py` | reads the tracking: ball position, distance from the ball to each team, player speed |
| `solution/tags.py` | turns Smart Tagging phases and set pieces into weak hints |
| `solution/decode.py` | per-frame scores for H / A / D, then Viterbi with switching costs |
| `solution/run.py` | all parameters + the one command that builds the file |
| `tests/test_solution.py` | unit tests for the decoding and the tag mapping |

Everything else (`games.py`, `starter/`, `evaluation/`) is the organisers' starter repository,
unchanged, under its MIT licence.

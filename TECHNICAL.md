# Team Ragnir — Technical document

More than a Hack 2027, Phase 1: Ball Possession. Code: see README.md for the repository and the reproduction command.

## Approach

For every frame we score three options — Barcelona has the ball (`H`), Athletic has it (`A`),
or the ball is dead (`D`) — from the tracking and the Smart Tagging. Then a **Viterbi pass**
(the decoding step of a hidden Markov model) picks the single best sequence of labels for each
half, where **changing label has a cost**.

Why this rather than a sliding majority vote (the baseline): the ranking metric rewards
recovering *whole* possessions. A majority vote still flickers wherever the evidence is mixed,
and every flicker splits a possession in two. With a switching cost, possession only changes
when the evidence for the other team is strong enough, for long enough, to pay for the switch.
Weak or missing evidence (ball hidden, players in a duel) costs nothing, so the current
possession simply carries on.

No machine learning and no training data: every number is a hand-set constant with a football
reason (table below). The public 5-minute ground truth was only used to check the file and the
frame alignment.

## Assets used

| asset | used for |
|---|---|
| Tactical tracking (ATD) | ball position, distance from the ball to the nearest player of each team, average player speed |
| Tracking metadata | which track slot belongs to which team |
| Smart Tagging | weak team hints from attacking / defending phases; dead-ball hints just before set pieces |
| Homography | not used — the tracking is already in pitch coordinates |
| Videos, broadcast tracking | not used in this phase (the broadcast feed is not synchronised; video models were out of scope) |

## Evidence per frame

**Who has the ball**

- A player within **2 m** of the ball who is clearly closer than any opponent → strong evidence for their team.
- Both teams within **1 m** of each other near the ball → a duel, no evidence (the possession does not change during a challenge).
- Otherwise a small bonus for whichever team is closer, capped at a 5 m difference (a pass in the air is near nobody).
- Smart Tagging: an attacking phase gives its team a small bonus; a defending phase gives the other team a small bonus.

**Is the ball dead**

- Ball visible but outside the pitch lines → strong dead-ball evidence.
- Ball not seen for a whole 2-second stretch → some dead-ball evidence.
- Players slow on average (under ~1 m/s, measured over 1 s) → dead-ball evidence; players fast → evidence that the ball is alive.
- Smart Tagging set piece → dead ball in its first 3.5 s (the restart), then a bonus for the team taking it for 5 s.

## Possession rules

- **Loose balls / passes:** nobody near the ball means no new evidence, so the team that last had it keeps possession until the other side clearly controls it.
- **Challenges:** players from both teams within 1 m of each other give no evidence, so a duel does not change possession by itself.
- **Turnovers:** the other team must clearly control the ball for about half a second before possession switches.
- **Restarts:** the ball is dead from when it leaves the pitch (or play stops) until the restart. Who takes the restart is not labelled — it is `D`, as the rules ask.
- **Halftime and before kickoff:** `X` (outside the tracked halves).

## Parameters

| parameter | value | reason |
|---|---|---|
| control radius | 2 m | a player can play a ball about one stride away |
| duel gap | 1 m | both teams that close = contested ball |
| distance cap | 5 m | further away, "closer team" says little |
| switch cost | 12 | about 0.5 s of clean control by the other team before a turnover |
| alive↔dead cost | 20 | stoppages last several seconds; stops one-frame blips |
| slow / fast player speed | 1.0 / 1.6 m/s | lower quartile and median of average player speed in play (measured on the tracking itself, no labels) |
| set-piece lead | 3.5 s | where the restart sits inside the tagged kick-off windows |
| tag weights | 0.15–0.6 | smaller than tracking: the tags are automatic and coarse (5–35 s windows) |

All of them are in `PARAMS` at the top of `solution/run.py`. The switch cost was lowered once, from
25 to 12, because 25 produced possessions averaging 22 s over the match, longer than real football.

## Results

On the public 5 minutes (a pipeline check only, too small to rank anything):

| | spellF1 | macroF1 |
|---|---|---|
| ours | 0.385 | 0.596 |
| baseline_smooth50 (official floor) | 0.326 | 0.621 |

Full match: 238 possessions, average 17.2 s.

## Limitations and what we would do with more time

- **Dead balls are under-detected** on the public window (we catch about 1 in 6 dead frames there): the ball often stays inside the lines while play is stopped (fouls, offsides, free kicks). Detecting the referee's whistle from the video audio, or the ball standing still with players walking, would help.
- Tracking slots swap identity, but we only need the team, so that does not hurt.
- When the ball is hidden at a turnover, the switch is placed late.
- No learned model: with hand-labelled data from **other** matches, the hand-set weights could be learned instead.

## Tools

An AI coding assistant helped write the code. No match data (tracking, tagging or video) was shared with it — only aggregate statistics such as counts, quantiles and scores.

## Reproduction

```bash
pip install -r requirements.txt
python solution/run.py --team Ragnir
```

The data goes in `data/tactical/`. The output is `evaluation/results/Ragnir.csv`, and the
same command always gives the same file (no randomness).

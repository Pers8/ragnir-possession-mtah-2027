# Technical document

The technical document is **[docs/Ragnir.pdf](docs/Ragnir.pdf)** (LaTeX source: [docs/Ragnir.tex](docs/Ragnir.tex)).

In short: for every frame we score *home has the ball*, *away has the ball* and *dead ball* from
the tracking and the Smart Tagging, then a Viterbi pass picks the best label sequence for each
half with a cost on every change of label. Nothing is trained or tuned on this match, the ground
truth or the video; every constant has a football or physics reason, listed in Table 1 of the PDF.

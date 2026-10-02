"""PitSense-Bench: frozen car-lap decision points with what happened next.

features.py  what was knowable at the decision time (as-of only)
labels.py    what happened afterwards (reads the future; never imported by features)
history.py   cross-race priors with a UTC cutoff
dataset.py   build one parquet file per race
models.py    baselines and first contenders
evaluate.py  expanding-window scoring and the leaderboard
leakcheck.py corrupted-future test
"""

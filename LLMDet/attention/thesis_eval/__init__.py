"""Unified, single-process evaluation stack for the Branch-B temporal models.

Everything the thesis cites for the cue model is produced here so that the
552 / 556 / 563 / 570 configurations are measured by *identical* code. The
package deliberately owns:

- :mod:`attention.thesis_eval.data`         sequence loading + feature slicing
- :mod:`attention.thesis_eval.metrics`      frame metrics + calibration
- :mod:`attention.thesis_eval.bootstrap`    video/track-level cluster bootstrap
- :mod:`attention.thesis_eval.segmentation` event + segmental metrics

Historic evaluation in this project was produced by three different harnesses
(the DDP trainer's per-shard ``validate()``, ``eval_baseline_chain.py`` and
``eval_events_vs_human.py``) that silently disagreed. Absolute numbers from
different harnesses were then placed in the same table. Nothing in this
package computes a publication number more than one way.
"""

EVALUATOR_VERSION = "thesis_eval/1.0.0"

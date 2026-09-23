"""LLMSTU dataset enhancement pipeline.

Detect + crop students from classroom frames, then produce structured
engagement/attention pseudo-labels with Qwen3-VL, and build a golden
evaluation set via a manual labeling tool.
"""

__version__ = "0.1.0"

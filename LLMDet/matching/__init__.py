"""Student-unit caption-to-box matching (thesis Branch A novelty).

Standalone package (numpy/scipy/torch only, no mmdet dependency) implementing
and evaluating strategies for assigning per-student caption units to detected
person boxes in whole-classroom frames:

- ordinal      : Student N -> Nth detection in detector output order (baseline)
- ordinal_lr   : Student N -> Nth detection sorted left-to-right
- hungarian    : global optimal assignment over a compatibility cost matrix
- sinkhorn     : entropic-regularized OT with dustbin padding, discretized

Ground truth comes from the LLMSTU per-student crop dataset, where the
caption<->box correspondence is exact by construction.
"""

from .student_unit_matcher import (
    StudentUnit,
    parse_student_units,
    match_ordinal,
    match_ordinal_lr,
    match_hungarian,
    match_sinkhorn,
)

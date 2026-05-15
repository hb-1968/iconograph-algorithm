# Failure taxonomy — read by `ruthless-grader`

Each entry is a recurring failure pattern the grader should detect and FAIL on.
Append-only — additions come from grader `TAXONOMY DELTA` blocks reviewed by a human
before being committed.

- id: hand-rolled-pixel-loop
  description: Python for-loop iterating over image pixels instead of a library call (H2)
  detection: nested `for` loops over `.shape[0]` / `.shape[1]`; per-pixel indexing in Python
  fix-template: Replace with cv2 / skimage / numpy vectorized equivalent; if none, escalate.

- id: unseeded-randomness
  description: Randomized call without an explicit seed (H8)
  detection: `KMeans(...)` without `random_state=`; `np.random.*` without seed; `random.*` unseeded
  fix-template: Add `random_state=<seed>` (or `np.random.seed(<seed>)` at boundary).

- id: dead-testbed-variant
  description: Losing methodology artifacts left in `testbed/` after acceptance (H7)
  detection: Multiple variant subdirs under `testbed/<stage>/` when the stage is marked accepted
  fix-template: Delete losing variants; testbed strip collapses to single best result.

- id: single-fixture-overfit
  description: Heuristic / threshold tuned to one logo without re-validation across the fixture set (H9)
  detection: Magic constant added in one commit, fixture set not re-rendered in `review.html`
  fix-template: Re-run the stage across all fixtures; revert or generalize if any regress.

- id: duplicate-of-sibling-branch
  description: New function in a sub-case branch duplicates logic already in a sibling or common.py (H5)
  detection: Function with substantially-similar body exists in same-stage sibling file
  fix-template: Lift to common.py; import from both branches.

- id: tuning-before-ground-truth
  description: Algorithm parameters changed before the current output is rendered in `review.html` or a testbed strip (H1)
  detection: Threshold / constant changed without an accompanying `review.html` regeneration in the same change
  fix-template: Render current output first; only then tune.

- id: stage-base-class-or-registry
  description: Plugin registry, stage base class, or DI container introduced (H4)
  detection: Abstract base class for stages; `register(...)` decorators; class-based dispatch
  fix-template: Replace with plain function call + import.

- id: framework-creep-in-bootstrap
  description: Bootstrap or scaffolding code includes runtime framework structure not asked for (H4, H7 spirit)
  detection: New stage adds wiring layers (factories, builders) instead of plain functions
  fix-template: Inline the function; remove the wiring layer.

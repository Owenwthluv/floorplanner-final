# `visualize.py` user guide — render floorplan results for failure analysis

`visualize.py` reads a **saved solutions file** (it does NOT run any optimizer),
re-scores every stored solution with the exact logic of `iccad2026_evaluate.py`,
and renders each case as a two-panel PNG:

```
[ OUR SOLUTION ]  |  [ GROUND TRUTH ]
```

Goal: one look at the image tells you why a case loses points
(boundary, grouping, MIB, ...).

## Requirements

```bash
pip install torch matplotlib shapely
```

- `shapely` is optional (falls back to union-find when absent), but install it so
  grouping components are counted **exactly** like the official evaluator.
- Test data: the `LiteTensorDataTest/` directory at the repo root (already in the repo).

## Step 1 — produce a results file

Run the official evaluator once with `--save-solutions`:

```bash
cd iccad2026contest
python iccad2026_evaluate.py --evaluate my_optimizer --save-solutions
# -> writes my_optimizer_solutions.json
```

Any optimizer module works; the file is named `<submission>_solutions.json`.

## Step 2 — render

Run from the `iccad2026contest/` directory:

```bash
# No flags: render ALL cases from the default file (my_optimizer_solutions.json)
python visualize.py

# Explicit results file
python visualize.py some_other_solutions.json

# Only the 8 worst cases by cost
python visualize.py --worst 8

# Specific cases
python visualize.py --ids 0,1,79 --out-dir viz_out
```

Images are written to `--out-dir` (default `iccad2026contest/viz_out/`) as
`case_XXX.png`. The terminal also prints a table per case:
`id, block count, cost, HPWL_gap, Area_gap, V_rel, bnd/grp/mib violation counts`.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `results` (positional) | `my_optimizer_solutions.json` | Solutions JSON produced by the evaluator's `--save-solutions` |
| (no flags) | — | Render **every** case stored in the results file |
| `--ids` | (none) | Comma-separated case ids to render |
| `--worst` | (none) | Render only the WORST N cases by cost |
| `--out-dir` | `viz_out/` | Output directory for the PNGs |
| `--data-path` | repo root | Directory containing `LiteTensorDataTest/` |

## How to read the images

| Marking | Meaning |
|---|---|
| Blocks sharing a **colour** | Same grouping cluster |
| Light-grey block, `///` hatch | Fixed block (fixed dimensions) |
| Wheat block, `xxx` hatch | Preplaced block (fixed position + dimensions) |
| Light-blue block | Regular soft block |
| Bold **GREEN** outline | Boundary block touching its required edge — OK |
| Bold **RED** outline | Boundary block VIOLATING its required edge |
| `L/R/T/B` label on a block | The bbox edges that block must touch (Left/Right/Top/Bottom) |
| **Red dotted rectangle** | A grouping group split into >1 disconnected piece → grouping violation |
| Grey dashed rectangle | Bounding box of the whole floorplan |
| Small number inside a block | Block index |

Each image title shows `cost`, `HPWL_gap`, `Area_gap`, `V_rel`, the
`bnd/grp/mib` violation counts and `FEASIBLE/INFEASIBLE` — these match the
official score because the constraint logic is copied verbatim from
`evaluate_solution`.

### Note on the GROUND TRUTH panel

The ground truth comes from the dataset labels (`sample["label"]` polygons,
converted to bounding rectangles by `ContestEvaluator._extract_baseline`). The
same constraint check is applied to it, and **the dataset labels themselves are
not perfectly consistent with the constraint tensors**: across the 100 Lite
test cases the labels contain 219 boundary violations (typically a block
sitting exactly 1 unit away from its required edge) and 10 grouping splits,
affecting 90/100 cases. Red marks on the GROUND TRUTH panel are therefore
expected and are NOT a rendering bug. The evaluator never constraint-checks the
ground truth — it only uses it for the baseline HPWL/area — so this has no
effect on scoring.

## Suggested debug workflow

1. Run the evaluator with `--save-solutions` to snapshot the current optimizer.
2. `python visualize.py --worst 10` to find the cases contributing the most cost.
3. Look for red outlines (boundary misses) and red dotted hulls (split groups).
4. Compare the left panel with the right one (ground truth) to see how the
   reference solution arranges the same blocks.
5. Fix the optimizer, regenerate the solutions file, re-render the same cases
   with `--ids ...` to compare before/after.

# NCCL allreduce hangs — 2026-05-04 14:11–16:21

Cluster-side incident on Daint. **9 train jobs and 1 test job died** in a ~2h
window with the same signature; not a model bug.

## Signature

```
[rank N] Watchdog caught collective operation timeout: WorkNCCL(
    SeqNum=…, OpType=ALLREDUCE, NumelIn={1, 269761, 346001},
    Timeout(ms)=600000) ran for 600.0s before timing out.
```

A single rank stalls on a tiny allreduce (often 1 element); the others wait
600 s for the default NCCL watchdog, then SIGABRT. No OOM, no NaN/Inf in
the last logged epoch — last successful epoch's loss/grad were healthy.

## Affected cells

| Cell | JobID | Stall rank | Node | Last epoch before hang |
|---|---|---|---|---|
| zinnia_v5 cahn_hilliard 1e-4 | 3330941 | 3 | (n/a) | mid-epoch |
| zinnia_v5 mickelin 1e-4 | 3330947 | 1 | (n/a) | epoch 25 valid OK |
| zinnia_v5 galewsky 1e-4 | 3330935 | 2 | nid006324 | mid-epoch |
| flower cahn_hilliard 5e-4 | 3330955 | 3 | (n/a) | SeqNum=69 (~30 s in) |
| sfno shock_caps 5e-4 | 3331657 | 2 | (n/a) | mid-epoch |
| fno shock_caps 1e-3 | 3331653 | 2 | (n/a) | mid-epoch |
| flower shock_caps 1e-4 | 3331673 | 2 | (n/a) | SeqNum=6984 |
| flower shock_caps 5e-4 | 3331675 | 2 | (n/a) | **SeqNum=6984** (same!) |
| flower shock_caps 1e-3 | 3331677 | 2 | (n/a) | mid-epoch |
| **fno shock_caps 1e-4** | 3331649 | — | (n/a) | **cuFFT INTERNAL_ERROR** at job start (different mode, also transient) |
| **zinnia_v5 galewsky 1e-4 (resubmit)** | 3332478 | — | nid005854 | **cuFFT INTERNAL_ERROR** ~1 min in (2nd cuFFT failure today) |

## Patterns

- **5 of 9 stalls on rank 2.** Mostly shock_caps cells launched 14:44–14:48
  → likely same flaky node/rack across that wave.
- **flower-shock_caps 1e-4 and 5e-4 hang at the *exact* same SeqNum=6984.**
  Strongly correlated (same step, two jobs) — points at a network-side
  event, not per-job state divergence.
- 1e-4 cells are over-represented (5/9), but only because they take longer
  per-epoch wall clock at lower LR and so were still running when the
  hiccup hit. 5e-4 / 1e-3 siblings on different nodes finished cleanly in
  the same window.

## Side-issue: galewsky FNO 1e-4 test wall-time

`fots-test-fno-galewsky-1e-4` (3330990) hit the 2h walltime today; sibling
tests finish in ~3 min. Resubmitted as 3332400 — if it times out again,
investigate (likely an autoregressive inference path stuck on something).

## Action so far

- 9 orphaned `afterok` test jobs cancelled (Slurm marked them
  DependencyNeverSatisfied anyway).
- Resubmitted: zinnia_v5 ×{CH, mickelin, galewsky} 1e-4, fno-galewsky-1e-4
  test only.
- **Pending decision:** 7 train+test resubmissions (CH-flower-5e-4 + 6
  shock_caps cells). Possibly worth `--exclude`ing the 14:44 rank-2 nodes
  if a Slingshot incident is still unresolved.

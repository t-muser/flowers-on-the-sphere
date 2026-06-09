# Capstor outage recovery — checklist

Drafted 2026-05-03 during a `/capstor` outage on Daint that killed in-flight
training jobs and prevented new pyxis containers from starting. Use this
when CSCS reports `/capstor` is back.

## 1. Sanity-check storage before resubmitting anything

Don't trust "back" until all three pass:

```bash
stat /capstor/scratch/cscs/$USER                                # no Permission denied
ls /capstor/store/cscs/swissai/a01/container_images2/enroot/aurora_arm64_modulus2412.sqsh
touch /capstor/scratch/cscs/$USER/.healthcheck && rm $_         # write works
```

## 2. Audit which checkpoints survived

For each finished-train cell, confirm
`/capstor/scratch/cscs/$USER/fots-runs/<run-name>/0/` has its latest `.pt`
intact (size > 0, opens with `torch.load`). Zero-byte or corrupt
checkpoints mean a full retrain, not a resubmit-with-`auto_resume`.

## 3. Identify mid-training casualties of the outage

```bash
sacct -u $USER --starttime 2026-05-02T21:00 -X --state FAILED \
    -o JobID,JobName%40,Start,End,State
```

Cross-reference against the "still running" set from the original audit
(2026-05-03) — those cells were not part of `scripts/resubmit_crashed.sh`
and need a fresh submission. From what was seen dying during the outage,
likely candidates:

- dahlia × galewsky × {1e-4, 5e-4, 1e-3}
- local_s2_transformer × planetswe × {1e-4, 5e-4, 1e-3}
- local_r_transformer × planetswe × {1e-4, 5e-4}

## 4. Resubmit the 29 crashed-from-before cells

```bash
bash scripts/resubmit_crashed.sh
```

`auto_resume=true` in the train command means surviving checkpoints get
picked up; missing ones start fresh. (Note: this script excludes planetswe
since its `train.sbatch` targets a non-Daint partition.)

## 5. Resubmit the new outage casualties

Extend `scripts/resubmit_crashed.sh` with the cells identified in step 3,
or run the `submit_cell` function inline for one-offs.

## 6. Check the 53 standalone test jobs (3318548–3318600)

```bash
squeue -u $USER -j 3318548-3318600
```

- If still `PENDING` → they'll start automatically, no action needed.
- If any got scheduled during the outage and `FAILED` →
  `bash scripts/resubmit_missing_tests.sh` (idempotent; rerunning is
  harmless since test jobs just write fresh test-set metrics).

## 7. Drain orphan tests

The `DependencyNeverSatisfied` test jobs from the original sweep should
auto-cancel once SLURM notices their `afterok` parent permanently failed.
Spot-check:

```bash
squeue -u $USER -t PENDING | grep -c DependencyNeverSatisfied
```

Should trend to 0 over an hour or two. If it doesn't, scancel by explicit
ID list.

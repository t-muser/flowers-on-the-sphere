#!/bin/bash
#SBATCH --job-name=fots-probe-vit-daint
#SBATCH --account=uba03
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#
# Peak GPU memory sweep for the ViT (ClimaX) baseline on 1xGH200 (96 GiB),
# AR(x2) training step, full-precision (trainer does not use AMP).
# Sweeps both target shapes in one job:
#   - mickelin   128x256 (in=4,  out=1)
#   - shock_caps 256x512 (in=12, out=3)
# Inputs are synthetic, matching the in/out channel counts the trainer
# builds via dim_in * n_steps_input and dim_out * model_predict_steps
# (see fots.train.build_model).
#
# Daint-native (uses the iopsstor modulus container). Pick the largest
# bs that fits with ~20% headroom for DDP/wandb/loader overhead and add
# to FOTS_BATCH_SIZE in scripts/batch_sizes.sh under the right resolution
# key (vit:128x256, vit:256x512).

set -uo pipefail
ulimit -c 0
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm/logs

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR}}"
TOMLPATH="${TOMLPATH:-${WORKDIR}/scripts/torchcontainer_daint_iopsstor.toml}"

srun --ntasks=1 --ntasks-per-node=1 --gpus=1 \
     --export=ALL --environment="${TOMLPATH}" \
     -u -l bash -c "
set -uo pipefail
cd '${WORKDIR}'
export PYTHONPATH=\"${WORKDIR}:\${PYTHONPATH:-}\"

python3 -c 'import timm' 2>/dev/null || pip install --user --no-deps timm 2>&1 | tail -3
python3 -c 'import torch; print(\"torch:\", torch.__version__, \"cuda:\", torch.cuda.is_available())'
nvidia-smi -L

python3 <<'PYEOF'
import gc
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = 'cuda'
AR_STEPS = 2

# (label, H, W, inp_chans, out_chans, [batch sizes to try])
SHAPES = [
    ('mickelin 128x256',  128, 256,  4, 1, [16, 24, 32, 48, 64, 96]),
    ('shock_caps 256x512', 256, 512, 12, 3, [2, 4, 6, 8, 12, 16, 24]),
]

model_cfg = OmegaConf.load('configs/models/vit.yaml').model

def probe(H, W, ic, oc, bs):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = instantiate(model_cfg, inp_shape=(H, W), inp_chans=ic, out_chans=oc).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(bs, ic, H, W, device=DEVICE)
    y = torch.randn(bs, oc, H, W, device=DEVICE)
    try:
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            cur = x
            total = 0.0
            for _step in range(AR_STEPS):
                out = model(cur)
                total = total + torch.nn.functional.mse_loss(out, y)
                if ic > oc:
                    cur = torch.cat([cur[:, oc:, ...], out], dim=1)
                else:
                    cur = out
            total.backward()
            opt.step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_reserved() / (1024 ** 3)
        return n_params, peak, None
    except RuntimeError as e:
        msg = str(e).split('\n')[0]
        return n_params, None, msg
    finally:
        del model, opt, x, y
        try:
            del out, cur, total
        except NameError:
            pass
        gc.collect()
        torch.cuda.empty_cache()

for label, H, W, ic, oc, bss in SHAPES:
    print(f'=== {label}  in={ic} out={oc}  patch={model_cfg.patch_size} ===', flush=True)
    for bs in bss:
        try:
            n_params, peak, err = probe(H, W, ic, oc, bs)
        except RuntimeError as e:
            print(f'  bs={bs:>3d}  OOM at setup ({str(e).split(chr(10))[0]})', flush=True)
            continue
        if err is None:
            print(f'  bs={bs:>3d}  params={n_params:.2f}M  peak_reserved={peak:.2f} GiB', flush=True)
        else:
            print(f'  bs={bs:>3d}  OOM ({err})', flush=True)
PYEOF
"

echo "Probe finished at: $(date)"

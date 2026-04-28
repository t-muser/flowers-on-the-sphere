# flowers-on-the-sphere

## Environment setup

The repo uses `uv`. From a login node (compute nodes have no PyPI access):

```bash
uv sync
```

### Optional: enable the Local-S2 / Local-R transformer baselines

These two baselines need extra setup because the PyPI wheels don't ship
working CUDA kernels for our torch build (`2.11.0+cu130`):

- `torch_harmonics`'s manylinux wheel is CPU-only; DISCO ops have no CUDA
  backend at runtime.
- `natten` ships without `libnatten`, so its only available path is the
  flex backend, which needs `torch.compile` + bf16 + power-of-two head_dim.

If you don't need these two models, you can skip this section — the rest
of the project (Zinnia, Dahlia, Flower, FNO, SFNO, U-Net, SegFormer)
works off `uv sync` alone.

**1. Add build dependencies** (login node, PyPI accessible):

```bash
uv pip install setuptools wheel setuptools-scm
```

**2. Clone torch-harmonics source** (login node):

```bash
mkdir -p third_party
git clone --depth 1 --branch v0.9.0 \
    https://github.com/NVIDIA/torch-harmonics.git third_party/torch-harmonics
```

`third_party/` is gitignored.

**3. Rebuild torch-harmonics with CUDA** (GPU node — needs `nvcc`):

```bash
sbatch scripts/build_torch_harmonics_cuda.sbatch
```

The sbatch loads `CUDA/13.0.0` (matches torch's `cu130` runtime), runs
`FORCE_CUDA_EXTENSION=1 uv pip install --no-build-isolation`, and prints
the CUDA symbols in the resulting `_C.so` for verification.

**4. Smoke-test both transformers** (GPU node):

```bash
sbatch scripts/smoke_transformers.sbatch
```

Expects `[probe] FINAL Local{S2,R}Transformer ...` lines in the `.out`
log; one forward + backward + optimizer step at PlanetSWE shape.

If the torch version ever changes, step 3 must be repeated against the
new torch's `cu*` runtime — the rebuilt extension is ABI-pinned.

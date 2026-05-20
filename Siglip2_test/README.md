# SigLIP 2 — local demo setup

Local reproduction of the official
[SigLIP 2 Colab](https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/SigLIP2_demo.ipynb).

## Hardware

Tested on:

- NVIDIA GeForce RTX 3070 Ti (8 GB VRAM)
- NVIDIA driver 570.x, CUDA runtime 12.x

The 8 GB VRAM budget limits you to the smaller checkpoints. Safe choices:

- `B/16 @ 224` or `B/16 @ 256`
- `B/32 @ 256`

Anything larger (`L/16`, `So400m/16`, `g-opt/16`) will likely OOM on 8 GB.
Use the CPU build if you want to try them: `pip install -U jax` (drop the
`[cuda12]` extra).

## 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate siglip2
```

This installs Python 3.11, JupyterLab, and all pip deps (JAX[cuda12],
flax 0.8.5, big_vision's requirements, gsutil, …).

## 2. Get the `big_vision` source

The demo imports `big_vision.models.proj.image_text.two_towers`, so the
repo needs to be on `PYTHONPATH`.

```bash
git clone --depth=1 https://github.com/google-research/big_vision.git
```

You can either:

- run notebooks from this directory (the included `set_up.ipynb` adds the
  cloned `big_vision/` folder to `sys.path` for you), **or**
- `pip install -e ./big_vision` if you prefer.

## 3. Verify the install

```bash
conda activate siglip2
python -c "import jax; print(jax.devices())"
```

You should see something like `[CudaDevice(id=0)]`. If you see
`[CpuDevice(id=0)]` JAX did not find the GPU — check that
`nvidia-smi` works in the same shell.

## 4. Run the demo

Open `set_up.ipynb` in JupyterLab:

```bash
jupyter lab
```

and pick the **siglip2** kernel. The first cell verifies the environment;
subsequent cells follow the official Colab.

## Notes

- Checkpoints live at `gs://big_vision/siglip2/`. The `gsutil` CLI from
  `requirements.txt` fetches them (`gsutil cp gs://big_vision/siglip2/<file> /tmp/`).
  The first time you run it, `gcloud auth` is **not** required — these
  files are public.
- `flax==0.8.5` is pinned because newer Flax APIs break `big_vision`'s
  model loading (this is the same pin the Colab applies).
- If pip's resolver complains about `jaxlib`, install JAX alone first:
  `pip install -U "jax[cuda12]==0.4.33"`.

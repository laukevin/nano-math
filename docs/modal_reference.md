# Modal Reference for math-nano GPU Training

Quick reference for running nanochat training (pretrain, SFT, eval) on Modal GPUs.
Extracted from Modal docs on 2026-03-18.

---

## 1. Core Concepts

Modal is a serverless GPU platform. You define Python functions, Modal runs them in
containers in the cloud. No Docker, no Kubernetes, no YAML.

```python
import modal

app = modal.App("math-nano")

@app.function(gpu="T4", timeout=3600)
def train():
    import subprocess
    subprocess.run(["python", "-m", "scripts.base_train", ...])
```

### Running

```bash
modal run train.py           # Run the @modal.local_entrypoint function
modal run train.py::train    # Run a specific function
modal deploy train.py        # Deploy as a persistent service
```

### Launching long jobs

Use `.spawn()` instead of `.remote()` for jobs that may exceed the client timeout:

```python
@app.local_entrypoint()
def main():
    train.spawn()  # fire-and-forget, view in Modal dashboard
```

Run with `--detach` to disconnect your terminal:
```bash
modal run --detach train.py
```

---

## 2. Images (Container Environment)

Every function runs in a container built from a `modal.Image`. Layers are cached;
put frequently-changing layers last.

### Installing Python packages

```python
# Preferred: uv is faster
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "torch>=2.4.0",
    "transformers>=4.50.0",
)

# Fallback: standard pip
image = modal.Image.debian_slim().pip_install("torch>=2.4.0")
```

Best practice: pin versions tightly (`torch==2.5.1`) for reproducibility.

### Adding local code

Three methods, with different tradeoffs:

#### `add_local_dir` -- copy a directory tree

```python
image = modal.Image.debian_slim().add_local_dir(
    "./vendor/nanochat",
    remote_path="/root/nanochat",
    ignore=["**/__pycache__/**", "**/.git/**"],
)
```

- By default, files are synced at container startup (not baked into image layer).
  This means fast iteration -- code changes don't rebuild the image.
- Use `copy=True` to bake into the image layer (required if subsequent build steps
  need the files, e.g. `run_commands("cd /root/nanochat && pip install -e .")`).
- The `ignore` parameter accepts gitignore-style patterns.

#### `add_local_python_source` -- importable Python packages

```python
image = modal.Image.debian_slim().add_local_python_source("my_package")
```

Uses Python's import system to locate the package. Good for library code, but
not for arbitrary directory trees.

#### `add_local_file` -- single file

```python
image = modal.Image.debian_slim().add_local_file(
    "./config.yaml", "/app/config.yaml"
)
```

### System packages

```python
image = modal.Image.debian_slim().apt_install("git", "curl")
```

### Environment variables

```python
image = modal.Image.debian_slim().env({
    "PYTHONUNBUFFERED": "1",
    "WANDB_MODE": "disabled",
})
```

### Running build-time commands

```python
image = modal.Image.debian_slim().run_commands(
    "pip install flash-attn --no-build-isolation",
    gpu="A10G",  # can attach GPU for build steps that need CUDA
)
```

### Downloading model weights at build time

```python
def download_model():
    from huggingface_hub import hf_hub_download
    hf_hub_download("meta-llama/Llama-2-7b")

image = modal.Image.debian_slim().run_function(
    download_model,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="A10G",
)
```

### Handling remote-only imports

When image has packages not installed locally, import inside function body:

```python
@app.function(image=image)
def process():
    import torch  # only available in the container
    ...
```

Or use the `imports()` context manager for module-level imports:

```python
with image.imports():
    import torch
```

### Image caching pitfalls

- Each method call is a layer. Changing an early layer invalidates all subsequent layers.
- Put `pip_install` early, `add_local_dir` late (code changes often, deps don't).
- Force rebuild: `MODAL_FORCE_BUILD=1 modal run ...` or add `force_build=True` to a method.
- `MODAL_IGNORE_CACHE=1` rebuilds without permanently busting the cache.

### Recommended image layer order

```python
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")                          # 1. system deps (rare changes)
    .pip_install("torch==2.5.1", ...)            # 2. python deps (occasional)
    .env({"PYTHONUNBUFFERED": "1"})               # 3. env vars
    .add_local_dir("./src", remote_path="/app")   # 4. local code (frequent changes)
)
```

---

## 3. Volumes (Persistent Storage)

Volumes are distributed filesystems accessible from any Modal function.
Optimized for write-once, read-many workloads (model weights, datasets).

### Creating volumes

```python
vol = modal.Volume.from_name("my-volume", create_if_missing=True)
```

Or via CLI:
```bash
modal volume create my-volume
```

### Mounting in functions

```python
@app.function(volumes={"/data": vol_data, "/checkpoints": vol_ckpts})
def train():
    # Read/write to /data and /checkpoints like a local filesystem
    with open("/checkpoints/model.pt", "wb") as f:
        torch.save(model.state_dict(), f)
```

### Commit and reload semantics

**This is the most common pitfall.** Writes are NOT automatically persisted.

```python
@app.function(volumes={"/data": vol})
def write_data():
    with open("/data/file.txt", "w") as f:
        f.write("hello")
    vol.commit()  # REQUIRED to persist changes
```

- Background commits happen every few seconds and on container shutdown,
  but explicit `vol.commit()` is safer for critical data.
- To see changes from other containers: `vol.reload()`.
- During reload, the volume appears empty momentarily.
- Close all file handles before calling `reload()`.

### CLI access (upload/download data)

```bash
modal volume put my-volume ./local/file.txt /remote/path/file.txt
modal volume get my-volume /remote/path/file.txt ./local/file.txt
modal volume ls my-volume /
```

### Python SDK access (outside Modal functions)

```python
vol = modal.Volume.from_name("my-volume")
vol.write_file("path/file.txt", b"content")
data = vol.read_file("path/file.txt")
```

### Volume limitations

- **v1 (default):** max 500K files (recommended <50K), max 5 concurrent writers.
- **v2 (beta):** unlimited files, hundreds of concurrent writers, better random access.
  Create with: `modal volume create --version=2 my-volume`
- Bandwidth: up to 2.5 GB/s (actual varies).
- Concurrent writes to same file: last-write-wins (avoid this).
- No distributed file locking.
- Deleting a volume breaks deployed apps referencing it, even if recreated with same name.

### Volume best practices for training

- Use separate volumes for data, checkpoints, and results (we do this).
- Commit after saving checkpoints: `vol_checkpoints.commit()`.
- Store training data on a volume to avoid re-downloading on each run.
- For checkpoint resumption, check if checkpoint exists on volume before training:
  ```python
  ckpt_path = "/checkpoints/last.ckpt"
  resume = ckpt_path if os.path.exists(ckpt_path) else None
  trainer.fit(model, ckpt_path=resume)
  ```

---

## 4. GPU Configuration

### Available GPUs

| GPU | VRAM | Max count | Notes |
|-----|------|-----------|-------|
| T4 | 16 GB | 8 | Budget option, good for small models |
| L4 | 24 GB | 8 | Good price/performance |
| A10G | 24 GB | 4 | Good for training and inference |
| L40S | 48 GB | 8 | Great for inference, good VRAM |
| A100-40GB | 40 GB | 8 | High-end training |
| A100-80GB | 80 GB | 8 | Large model training |
| H100 | 80 GB | 8 | Fastest training |
| H200 | 141 GB | 8 | Latest gen |
| B200 | 192 GB | 8 | Newest |

### Specifying GPUs

```python
@app.function(gpu="T4")       # single GPU
@app.function(gpu="A100:2")   # 2x A100
@app.function(gpu="H100:8")   # 8x H100
```

### GPU selection guidance

- More than 2 GPUs per container = longer wait times for scheduling.
- Small batch sizes are memory-bound, not compute-bound. Expensive GPUs may
  not help much if you're just loading a model that fits on a cheaper card.
- For our depth-2 model (tiny): T4 is fine.
- For depth-6+ or larger batch sizes: A10G or A100.

### Multi-GPU training

Modal supports multi-GPU on a single node. For PyTorch:
- Use `torchrun` or set `strategy="ddp_spawn"` in PyTorch Lightning.
- Or run training as a subprocess (which is what we do).
- Multi-node training is in private beta.

### CUDA

Modal pre-installs NVIDIA drivers and CUDA driver API. PyTorch bundles its own
CUDA runtime via pip, so `pip_install("torch")` just works.

```python
image = modal.Image.debian_slim().pip_install("torch")  # CUDA included
```

For the full CUDA toolkit (e.g., compiling custom kernels):
```python
image = modal.Image.from_registry("nvidia/cuda:12-devel-ubuntu22.04")
```

Use CUDA 12.x or 13.x. Older versions (11.x) may have driver compatibility issues.

---

## 5. Secrets

Secrets inject environment variables into containers.

### Creating secrets

**CLI:**
```bash
modal secret create wandb-secret WANDB_API_KEY=wk_abc123
modal secret create hf-secret HF_TOKEN=hf_abc123
```

**Dashboard:** https://modal.com/secrets -- has templates for W&B, HuggingFace, etc.

**In code (for dev/testing):**
```python
modal.Secret.from_dict({"WANDB_API_KEY": "wk_abc123"})
modal.Secret.from_dotenv()  # reads .env file (requires python-dotenv)
```

### Using secrets in functions

```python
@app.function(secrets=[modal.Secret.from_name("wandb-secret")])
def train():
    import os
    api_key = os.environ["WANDB_API_KEY"]  # available as env var
```

Multiple secrets:
```python
@app.function(secrets=[
    modal.Secret.from_name("wandb-secret"),
    modal.Secret.from_name("hf-secret"),
])
```

If secrets overlap, later ones win.

### Optional secrets pattern (what we use)

```python
WANDB_SECRET = modal.Secret.from_name("wandb-secret")  # or None

@app.function(secrets=[s for s in [WANDB_SECRET] if s is not None])
def train():
    ...
```

---

## 6. Timeouts and Retries

### Timeouts

- **Default:** 300 seconds (5 minutes) -- way too short for training.
- **Maximum:** 86400 seconds (24 hours).
- Timeout measures execution time only (excludes scheduling/startup).
- Each retry attempt gets a fresh timeout.

```python
@app.function(timeout=2 * 3600)  # 2 hours
def train():
    ...
```

### Startup timeout

Separate from execution timeout. Useful when container initialization is slow
(loading large models, etc.):

```python
@app.function(timeout=3600, startup_timeout=600)
def train():
    ...
```

### Retries for long training

For jobs longer than 24 hours, use retries + checkpointing:

```python
@app.function(
    timeout=86400,  # 24 hours max per attempt
    retries=modal.Retries(max_retries=10, backoff_base=0.0),
    gpu="A100",
    volumes={"/checkpoints": vol},
    single_use_containers=True,  # fresh container per retry
)
def train():
    ckpt = find_latest_checkpoint("/checkpoints/")
    resume_from = ckpt if ckpt else None
    run_training(resume_from=resume_from)
    vol.commit()
```

### FunctionTimeoutError

```python
try:
    result = train.remote()
except modal.exception.FunctionTimeoutError:
    print("Training timed out")
```

---

## 7. Subprocess Pattern (What We Use)

Our training functions run nanochat via `subprocess.run()`. This is the
recommended pattern when your training code is a standalone script.

### Basic pattern

```python
@app.function(image=train_image, gpu="T4", timeout=7200, volumes=VOLUME_MOUNTS)
def run_pretrain(depth: int = 2):
    import subprocess, os, time

    cmd = [
        "python", "-m", "scripts.base_train",
        f"--depth={depth}",
        "--window-pattern=L",
        "--pos-encoding=nope",
    ]

    env = {**os.environ, "WANDB_MODE": "disabled"}
    start = time.monotonic()
    result = subprocess.run(
        cmd,
        cwd="/root/math-nano/vendor/nanochat",
        capture_output=True,
        text=True,
        env=env,
    )
    elapsed = time.monotonic() - start

    # Print output for Modal dashboard logs
    print(result.stdout[-2000:])
    if result.returncode != 0:
        print("STDERR:", result.stderr[-1000:])

    # Persist checkpoints
    vol_checkpoints.commit()

    return {"exit_code": result.returncode, "wall_clock_s": elapsed}
```

### Key considerations

- Use `capture_output=True, text=True` for clean log capture.
- Set `cwd` to the nanochat directory so relative imports work.
- Pass env vars via the `env` parameter (merge with `os.environ`).
- Always `vol.commit()` after writing checkpoints or results.
- Print tail of stdout/stderr so it's visible in Modal dashboard.
- Modal's GPU note: for PyTorch DDP training as subprocess, just use
  `torchrun` or set up DDP manually in the script.

---

## 8. Complete Example Pattern

Putting it all together for our use case:

```python
import modal

app = modal.App("math-nano")

# Volumes
vol_data = modal.Volume.from_name("math-nano-data", create_if_missing=True)
vol_ckpts = modal.Volume.from_name("math-nano-checkpoints", create_if_missing=True)

# Image: deps first (cached), code last (changes often)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.4.0", "transformers>=4.50.0", ...)
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_dir(
        "./vendor/nanochat",
        remote_path="/root/nanochat",
        ignore=["**/__pycache__/**", "**/.git/**"],
    )
)

@app.function(
    image=image,
    gpu="T4",
    timeout=2 * 3600,
    volumes={"/data": vol_data, "/checkpoints": vol_ckpts},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def pretrain(depth: int = 2):
    import subprocess, os

    cmd = ["python", "-m", "scripts.base_train",
           f"--depth={depth}", "--pos-encoding=nope",
           "--window-pattern=L", "--save-every=100",
           "--core-metric-every=-1"]

    result = subprocess.run(
        cmd, cwd="/root/nanochat",
        env={**os.environ, "NANOCHAT_BASE_DIR": "/data"},
        capture_output=True, text=True,
    )
    print(result.stdout[-2000:])
    vol_ckpts.commit()
    return {"exit_code": result.returncode}

@app.local_entrypoint()
def main():
    pretrain.spawn()  # non-blocking, view in dashboard
```

```bash
modal run --detach train.py  # launch and disconnect
```

---

## 9. Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Function times out at 5 min | Set `timeout=7200` (or whatever you need, max 86400) |
| Code changes rebuild entire image | Put `add_local_dir` last in image chain |
| Checkpoints lost after run | Call `vol.commit()` after saving |
| Volume appears empty | Close file handles, then `vol.reload()` |
| `add_local_dir` with `copy=True` rebuilds image on every code change | Use default (no `copy=True`) unless build steps need the files |
| Can't import local module in container | Use `add_local_python_source("module")` or `add_local_dir` with correct `remote_path` |
| Environment variables not set | Use `image.env({...})` for build-time, or pass via `subprocess.run(env={...})` for runtime |
| CUDA not found | PyTorch pip package includes CUDA runtime. Just `pip_install("torch")`. |
| Slow image builds | Pin versions, order layers by change frequency, use `uv_pip_install` instead of `pip_install` |
| Function killed with no error | Likely OOM. Check GPU VRAM usage, try larger GPU or smaller batch size. |
| Multiple volumes with same data | Use separate volumes for data/checkpoints/results (isolation + parallelism) |
| `modal.Secret.from_name()` fails | Secret must exist in Modal dashboard/CLI first. Use the optional pattern. |

---

## 10. CLI Quick Reference

```bash
# Running
modal run file.py                    # run local_entrypoint
modal run --detach file.py           # run detached (keeps running after terminal closes)
modal run file.py::function_name     # run specific function

# Volumes
modal volume create my-vol           # create volume
modal volume ls my-vol /             # list files
modal volume put my-vol local remote # upload
modal volume get my-vol remote local # download
modal volume rm my-vol               # delete volume

# Secrets
modal secret create name KEY=VALUE   # create secret
modal secret list                    # list secrets

# Monitoring
modal app list                       # list running apps
modal app logs app-name              # stream logs

# Images
MODAL_FORCE_BUILD=1 modal run ...    # force image rebuild
MODAL_IGNORE_CACHE=1 modal run ...   # rebuild without busting cache
```

---

## Sources

- https://modal.com/docs/guide (overview)
- https://modal.com/docs/guide/images (image building)
- https://modal.com/docs/guide/volumes (persistent storage)
- https://modal.com/docs/guide/gpu (GPU config)
- https://modal.com/docs/guide/secrets (secrets management)
- https://modal.com/docs/guide/cuda (CUDA setup)
- https://modal.com/docs/guide/timeouts (timeout config)
- https://modal.com/docs/examples/long-training (resumable training pattern)
- https://modal.com/docs/examples/hp_sweep_gpt (SLM training example)

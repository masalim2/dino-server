# DINOv3 Segmentation Inference Server

FastAPI server keeping fine-tuned DINOv3 segmentation models hot in
GPU memory for on-demand inference. Requests and responses carry **local
filesystem paths** — data is staged in and out through the caller's FIRST staging
area.

- **One request = one input directory.** The server loops over every image in
  `input_dir` and writes one mask per image into
  `<input_dir>.dinov3.output/semantic_masks/`.
- **Parallelism is across directories.** Each server owns one GPU; FIRST runs one
  replica per GPU and routes each directory to a replica.
- **Multiple checkpoints** can be served from one process. They are configured
  **only at launch** and all preloaded; a request can select one by name but can
  never supply a checkpoint path.

## Install

```bash
git clone https://github.com/masalim2/dino-server && cd dino-server
uv venv && source .venv/bin/activate
uv pip install --torch-backend=auto -e .
```

This installs the `dino_server` package and a `dino-server` console entrypoint.

## Run

The server listens on a Unix domain socket. On FIRST this is a pilot replica; see
the `sophia/dinov3` LaunchTemplate in gateway-api-fixtures.

```bash
CUDA_VISIBLE_DEVICES=0 dino-server --uds /tmp/dino.sock --default-checkpoint /path/to/best.ckpt
```

All checkpoints are loaded before the socket is bound, so the first request is
already warm, and nothing is loaded after that.

| Flag | Description |
| ---- | ----------- |
| `--default-checkpoint PATH` | Used when a request omits `checkpoint` (also selectable as `"default"`). |
| `--checkpoint NAME=PATH` | Additional checkpoint a request may select by `NAME`. Repeatable. |
| `--staging-root DIR` | Optional, recommended. The canonical `.../user-staging` directory; `input_dir` must then be under `DIR/<user id>/`. Without it, any path containing `/user-staging/<user id>/` is accepted. |

At least one checkpoint is required. Checkpoints are pickled code, so the server
refuses to start if any checkpoint path (before or after resolving symlinks) is
inside a `user-staging` directory or `--staging-root`.

Check a server's status (device, loaded checkpoints, projects):

```bash
curl -s --unix-socket /tmp/dino.sock http://localhost/health | jq
```

## Request example

`POST /v1/segmentation` with a JSON body of filesystem paths. The call blocks until
the whole directory is processed, then returns a summary.

The FIRST gateway authenticates the caller and sets the `X-First-User-Id` header.
`input_dir` must be a subdirectory of `/user-staging/<that id>/`. Output paths are
derived from it and can't be chosen by the caller.

```python
import httpx

client = httpx.Client(transport=httpx.HTTPTransport(uds="/tmp/dino.sock"))
resp = client.post(
    "http://localhost/v1/segmentation",
    headers={"X-First-User-Id": USER_ID},
    json={
        "input_dir":  f"/eagle/IRIBeta/inference_data_staging/user-staging/{USER_ID}/job123",
        "project":    "moon",                  # class colors/names: "moon" | "synaps-I"
        "save_overlay": True,                  # also write color overlays
        "batch_size": 4,
    },
    timeout=3600,  # a large directory can take minutes — set a generous timeout
)
resp.raise_for_status()
print(resp.json())
# {'status': 'ok', 'num_images': 128,
#  'mask_dir': '.../job123.dinov3.output/semantic_masks',
#  'overlay_dir': '.../job123.dinov3.output/semantic_masks_overlay',
#  'elapsed_sec': 12.4}
```

### Selecting a checkpoint

Set `checkpoint` in the request to the `NAME` of a checkpoint the server was
launched with (`--checkpoint NAME=PATH`). To serve a new fine-tune, add it to the
launch configuration and restart the replica. Paths are not accepted.

## Request fields

| Field          | Type   | Default      | Description                                            |
| -------------- | ------ | ------------ | ------------------------------------------------------ |
| `input_dir`    | str    | *(required)* | Directory of images (`.png/.jpg/.jpeg/.tif/.tiff`).    |
| `project`      | str    | `"synaps-I"` | Class palette/names: `"moon"` or `"synaps-I"`.         |
| `checkpoint`   | str    | `"default"`  | Name of a launch-configured checkpoint (not a path).   |
| `save_overlay` | bool   | `false`      | Also write overlays to `semantic_masks_overlay/`.      |
| `batch_size`   | int    | `4`          | Images per forward batch (1–64).                       |
| `num_workers`  | int    | `4`          | DataLoader workers (0–16).                             |
| `model`        | str    | —            | Set by the FIRST gateway; accepted and ignored.        |

Unknown fields are rejected.

## Handling of staged data

Everything under a user's staging directory is treated as untrusted:

- `X-First-User-Id` must be a plain identifier (`[A-Za-z0-9][A-Za-z0-9._@-]*`).
- `input_dir` is resolved and confined to `/user-staging/<id>/`. All later I/O
  walks from that directory with `O_NOFOLLOW`, so symlinks swapped in after the
  check are caught.
- Only regular, non-hidden, singly-linked image files directly in `input_dir` are
  read; symlinks and special files are skipped, and a hardlinked image fails the
  request. Images are decoded only as PNG/JPEG/TIFF, and oversized
  (decompression-bomb) images are rejected.
- The output directories must be real directories (not symlinks). Each output
  file is written to a temp file and renamed into place, so a pre-planted
  symlink or hardlink at the destination is replaced, never written through.

## Notes

- Requests to a single server are **serialized** (one GPU, one CUDA context).
- To add a new project's class colors/names, edit `PROJECTS` in
  `dino_server/inference.py`.

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
- **Multiple checkpoints** can be served from one process — a request may point at
  a custom fine-tune, which is loaded lazily and then kept hot.

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

`--default-checkpoint` is preloaded before the socket is bound, so the first
request is already warm. It's used by any request that doesn't specify its own
`checkpoint`.

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

### Pointing at a custom fine-tune

Set `checkpoint` in the request to a checkpoint path. The first request for a new
checkpoint pays a one-time load cost; after that it stays hot alongside any others
in the process.

## Request fields

| Field          | Type   | Default      | Description                                            |
| -------------- | ------ | ------------ | ------------------------------------------------------ |
| `input_dir`    | str    | *(required)* | Directory of images (`.png/.jpg/.jpeg/.tif/.tiff`).    |
| `project`      | str    | `"synaps-I"` | Class palette/names: `"moon"` or `"synaps-I"`.         |
| `checkpoint`   | str    | server default | Path to a fine-tuned checkpoint to use.              |
| `save_overlay` | bool   | `false`      | Also write overlays to `semantic_masks_overlay/`.      |
| `batch_size`   | int    | `4`          | Images per forward batch.                              |
| `num_workers`  | int    | `4`          | DataLoader workers.                                    |

## Notes

- Requests to a single server are **serialized** (one GPU, one CUDA context).
- To add a new project's class colors/names, edit `PROJECTS` in
  `dino_server/inference.py`.

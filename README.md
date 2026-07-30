# DINOv3 Segmentation Inference Server

FastAPI server keeping fine-tuned DINOv3 segmentation models hot in
GPU memory for on-demand inference. Requests and responses carry **local
filesystem paths** — an external coordinator stages data in and out.

- **One request = one input directory.** The server loops over every image in
  `input_dir` and writes one mask per image into `output_dir/semantic_masks/`.
- **Parallelism is across directories.** Run one server per GPU (e.g. 8 copies on
  an 8×A100 node) and let your coordinator dispatch one directory per GPU.
- **Multiple checkpoints** can be served from one process — a request may point at
  a custom fine-tune, which is loaded lazily and then kept hot.
- **No port coordination.** Each server grabs an OS-assigned free port on startup
  and publishes a record under `$DINO_REGISTRY_DIR`; clients discover the live
  servers instead of being told a port. This is what lets Globus Compute drop 8
  independent jobs (one per GPU) onto a node with no orchestration.

## Install

```bash
git clone https://github.com/masalim2/dino-server && cd dino-server
uv venv && source .venv/bin/activate
uv pip install --torch-backend=auto -e .
```

This installs the `dino_server` package and a `dino-server` console entrypoint.

## Run

Start one resident server per GPU. There is **no `--port`** — each server picks a
free port itself and advertises it (see [Port discovery](#port-discovery)):

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 dino-server --default-checkpoint /path/to/best.ckpt

# GPU 1, ... up to the number of GPUs on the node
CUDA_VISIBLE_DEVICES=1 dino-server --default-checkpoint /path/to/best.ckpt
```

`--default-checkpoint` is preloaded before the server accepts traffic, so the
first request is already warm. It's used by any request that doesn't specify its
own `checkpoint`.

Find the running servers and check one's status (device, loaded checkpoints,
projects):

```bash
python -c "from dino_server import registry; print(registry.discover_servers())"
curl -s http://127.0.0.1:$PORT/health | jq   # $PORT from a discovered record
```

The servers on a node coordinate purely through the filesystem — no port is ever
assigned by hand.  On startup a server **binds a socket to port 0**, letting the kernel hand it a free port. The kernel guarantees uniqueness across the concurrently-starting servers.  That same live socket is handed
straight to uvicorn, so there is no window where the reserved port is free for another process to steal.

Once its model is warm, the server writes an endpoint record to
`$DINO_REGISTRY_DIR` (atomically, via temp-file + rename).  The server makes a
best effort to delete its file on teardown.  If it's hard-killed the file
lingers, so discovery treats the record's PID as the source of truth:
dead-process records are ignored and pruned.

Clients discover servers with `dino_server.registry.discover_servers()`, which
returns one dict per live server.

## Globus Compute integration

The intended deployment provisions **one server per GPU via Globus Compute**. PBS
assigns each job a single GPU (visible as GPU 0), so 8 independent jobs land on a
node with no knowledge of each other.

- `user_config_template.yaml.j2` — each job's PBS `worker_init` starts one
  `dino-server` in the background and blocks until that server is healthy, before
  the Globus Compute worker itself comes up.
- `register_globus_compute_func.py` — registers a self-contained `submit(payload)`
  function. On the node it discovers all the live servers (one per GPU job),
  health-checks one, forwards the full payload to it over loopback (with a generous
  timeout), and returns the server's response as a plain dict. It shuffles across
  servers and fails over if one is unavailable.

```bash
python register_globus_compute_func.py   # prints the registered function UUID
```

## Request example (Python)

`POST /infer` with a JSON body of filesystem paths. The call blocks until the
whole directory is processed, then returns a summary.

```python
import requests
from dino_server import registry

server = registry.discover_servers()[0]   # pick any live server on the node

resp = requests.post(
    f"{server['url']}/infer",
    json={
        "input_dir":  "/scratch/job123/in",    # dir of images to segment
        "output_dir": "/scratch/job123/out",   # masks written under here
        "project":    "moon",                  # class colors/names: "moon" | "synaps-I"
        "save_overlay": True,                  # also write color overlays
        "batch_size": 4,
    },
    timeout=3600,  # a large directory can take minutes — set a generous timeout
)
resp.raise_for_status()
print(resp.json())
# {'status': 'ok', 'num_images': 128,
#  'mask_dir': '/scratch/job123/out/semantic_masks',
#  'overlay_dir': '/scratch/job123/out/semantic_masks_overlay',
#  'elapsed_sec': 12.4}
```

### Pointing at a custom fine-tune

Override the checkpoint per request. The first request for a new checkpoint pays
a one-time load cost; after that it stays hot alongside any others in the process.

```python
requests.post(
    f"{server['url']}/infer",
    json={
        "input_dir":  "/scratch/job124/in",
        "output_dir": "/scratch/job124/out",
        "project":    "synaps-I",
        "checkpoint": "/models/my_finetune/checkpoints/best.ckpt",  # custom fine-tune
    },
    timeout=3600,
).raise_for_status()
```

## Request fields

| Field          | Type   | Default      | Description                                            |
| -------------- | ------ | ------------ | ------------------------------------------------------ |
| `input_dir`    | str    | *(required)* | Directory of images (`.png/.jpg/.jpeg/.tif/.tiff`).    |
| `output_dir`   | str    | *(required)* | Masks written to `output_dir/semantic_masks/`.         |
| `project`      | str    | `"synaps-I"` | Class palette/names: `"moon"` or `"synaps-I"`.         |
| `checkpoint`   | str    | server default | Path to a fine-tuned checkpoint to use.              |
| `save_overlay` | bool   | `false`      | Also write overlays to `output_dir/semantic_masks_overlay/`. |
| `batch_size`   | int    | `4`          | Images per forward batch.                              |
| `num_workers`  | int    | `4`          | DataLoader workers.                                    |

## Notes

- Requests to a single server are **serialized** (one GPU, one CUDA context). Send
  concurrent work to different server copies, not the same one.
- Servers advertise on `0.0.0.0` but publish a loopback `url`; on-node clients
  (including the Globus Compute function) reach them via `127.0.0.1`.
- To add a new project's class colors/names, edit `PROJECTS` in
  `dino_server/inference.py`.
```

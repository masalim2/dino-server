from typing import TypedDict
from globus_compute_sdk import Client


class Payload(TypedDict, total=False):
    input_dir: str      # required
    output_dir: str     # required
    project: str
    checkpoint: str
    save_overlay: bool
    batch_size: int
    num_workers: int


def submit(payload: Payload):
    import random
    import requests
    import json.decoder
    from dino_server import registry

    # Discover the DINO servers running locally on this node. Each server grabs a
    # free port and publishes its record under /eagle/inference_service/.dino_server_registry; discovery skips
    # any whose process has died (see dino_server/registry.py).
    servers = registry.discover_servers()
    if not servers:
        raise RuntimeError("No DINO servers are running on this node")

    # Spread load across servers and fail over: shuffle, then try each until one
    # both passes a /health check and accepts the work.
    random.shuffle(servers)
    busy_servers = []
    last_error = {}

    for server in servers:
        base = server["url"]
        try:
            resp = requests.get(f"{base}/health", timeout=5).json()
        except (requests.RequestException, json.decoder.JSONDecodeError) as e:
            last_error[base] = str(e)
            continue

        if (stat := resp.get("status")) != "ok":
            last_error[base] = f"/health response status={stat}"
            continue

        # Defer busy servers; prefer landing on a free one:
        if resp.get("busy"):
            busy_servers.append(server)
            continue

        try:
            resp = requests.post(f"{base}/infer", json=payload, timeout=3600)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_error[base] = str(e)
            if isinstance(e, requests.HTTPError):
                last_error[base] += f" (Response: {e.response.text[:300]})"
            continue


    # No idle servers; try the busy ones:
    for server in busy_servers:
        base = server["url"]
        try:
            resp = requests.post(f"{base}/infer", json=payload, timeout=3600)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_error[base] = str(e)
            if isinstance(e, requests.HTTPError):
                last_error[base] += f" (Response: {e.response.text[:300]})"
            continue


    raise RuntimeError(
        f"All {len(servers)} local DINO servers failed; last errors: {last_error}"
    )


if __name__ == "__main__":
    gcc = Client()
    func_uuid = gcc.register_function(submit)
    print(f"Registered {submit.__name__}: {func_uuid=}")

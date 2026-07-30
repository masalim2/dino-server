"""
Node-local service registry for DINOv3 inference servers.

The deployment model: Globus Compute provisions one server per GPU, so up to 8
independent ``dino-server`` processes land on the same node with no knowledge of
each other. Each must grab its own port and advertise itself; client Globus
Compute functions that arrive later need to discover which servers are live.
"""
import json
import os
import socket
import tempfile
from pathlib import Path
from typing import Optional

# Well-known directory holding one <pid>.json per running server.
REGISTRY_DIR = Path(os.environ.get("DINO_REGISTRY_DIR", "/eagle/inference_service/.dino_server_registry"))


def reserve_free_socket(host: str = "0.0.0.0") -> socket.socket:
    """
    Bind a TCP socket to an OS-assigned free port and return it, still open.

    The returned socket owns the port for as long as it stays open, which closes
    the reservation race: hand this socket to the HTTP server (uvicorn accepts a
    pre-bound socket) instead of closing it and re-binding by number.

    We deliberately do NOT set SO_REUSEADDR/SO_REUSEPORT before binding: reuse
    could let two processes bind the same port, which is exactly what we're
    guarding against here.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, 0))  # port 0 -> kernel picks a free port atomically
    except Exception:
        sock.close()
        raise
    return sock


def port_of(sock: socket.socket) -> int:
    """Return the concrete port the kernel assigned to a bound socket."""
    return sock.getsockname()[1]


def _record_path(pid: Optional[int] = None) -> Path:
    return REGISTRY_DIR / f"{pid or os.getpid()}.json"


def register(port: int, *, host: str = "0.0.0.0", **extra) -> Path:
    """
    Publish this process's server record atomically and return the file path.

    ``extra`` (e.g. ``gpu``, ``default_checkpoint``) is merged into the record so
    clients can pick a server by GPU or checkpoint if they care to.
    """
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "pid": os.getpid(),
        "host": host,
        "hostname": socket.gethostname(),
        "port": port,
        # Clients on the node talk to servers over the loopback interface.
        "url": f"http://127.0.0.1:{port}",
        **extra,
    }

    path = _record_path()
    # Write to a temp file in the same dir, then atomically rename into place so a
    # concurrent reader never observes a partially written record.
    fd, tmp_name = tempfile.mkstemp(dir=str(REGISTRY_DIR), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(record, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


def unregister(path: Optional[Path] = None) -> None:
    """Best-effort removal of this process's record. Safe to call more than once."""
    target = Path(path) if path is not None else _record_path()
    try:
        target.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _pid_alive(pid: int) -> bool:
    """True if a process with ``pid`` exists on this host (POSIX)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists but owned by another user — still counts as alive.
        return True
    return True


def discover_servers(*, prune_stale: bool = True) -> list[dict]:
    """
    Return the records of DINO servers believed to be running on this node.

    Each record is a dict with at least ``pid``, ``host``, ``port`` and ``url``.
    Records whose owning process is no longer alive are skipped, and (when
    ``prune_stale``) their leftover files are removed on a best-effort basis so the
    directory doesn't accumulate corpses from hard-killed jobs.

    Liveness here means "the process exists," which is cheap and sufficient for
    routing. A caller that needs a stronger guarantee can additionally GET
    ``<url>/health`` before dispatching real work.
    """
    if not REGISTRY_DIR.is_dir():
        return []

    servers: list[dict] = []
    for entry in sorted(REGISTRY_DIR.glob("*.json")):
        try:
            record = json.loads(entry.read_text())
        except (OSError, ValueError):
            # Unreadable or mid-write; skip this pass.
            continue

        if record.get("hostname") != socket.gethostname():
            continue

        pid = record.get("pid", 0)
        if not _pid_alive(pid):
            if prune_stale:
                unregister(entry)
            continue

        servers.append(record)

    return servers

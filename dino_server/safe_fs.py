"""
Symlink-safe filesystem access for user-staged (untrusted) directories.

Everything below a user's staging directory is controlled by that user, who can
swap any component for a symlink at any time — including after we've validated
the path. So instead of checking a path and then re-opening it by name, we walk
from a trusted *anchor* directory one component at a time with ``O_NOFOLLOW``
and do all further I/O relative to the resulting directory file descriptors.

- Reads only accept regular, singly-linked files (no symlinks, FIFOs, devices,
  or hardlinks to files elsewhere).
- Writes go to a fresh ``O_EXCL`` temp file which is then renamed into place, so
  a pre-planted symlink or hardlink at the destination is replaced, never
  written through.
"""
import errno
import io
import os
import secrets
import stat
from typing import Sequence

_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | _CLOEXEC
# O_NONBLOCK so opening a planted FIFO can't hang the server; we reject
# non-regular files via fstat right after.
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | _CLOEXEC
_WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | _CLOEXEC


class UnsafePathError(ValueError):
    """A user-staged path is a symlink, special file, or otherwise not allowed."""


def _check_component(name: str) -> None:
    if not name or name in (".", "..") or "/" in name or "\0" in name:
        raise UnsafePathError(f"invalid path component: {name!r}")


def open_dir(anchor: str, parts: Sequence[str]) -> int:
    """Open ``anchor/parts[0]/.../parts[-1]`` without following any symlink below anchor.

    ``anchor`` must be a trusted, already-canonical directory path. Returns a
    directory fd the caller must close.
    """
    fd = os.open(anchor, _DIR_FLAGS)
    try:
        for part in parts:
            _check_component(part)
            try:
                nfd = os.open(part, _DIR_FLAGS, dir_fd=fd)
            except OSError as e:
                if e.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise UnsafePathError(f"{part!r} is not a real directory") from e
                raise
            os.close(fd)
            fd = nfd
    except BaseException:
        os.close(fd)
        raise
    return fd


def ensure_subdir(parent_fd: int, name: str) -> int:
    """Create (if needed) and open a real subdirectory of ``parent_fd``."""
    _check_component(name)
    try:
        os.mkdir(name, dir_fd=parent_fd)
    except FileExistsError:
        pass
    try:
        return os.open(name, _DIR_FLAGS, dir_fd=parent_fd)
    except OSError as e:
        if e.errno in (errno.ELOOP, errno.ENOTDIR):
            raise UnsafePathError(f"{name!r} exists and is not a real directory") from e
        raise


def list_regular_files(dir_fd: int) -> list[str]:
    """Names of regular (non-symlink) files directly inside ``dir_fd``."""
    # scandir dups the fd internally; dir_fd stays open and owned by the caller.
    with os.scandir(dir_fd) as it:
        return [e.name for e in it if e.is_file(follow_symlinks=False)]


def read_file(dir_fd: int, name: str) -> bytes:
    """Read a regular, singly-linked file from ``dir_fd`` without following symlinks."""
    _check_component(name)
    try:
        fd = os.open(name, _READ_FLAGS, dir_fd=dir_fd)
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise UnsafePathError(f"{name!r} is a symlink") from e
        raise
    with os.fdopen(fd, "rb") as f:
        st = os.fstat(f.fileno())
        if not stat.S_ISREG(st.st_mode):
            raise UnsafePathError(f"{name!r} is not a regular file")
        if st.st_nlink != 1:
            raise UnsafePathError(f"{name!r} is hardlinked; refusing to read it")
        os.set_blocking(f.fileno(), True)
        return f.read()


def write_file_atomic(dir_fd: int, name: str, data: bytes) -> None:
    """Write ``data`` to ``name`` in ``dir_fd`` via temp file + rename."""
    _check_component(name)
    tmp = f".{name}.{secrets.token_hex(8)}.tmp"
    fd = os.open(tmp, _WRITE_FLAGS, 0o644, dir_fd=dir_fd)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    except BaseException:
        try:
            os.unlink(tmp, dir_fd=dir_fd)
        except FileNotFoundError:
            pass
        raise


def encode_image(img, name: str) -> bytes:
    """Encode a PIL image in the format implied by ``name``'s extension."""
    from PIL import Image

    ext = os.path.splitext(name)[1].lower()
    fmt = Image.registered_extensions().get(ext)
    if fmt is None:
        raise UnsafePathError(f"unsupported output extension: {ext!r}")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

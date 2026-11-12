"""Upload handling: stream a multipart upload to a temp file with a byte limit, then validate the HDF5."""

import contextlib
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt
from python_multipart.multipart import MultipartParser, parse_options_header
from starlette.requests import Request

from fastmri_knee.gateway.errors import ProblemError

MIN_SIDE, MAX_SIDE = 320, 1024  # k-space must cover the 320x320 crop; the cap bounds memory per slice
_BODY_OVERHEAD = 1 << 20  # multipart framing and small non-file fields allowed on top of the file limit


@contextlib.contextmanager
def temp_upload(directory: str | None) -> Iterator[Path]:
    """A temp file path that is always removed, on success, error, or client disconnect (cancellation)."""
    fd, name = tempfile.mkstemp(suffix=".h5", dir=directory)
    os.close(fd)
    path = Path(name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


async def stream_file_part(request: Request, dest: Path, limit: int) -> None:
    """Write the multipart `file` part to `dest`, counting bytes and aborting as soon as `limit` is exceeded.

    Nothing is held in memory beyond one network chunk. Raises ProblemError 415/413/400.
    """
    content_type, options = parse_options_header(request.headers.get("content-type", ""))
    if content_type != b"multipart/form-data" or b"boundary" not in options:
        raise ProblemError(415, "Content-Type must be multipart/form-data")
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit + _BODY_OVERHEAD:
        raise ProblemError(413, f"Upload exceeds {limit // (1 << 20)} MB")

    state: dict[str, Any] = {"in_file": False, "found": False, "written": 0, "name": b""}
    out = dest.open("wb")

    def on_part_begin() -> None:
        state["name"] = b""
        state["in_file"] = False

    headers: dict[bytes, bytes] = {}
    field_buf = bytearray()

    def on_header_field(data: bytes, start: int, end: int) -> None:
        field_buf.extend(data[start:end])

    def on_header_value(data: bytes, start: int, end: int) -> None:
        headers[bytes(field_buf).lower()] = headers.get(bytes(field_buf).lower(), b"") + data[start:end]

    def on_header_end() -> None:
        field_buf.clear()

    def on_headers_finished() -> None:
        _, disp = parse_options_header(headers.get(b"content-disposition", b""))
        state["in_file"] = disp.get(b"name") == b"file"
        state["found"] = state["found"] or state["in_file"]
        headers.clear()

    def on_part_data(data: bytes, start: int, end: int) -> None:
        if state["in_file"]:
            state["written"] += end - start
            if state["written"] > limit:
                raise ProblemError(413, f"Upload exceeds {limit // (1 << 20)} MB")
            out.write(data[start:end])

    parser = MultipartParser(
        options[b"boundary"],
        callbacks={
            "on_part_begin": on_part_begin,
            "on_header_field": on_header_field,
            "on_header_value": on_header_value,
            "on_header_end": on_header_end,
            "on_headers_finished": on_headers_finished,
            "on_part_data": on_part_data,
        },
    )
    total = 0
    try:
        async for chunk in request.stream():
            total += len(chunk)
            if total > limit + _BODY_OVERHEAD:
                raise ProblemError(413, f"Upload exceeds {limit // (1 << 20)} MB")
            parser.write(chunk)
        parser.finalize()
    except ProblemError:
        raise
    except Exception:
        raise ProblemError(400, "Malformed multipart body") from None
    finally:
        out.close()
    if not state["found"]:
        raise ProblemError(400, "Missing multipart field 'file'")


def read_kspace(path: Path, max_slices: int) -> npt.NDArray[np.complex64]:
    """Validate the HDF5 (dataset, rank, dtype, size, slice count) before any compute, then load `kspace`.

    Blocking: call from a worker thread. Raises ProblemError 400/413.
    """
    try:
        f = h5py.File(path, "r")
    except OSError:
        raise ProblemError(400, "File is not a valid HDF5 file") from None
    with f:
        ds = f.get("kspace")
        if not isinstance(ds, h5py.Dataset):
            raise ProblemError(400, "HDF5 file has no 'kspace' dataset")
        if ds.ndim != 3:
            raise ProblemError(400, f"'kspace' must have rank 3 (slices, H, W), got rank {ds.ndim}")
        if ds.dtype != np.complex64:
            raise ProblemError(400, f"'kspace' must be complex64, got {ds.dtype}")
        slices, h, w = ds.shape
        if slices < 1:
            raise ProblemError(400, "'kspace' has no slices")
        if slices > max_slices:
            raise ProblemError(413, f"At most {max_slices} slices per request, got {slices}")
        if not (MIN_SIDE <= h <= MAX_SIDE and MIN_SIDE <= w <= MAX_SIDE):
            raise ProblemError(400, f"'kspace' H and W must be within [{MIN_SIDE}, {MAX_SIDE}], got {h}x{w}")
        try:
            return np.asarray(ds[...], dtype=np.complex64)
        except Exception:
            raise ProblemError(400, "Could not read 'kspace' (corrupt data)") from None

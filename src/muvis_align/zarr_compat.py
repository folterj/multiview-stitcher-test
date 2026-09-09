"""Windows-only workaround for zarr's atomic metadata write losing a race for its own file.

zarr writes each metadata document by writing a temporary file and then os.replace()-ing it onto
the real name (zarr.storage._local._atomic_write). On Windows that rename fails outright -
PermissionError, WinError 5 - if anything holds the destination open at that instant, where on
POSIX it would simply succeed. Something does, briefly and unpredictably: writing an OME-Zarr
store failed this way in roughly one full test run in four, always inside the fusion export, and
never reproducibly on its own.

The hold is transient. Retrying the rename for a few seconds absorbed it in every case observed:
nine full test runs with this in place passed, four of them absorbing a retry, against four
outright failures in the fifteen runs before it. What holds the handle was never identified -
the obvious candidates were each ruled out by experiment (the reader fusion.fuse() opens on its
own output store does not block the rename, and the failure happens equally to a store no other
code has touched) - so this waits the hold out rather than claiming to prevent it.

Applies only on Windows, and only if zarr's private helper still looks the way this expects; on
any other platform, or if zarr changes, it does nothing and the original is left alone.
"""
import contextlib
import logging
import os
import time
import uuid
from pathlib import Path

_MAX_WAIT_SECONDS = 4.0
_FIRST_DELAY_SECONDS = 0.01

_applied = False


def apply_windows_atomic_write_retry():
    """Install the retry. Idempotent, and a no-op where it does not apply."""
    global _applied
    if _applied or os.name != 'nt':
        return False

    try:
        import zarr.storage._local as zarr_local
    except ImportError:
        return False
    original = getattr(zarr_local, '_atomic_write', None)
    safe_move = getattr(zarr_local, '_safe_move', None)
    if original is None or safe_move is None:
        # zarr's internals have moved - leave them alone rather than guess
        return False

    @contextlib.contextmanager
    def _atomic_write_with_retry(path, mode, exclusive=False):
        path = Path(path)
        tmp_path = path.with_suffix(f'.{uuid.uuid4().hex}.partial')
        # cleanup on failure only, exactly as the original does: on success the temp file has
        # been renamed away, so there is nothing left to remove
        try:
            with tmp_path.open(mode) as file:
                yield file
            delay, waited, attempts = _FIRST_DELAY_SECONDS, 0.0, 0
            while True:
                try:
                    if exclusive:
                        safe_move(tmp_path, path)
                    else:
                        tmp_path.replace(path)
                    break
                except PermissionError:
                    if waited >= _MAX_WAIT_SECONDS:
                        # a hold this long is not the transient one this exists for
                        raise
                    time.sleep(delay)
                    waited += delay
                    attempts += 1
                    delay = min(delay * 2, 0.2)
            if attempts:
                logging.debug(f'{path.name}: atomic write retried {attempts}x'
                              f' ({waited:.2f}s) - destination was briefly held open')
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    zarr_local._atomic_write = _atomic_write_with_retry
    _applied = True
    return True

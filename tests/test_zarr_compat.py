"""The Windows atomic-write retry must absorb a brief hold and nothing more.

zarr renames each metadata document into place; on Windows that rename fails outright if
anything holds the destination open for that instant. Measured: roughly one full test run in
four failed that way inside the fusion export, and nine runs with this retry in place passed,
four of them absorbing a retry. The hold is transient, so waiting it out is the fix - but a
retry that waits forever would turn a real leak into a hang, so the bound matters too.
"""
import os
import threading
import time
import uuid
from pathlib import Path

import pytest

from muvis_align.zarr_compat import apply_windows_atomic_write_retry

pytestmark = pytest.mark.skipif(os.name != 'nt', reason='the rename only fails this way on Windows')


@pytest.fixture(autouse=True)
def patched():
    # constants.py applies this at import; make the dependency explicit for these tests
    apply_windows_atomic_write_retry()
    import zarr.storage._local as zarr_local
    return zarr_local._atomic_write


def write_through(atomic_write, path, payload=b'{"v": 2}'):
    with atomic_write(path, 'wb') as file:
        file.write(payload)


def test_it_is_installed_by_importing_the_package():
    import muvis_align.constants  # noqa: F401
    import zarr.storage._local as zarr_local

    assert zarr_local._atomic_write.__name__ == '_atomic_write_with_retry'


def test_applying_twice_is_a_no_op(patched):
    import zarr.storage._local as zarr_local

    apply_windows_atomic_write_retry()
    assert zarr_local._atomic_write is patched


def test_plain_write_still_works(tmp_path, patched):
    target = tmp_path / 'zarr.json'
    target.write_bytes(b'{"v": 1}')

    write_through(patched, target)

    assert target.read_bytes() == b'{"v": 2}'


def test_creates_a_file_that_does_not_exist_yet(tmp_path, patched):
    target = tmp_path / 'zarr.json'

    write_through(patched, target)

    assert target.read_bytes() == b'{"v": 2}'


def test_a_brief_hold_on_the_destination_is_absorbed(tmp_path, patched):
    """Exactly the observed failure: the destination is open when the rename happens, and let
    go a moment later. Without the retry this raises PermissionError, WinError 5."""
    target = tmp_path / 'zarr.json'
    target.write_bytes(b'{"v": 1}')

    handle = target.open('rb')

    def release_shortly():
        time.sleep(0.3)
        handle.close()

    releaser = threading.Thread(target=release_shortly)
    releaser.start()
    try:
        write_through(patched, target)
    finally:
        releaser.join()
        if not handle.closed:
            handle.close()

    assert target.read_bytes() == b'{"v": 2}'


def test_a_hold_that_never_lets_go_still_raises(tmp_path, patched, monkeypatch):
    """A retry that waits forever would turn a genuinely leaked handle into a hang. The wait is
    bounded, and past it the original error surfaces."""
    import muvis_align.zarr_compat as compat

    monkeypatch.setattr(compat, '_MAX_WAIT_SECONDS', 0.2)
    apply_windows_atomic_write_retry()
    import zarr.storage._local as zarr_local

    target = tmp_path / 'zarr.json'
    target.write_bytes(b'{"v": 1}')
    with target.open('rb'):
        with pytest.raises(PermissionError):
            write_through(zarr_local._atomic_write, target)


def test_no_partial_files_are_left_behind(tmp_path, patched):
    target = tmp_path / 'zarr.json'
    for _ in range(3):
        write_through(patched, target)

    assert [p.name for p in tmp_path.iterdir()] == ['zarr.json']


def test_partial_is_cleaned_up_when_the_body_raises(tmp_path, patched):
    target = tmp_path / 'zarr.json'
    with pytest.raises(ValueError):
        with patched(target, 'wb') as file:
            file.write(b'x')
            raise ValueError('boom')

    assert not list(tmp_path.glob('*.partial'))

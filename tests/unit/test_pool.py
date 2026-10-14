import asyncio

import pytest

from src.application.errors import (
    AudioDecodeError,
    RequestTimeoutError,
    ServiceBusyError,
    ServiceNotReadyError,
    TranscriptionError,
    WorkerCrashedError,
)
from src.infrastructure.pool import ProcessPool
from tests.fakes import fake_factory


def make_pool(workers=1, queue_limit=1, request_timeout=10.0, fail=False):
    return ProcessPool(
        fake_factory,
        {"fail": fail},
        workers=workers,
        queue_limit=queue_limit,
        request_timeout=request_timeout,
        start_timeout=60.0,
        restart_delay=0.2,
    )


async def test_result_comes_back_from_worker_process():
    pool = make_pool(workers=2)
    await pool.start()
    try:
        status = pool.status()
        assert status.ready and status.alive == 2 and status.capacity == 3
        outcome = await pool.transcribe(b"hello")
        assert outcome.transcript.pcm_sha256.endswith(":hello")
        assert outcome.wait_ms >= 0
    finally:
        await pool.stop()
    assert not pool.status().ready


async def test_worker_errors_keep_their_codes():
    pool = make_pool()
    await pool.start()
    try:
        with pytest.raises(AudioDecodeError):
            await pool.transcribe(b"bad")
        with pytest.raises(TranscriptionError) as error:
            await pool.transcribe(b"bug")
        assert error.value.code == "internal" and "ValueError" in str(error.value)
        assert (await pool.transcribe(b"ok")).transcript.pcm_sha256.endswith(":ok")
    finally:
        await pool.stop()


async def test_crashed_worker_is_replaced():
    pool = make_pool()
    await pool.start()
    try:
        first = (await pool.transcribe(b"pid")).transcript.pcm_sha256.split(":")[0]
        with pytest.raises(WorkerCrashedError):
            await pool.transcribe(b"crash")
        assert pool.status().alive == 1
        second = (await pool.transcribe(b"pid")).transcript.pcm_sha256.split(":")[0]
        assert first != second
    finally:
        await pool.stop()


async def test_stuck_worker_is_killed_on_timeout():
    pool = make_pool(request_timeout=1.0)
    await pool.start()
    try:
        with pytest.raises(RequestTimeoutError):
            await pool.transcribe(b"sleep:30")
        assert (await pool.transcribe(b"after")).transcript.pcm_sha256.endswith(":after")
    finally:
        await pool.stop()


async def test_full_queue_answers_busy_at_once():
    pool = make_pool(workers=1, queue_limit=0)
    await pool.start()
    try:
        slow = asyncio.create_task(pool.transcribe(b"sleep:1.5"))
        await asyncio.sleep(0.2)
        assert pool.status().busy == 1
        with pytest.raises(ServiceBusyError):
            await pool.transcribe(b"second")
        await slow
        assert pool.status().busy == 0 and pool.status().waiting == 0
    finally:
        await pool.stop()


async def test_queued_request_waits_for_free_worker():
    pool = make_pool(workers=1, queue_limit=1)
    await pool.start()
    try:
        slow = asyncio.create_task(pool.transcribe(b"sleep:1"))
        await asyncio.sleep(0.2)
        queued = await pool.transcribe(b"next")
        assert queued.wait_ms >= 500
        await slow
    finally:
        await pool.stop()


async def test_cancelled_request_frees_its_place_when_worker_finishes():
    pool = make_pool(workers=1, queue_limit=0)
    await pool.start()
    try:
        task = asyncio.create_task(pool.transcribe(b"sleep:1"))
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(ServiceBusyError):
            await pool.transcribe(b"now")
        await asyncio.sleep(1.5)
        assert (await pool.transcribe(b"later")).transcript.pcm_sha256.endswith(":later")
    finally:
        await pool.stop()


async def test_pool_does_not_start_when_models_fail_to_load():
    pool = make_pool(fail=True)
    with pytest.raises(RuntimeError, match="не загрузил"):
        await pool.start()
    with pytest.raises(ServiceNotReadyError):
        await pool.transcribe(b"x")

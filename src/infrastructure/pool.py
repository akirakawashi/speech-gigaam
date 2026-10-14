"""Пул процессов-обработчиков.

Каждый обработчик — отдельный процесс: модели загружаются в нём один раз, файлы идут
по одному. Поиск речи и декодер — циклы на Python, поэтому параллельность даёт число
процессов, а не потоков. Запросы сверх ``workers + queue_limit`` сразу получают
«занято». Упавший или зависший обработчик заменяется новым; запрос, на котором это
случилось, получает ошибку и должен быть повторён вызывающим.
"""

import asyncio
import functools
import itertools
import logging
import multiprocessing
import os
import signal
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any, TypeVar

from src.application.errors import (
    RequestTimeoutError,
    ServiceBusyError,
    ServiceNotReadyError,
    TranscriptionError,
    WorkerCrashedError,
    error_from_code,
)
from src.application.ports import Outcome, ServiceStatus
from src.domain.transcript import Transcript

log = logging.getLogger("speech_gigaam.pool")

T = TypeVar("T")
Engine = Callable[[bytes], Transcript]
EngineFactory = Callable[[Any], Engine]


def _worker_main(factory: EngineFactory, config: Any, conn: Connection) -> None:
    """Точка входа процесса-обработчика.

    Строит движок фабрикой и сообщает родителю ``("ready", pid)`` или
    ``("failed", причина)``. Затем принимает ``(номер, байты)`` и отвечает
    ``(номер, "ok", Transcript)`` или ``(номер, "error", (код, сообщение))``.
    ``None`` или закрытый канал завершают процесс.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    logging.basicConfig(
        level=os.environ.get("GIGAAM_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s[%(process)d] %(message)s",
    )
    try:
        engine = factory(config)
    except BaseException as exc:
        try:
            conn.send(("failed", f"{type(exc).__name__}: {exc}"))
        finally:
            conn.close()
        return
    conn.send(("ready", os.getpid()))
    while True:
        try:
            message = conn.recv()
        except (EOFError, OSError):
            return
        if message is None:
            return
        job, data = message
        try:
            result = engine(data)
        except TranscriptionError as exc:
            conn.send((job, "error", (exc.code, str(exc))))
        except Exception as exc:
            logging.getLogger("speech_gigaam.worker").exception("Сбой обработки записи")
            conn.send((job, "error", ("internal", f"{type(exc).__name__}: {exc}")))
        else:
            conn.send((job, "ok", result))


@dataclass(slots=True)
class _Worker:
    """Процесс-обработчик и его канал."""

    slot: int
    process: BaseProcess
    conn: Connection


class ProcessPool:
    """Пул обработчиков с ограниченной очередью; реализует ``TranscriptionService``."""

    def __init__(
        self,
        factory: EngineFactory,
        config: Any,
        workers: int,
        queue_limit: int,
        request_timeout: float,
        start_timeout: float,
        restart_delay: float = 5.0,
    ) -> None:
        self._ctx = multiprocessing.get_context("spawn")
        self._factory = factory
        self._config = config
        self._workers = workers
        self._capacity = workers + queue_limit
        self._request_timeout = request_timeout
        self._start_timeout = start_timeout
        self._restart_delay = restart_delay
        self._idle: asyncio.Queue[_Worker] | None = None
        self._registry: dict[int, _Worker] = {}
        self._background: set[asyncio.Task[Any]] = set()
        self._jobs = itertools.count(1)
        # Свой пул потоков: каждый занятый обработчик держит поток, пока ждёт ответа.
        # Общий пул asyncio меньше (не больше 32 потоков) и нужен ещё /ready.
        self._threads = ThreadPoolExecutor(
            max_workers=workers * 2 + 4, thread_name_prefix="gigaam-pool"
        )
        self._alive = 0
        self._busy = 0
        self._in_flight = 0
        self._ready = False
        self._stopping = False

    async def start(self) -> None:
        """Запустить все обработчики и дождаться, пока каждый загрузит модели.

        Ошибка ``RuntimeError`` — если хоть один не загрузился; остальные тогда
        останавливаются, и служба не стартует.
        """
        self._idle = asyncio.Queue()
        results = await asyncio.gather(
            *(self._in_thread(self._spawn, slot) for slot in range(self._workers)),
            return_exceptions=True,
        )
        started = [r for r in results if isinstance(r, _Worker)]
        failures = [r for r in results if isinstance(r, BaseException)]
        if failures:
            await self._in_thread(self._shutdown, started)
            raise RuntimeError(str(failures[0])) from failures[0]
        for worker in started:
            self._idle.put_nowait(worker)
        self._alive = len(started)
        self._ready = True
        log.info("Обработчиков готово: %d, очередь до %d запросов", self._alive, self._capacity)

    async def stop(self) -> None:
        """Остановить обработчики; запросы в работе получают ошибку."""
        if self._stopping:
            return
        self._stopping = True
        self._ready = False
        for task in list(self._background):
            task.cancel()
        await self._in_thread(self._shutdown, list(self._registry.values()))
        self._threads.shutdown(wait=False, cancel_futures=True)

    async def _in_thread(self, func: Callable[..., T], *args: Any) -> T:
        """Выполнить блокирующий вызов в потоке пула."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._threads, functools.partial(func, *args))

    def status(self) -> ServiceStatus:
        """Состояние пула для проверки готовности."""
        return ServiceStatus(
            ready=self._ready and self._alive > 0,
            workers=self._workers,
            alive=self._alive,
            busy=self._busy,
            waiting=max(0, self._in_flight - self._busy),
            capacity=self._capacity,
        )

    async def transcribe(self, data: bytes) -> Outcome:
        """Распознать файл на свободном обработчике.

        Ошибки: ``ServiceNotReadyError`` — нет живых обработчиков; ``ServiceBusyError`` —
        очередь полна; ``RequestTimeoutError`` — не дождались обработчика или он не
        успел; ``WorkerCrashedError`` — обработчик упал; ошибки распознавания —
        как их вернул обработчик. Отмена запроса не прерывает начатую запись:
        обработчик вернётся в пул, когда закончит.
        """
        if not self._ready or self._idle is None or self._alive == 0:
            raise ServiceNotReadyError("Обработчики не готовы")
        if self._in_flight >= self._capacity:
            raise ServiceBusyError("Все обработчики заняты, очередь полна")
        self._in_flight += 1
        queued = time.perf_counter()
        try:
            worker = await asyncio.wait_for(self._idle.get(), timeout=self._request_timeout)
        except TimeoutError:
            self._in_flight -= 1
            raise RequestTimeoutError(
                f"Свободный обработчик не появился за {self._request_timeout:.0f} с"
            ) from None
        except BaseException:
            self._in_flight -= 1
            raise
        wait_ms = round((time.perf_counter() - queued) * 1000)
        task = asyncio.create_task(self._exchange(worker, data))
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        transcript = await asyncio.shield(task)
        return Outcome(transcript=transcript, wait_ms=wait_ms)

    async def _exchange(self, worker: _Worker, data: bytes) -> Transcript:
        """Отдать файл обработчику и дождаться ответа; освобождает место в очереди."""
        self._busy += 1
        current: _Worker | None = worker
        try:
            if not worker.process.is_alive():
                # Обработчик умер, пока ждал работы: меняем до отправки, запрос не теряем.
                current = await self._replace(worker)
            if current is None:
                raise WorkerCrashedError("Обработчик завершился и не перезапустился")
            active = current
            try:
                return await self._in_thread(self._roundtrip, active, data)
            except (WorkerCrashedError, RequestTimeoutError):
                current = await self._replace(active)
                raise
        finally:
            self._busy -= 1
            self._in_flight -= 1
            if current is not None and self._idle is not None and not self._stopping:
                self._idle.put_nowait(current)

    def _roundtrip(self, worker: _Worker, data: bytes) -> Transcript:
        """Отправить файл и дождаться ответа с тем же номером. Блокирует поток."""
        job = next(self._jobs)
        try:
            worker.conn.send((job, data))
            deadline = time.monotonic() + self._request_timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not worker.conn.poll(remaining):
                    raise RequestTimeoutError(
                        f"Запись не обработана за {self._request_timeout:.0f} с"
                    )
                reply = worker.conn.recv()
                if reply[0] == job:
                    break
        except (EOFError, OSError) as exc:
            raise WorkerCrashedError(f"Обработчик {worker.slot} завершился: {exc}") from exc
        _, state, payload = reply
        if state == "ok":
            result: Transcript = payload
            return result
        code, message = payload
        raise error_from_code(code, message)

    async def _replace(self, worker: _Worker) -> _Worker | None:
        """Остановить обработчик и запустить новый в том же месте.

        Возвращает новый обработчик или ``None``, если запуск не удался; тогда
        повторные попытки идут в фоне с растущей паузой.
        """
        self._alive -= 1
        self._registry.pop(worker.slot, None)
        await self._in_thread(self._terminate, worker)
        if self._stopping:
            return None
        try:
            fresh = await self._in_thread(self._spawn, worker.slot)
        except Exception as exc:
            log.error("Обработчик %d не перезапустился: %s", worker.slot, exc)
            task = asyncio.create_task(self._revive(worker.slot))
            self._background.add(task)
            task.add_done_callback(self._background.discard)
            return None
        self._alive += 1
        log.warning("Обработчик %d перезапущен", worker.slot)
        return fresh

    async def _revive(self, slot: int) -> None:
        """Пробовать запустить обработчик, пока не получится или служба не остановится."""
        delay = self._restart_delay
        while not self._stopping:
            await asyncio.sleep(delay)
            try:
                fresh = await self._in_thread(self._spawn, slot)
            except Exception as exc:
                log.error("Обработчик %d снова не запустился: %s", slot, exc)
                delay = min(delay * 2, 60.0)
                continue
            self._alive += 1
            if self._idle is not None:
                self._idle.put_nowait(fresh)
            log.warning("Обработчик %d запущен после сбоя", slot)
            return

    def _spawn(self, slot: int) -> _Worker:
        """Запустить процесс и дождаться загрузки моделей. Блокирует поток."""
        parent, child = self._ctx.Pipe(duplex=True)
        process = self._ctx.Process(
            target=_worker_main,
            args=(self._factory, self._config, child),
            name=f"gigaam-worker-{slot}",
            daemon=True,
        )
        process.start()
        child.close()
        worker = _Worker(slot, process, parent)
        try:
            if not parent.poll(self._start_timeout):
                raise RuntimeError(
                    f"Обработчик {slot} не загрузил модели за {self._start_timeout:.0f} с"
                )
            state, detail = parent.recv()
        except (EOFError, OSError) as exc:
            process.join(1)
            self._terminate(worker)
            raise RuntimeError(
                f"Обработчик {slot} завершился при загрузке, код {process.exitcode}"
            ) from exc
        except RuntimeError:
            self._terminate(worker)
            raise
        if state != "ready":
            self._terminate(worker)
            raise RuntimeError(f"Обработчик {slot} не загрузил модели: {detail}")
        self._registry[slot] = worker
        log.info("Обработчик %d готов, процесс %s", slot, detail)
        return worker

    @staticmethod
    def _terminate(worker: _Worker) -> None:
        """Закрыть канал и завершить процесс, при необходимости принудительно."""
        try:
            worker.conn.close()
        except OSError:
            pass
        process = worker.process
        if process.is_alive():
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)

    def _shutdown(self, workers: list[_Worker]) -> None:
        """Попросить обработчики завершиться и остановить оставшиеся."""
        for worker in workers:
            try:
                worker.conn.send(None)
            except OSError:
                pass
        for worker in workers:
            worker.process.join(5)
            self._terminate(worker)
        self._registry.clear()

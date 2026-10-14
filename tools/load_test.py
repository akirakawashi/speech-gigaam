"""Нагрузка на службу: N параллельных запросов по папке записей.

Показывает скорость по часам: сколько записей и часов звука прошло, во сколько раз
быстрее реального времени и сколько записей выйдет за 15 минут. Нужен только Python 3.

    python3 tools/load_test.py --url http://127.0.0.1:8090 --audio-dir ../audio \
        --concurrency 10 --limit 500
"""

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

AUDIO = {".mp3", ".wav", ".ogg", ".opus", ".m4a", ".flac", ".webm"}


def send(url: str, path: Path, timeout: float) -> tuple[str, float, int]:
    """Отправить файл, повторяя на «занято». Возвращает итог, длительность звука, повторы."""
    data = path.read_bytes()
    retries = 0
    while True:
        request = urllib.request.Request(
            f"{url}/v1/transcriptions",
            data=data,
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return "ok", float(json.loads(response.read())["duration_s"]), retries
        except urllib.error.HTTPError as error:
            if error.code == 503:
                retries += 1
                time.sleep(float(error.headers.get("Retry-After", "2")))
                continue
            try:
                code = json.loads(error.read() or b"{}").get("error", {}).get("code")
            except ValueError:
                code = None
            return str(code or error.code), 0.0, retries
        except (urllib.error.URLError, TimeoutError) as error:
            return f"сеть: {error}", 0.0, retries


def main() -> int:
    cli = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    cli.add_argument("--url", default="http://127.0.0.1:8090")
    cli.add_argument("--audio-dir", type=Path, required=True)
    cli.add_argument("--concurrency", type=int, default=4)
    cli.add_argument("--limit", type=int, default=0, help="сколько файлов отправить; 0 — все")
    cli.add_argument("--timeout", type=float, default=900)
    args = cli.parse_args()

    files = sorted(
        p for p in args.audio_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO
    )
    if args.limit:
        files = files[: args.limit]
    if not files:
        print("В папке нет записей", file=sys.stderr)
        return 2
    lock = threading.Lock()
    outcomes: dict[str, int] = {}
    audio_seconds = 0.0
    retries_total = 0
    done = 0

    def task(path: Path) -> None:
        nonlocal audio_seconds, retries_total, done
        outcome, seconds, retries = send(args.url, path, args.timeout)
        with lock:
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            audio_seconds += seconds
            retries_total += retries
            done += 1
            print(f"\r{done}/{len(files)}", end="", file=sys.stderr, flush=True)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        list(executor.map(task, files))
    wall = time.perf_counter() - started
    print(file=sys.stderr)
    print(f"Файлов: {len(files)}, параллельно {args.concurrency}, по часам {wall:.1f} с")
    print(f"Итоги: {outcomes}; повторов после «занято»: {retries_total}")
    print(
        f"Звука: {audio_seconds / 3600:.2f} ч; "
        f"быстрее реального времени в {audio_seconds / wall:.0f} раз"
    )
    print(f"Записей за 15 минут: около {len(files) / wall * 900:.0f}")
    return 0 if outcomes.get("ok", 0) == len(files) else 1


if __name__ == "__main__":
    sys.exit(main())

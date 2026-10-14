"""Сверка работающей службы с прогоном стенда temptest (урезанный режим GigaAM).

Отправляет записи прогона в службу и сравнивает с результатом стенда: SHA256
декодированного звука, число фрагментов речи и текст. Нужен только Python 3 —
зависимости проекта не требуются, скрипт можно запускать прямо на сервере.

    python3 tools/compare_with_stand.py --url http://127.0.0.1:8090 \
        --stand-run ../temptest/benchmark-results/gigaam-trt-… --audio-dir ../audio --limit 100
"""

import argparse
import difflib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def post(url: str, data: bytes, timeout: float) -> dict[str, Any]:
    """Отправить файл; на «занято» подождать и повторить."""
    while True:
        request = urllib.request.Request(
            f"{url}/v1/transcriptions",
            data=data,
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body: dict[str, Any] = json.loads(response.read())
                return body
        except urllib.error.HTTPError as error:
            if error.code == 503:
                time.sleep(float(error.headers.get("Retry-After", "2")))
                continue
            raise RuntimeError(
                f"HTTP {error.code}: {error.read().decode('utf-8', 'replace')}"
            ) from error


def stand_texts(run: Path, records: list[dict[str, Any]], system: str) -> list[str | None]:
    """Тексты стенда: из записи/NNNN/<system>.txt, иначе из отчёт.md по порядку записей."""
    texts: list[str | None] = []
    for record in records:
        path = run / record["directory"] / f"{system}.txt"
        texts.append(path.read_text(encoding="utf-8").strip() if path.is_file() else None)
    if all(text is not None for text in texts):
        return texts
    report = run / "отчёт.md"
    if not report.is_file():
        return texts
    blocks: list[str] = []
    current: list[str] | None = None
    for line in report.read_text(encoding="utf-8").split("\n"):
        if line.startswith("## "):
            if current is not None:
                blocks.append("\n".join(current[1:]).strip())
            current = []
        elif current is not None:
            current.append(line)
    if current is not None:
        blocks.append("\n".join(current[1:]).strip())
    if len(blocks) != len(records):
        return texts
    return [text if text is not None else blocks[i] for i, text in enumerate(texts)]


def stand_segments(run: Path, record: dict[str, Any], system: str) -> int | None:
    """Число фрагментов речи у стенда, если файл записи есть."""
    path = run / record["directory"] / f"{system}.json"
    if not path.is_file():
        return None
    return len(json.loads(path.read_text(encoding="utf-8")).get("speech_segments", []))


def source_path(audio_dir: Path, source: str) -> Path:
    """Путь к записи: путь стенда внутри /recordings переносится в папку аудио."""
    relative = (
        source.split("/recordings/", 1)[-1] if "/recordings/" in source else Path(source).name
    )
    return audio_dir / relative


def main() -> int:
    cli = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    cli.add_argument("--url", default="http://127.0.0.1:8090")
    cli.add_argument(
        "--stand-run", type=Path, required=True, help="папка прогона стенда с условия.json"
    )
    cli.add_argument(
        "--audio-dir", type=Path, required=True, help="папка записей, которую видел стенд"
    )
    cli.add_argument("--system", default="gigaam_trt", help="ключ системы в прогоне стенда")
    cli.add_argument("--limit", type=int, default=0, help="сколько записей проверить; 0 — все")
    cli.add_argument("--timeout", type=float, default=900)
    args = cli.parse_args()

    records = json.loads((args.stand_run / "условия.json").read_text(encoding="utf-8"))["records"]
    if args.limit:
        records = records[: args.limit]
    texts = stand_texts(args.stand_run, records, args.system)
    pcm_same = text_same = text_known = segments_same = segments_known = words = changed = 0
    examples: list[str] = []
    failures: list[str] = []
    started = time.perf_counter()
    for index, (record, expected) in enumerate(zip(records, texts, strict=True), 1):
        audio = record["audio"]
        try:
            result = post(
                args.url, source_path(args.audio_dir, audio["source"]).read_bytes(), args.timeout
            )
        except (RuntimeError, OSError, ValueError) as error:
            # Один сбой не должен останавливать сверку: запись засчитывается как ошибка.
            failures.append(f"{record['directory']}: {error}")
            continue
        pcm_same += result["pcm_sha256"] == audio["sha256_pcm"]
        segments = stand_segments(args.stand_run, record, args.system)
        if segments is not None:
            segments_known += 1
            segments_same += result["speech_segments"] == segments
        ours = "\n".join(reply["text"] for reply in result["replies"])
        if expected is not None:
            a, b = expected.split(), ours.split()
            text_known += 1
            words += len(a)
            text_same += ours == expected
            for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
                None, a, b, autojunk=False
            ).get_opcodes():
                if tag != "equal":
                    changed += max(i2 - i1, j2 - j1)
                    if len(examples) < 20:
                        examples.append(
                            f"{record['directory']}: "
                            f"стенд «{' '.join(a[max(0, i1 - 3) : i2 + 3])}» — "
                            f"служба «{' '.join(b[max(0, j1 - 3) : j2 + 3])}»"
                        )
        print(f"\r{index}/{len(records)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    total = len(records)
    print(f"Записей: {total}, за {time.perf_counter() - started:.0f} с")
    print(f"Звук совпал (SHA256 PCM): {pcm_same} из {total}")
    if segments_known:
        print(f"Число фрагментов речи совпало: {segments_same} из {segments_known}")
    print(
        f"Текст совпал полностью: {text_same} из {text_known}; "
        f"слов у стенда {words}, отличается {changed}"
    )
    for example in examples:
        print("  ", example)
    if failures:
        print(f"Ошибок запроса: {len(failures)}")
        for failure in failures[:20]:
            print("  ", failure)
    ok = not failures and pcm_same == total and text_same == text_known
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""Ошибки распознавания с устойчивыми кодами.

Код передаётся из процесса-обработчика в HTTP-процесс и попадает в ответ API:
по нему вызывающий решает, повторять запрос или нет.
"""

from typing import ClassVar


class TranscriptionError(Exception):
    """Общая ошибка распознавания записи."""

    code: ClassVar[str] = "internal"


class AudioDecodeError(TranscriptionError):
    """Файл не читается как звук или в нём нет звука. Повтор не поможет."""

    code = "bad_audio"


class AudioTooLongError(TranscriptionError):
    """Запись длиннее разрешённого предела."""

    code = "audio_too_long"


class FileTooLargeError(TranscriptionError):
    """Файл больше разрешённого размера запроса."""

    code = "file_too_large"


class EngineError(TranscriptionError):
    """Сбой модели или сервера Triton. Повтор позже может помочь."""

    code = "engine_failed"


class ServiceBusyError(TranscriptionError):
    """Все обработчики заняты и очередь полна. Нужно повторить позже."""

    code = "busy"


class ServiceNotReadyError(TranscriptionError):
    """Обработчики ещё загружают модели или служба останавливается."""

    code = "not_ready"


class WorkerCrashedError(TranscriptionError):
    """Процесс-обработчик завершился во время записи; он перезапускается."""

    code = "worker_failed"


class RequestTimeoutError(TranscriptionError):
    """Обработка записи не уложилась в отведённое время; обработчик перезапущен."""

    code = "timeout"


_BY_CODE: dict[str, type[TranscriptionError]] = {
    cls.code: cls
    for cls in (
        TranscriptionError,
        AudioDecodeError,
        AudioTooLongError,
        FileTooLargeError,
        EngineError,
        ServiceBusyError,
        ServiceNotReadyError,
        WorkerCrashedError,
        RequestTimeoutError,
    )
}


def error_from_code(code: str, message: str) -> TranscriptionError:
    """Восстановить ошибку по коду, пришедшему из процесса-обработчика.

    Неизвестный код становится общей ошибкой ``internal``. Побочных эффектов нет.
    """
    return _BY_CODE.get(code, TranscriptionError)(message)

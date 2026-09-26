import threading


_stop_requested = False
_lock = threading.Lock()


def reset_stop_request() -> None:
    global _stop_requested

    with _lock:
        _stop_requested = False


def request_stop() -> None:
    global _stop_requested

    with _lock:
        _stop_requested = True


def is_stop_requested() -> bool:
    with _lock:
        return _stop_requested
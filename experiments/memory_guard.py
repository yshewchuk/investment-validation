"""Resident-memory watchdog for SIZE ceilings; RLIMIT_RSS is a no-op on Linux."""
import signal
import threading
from contextlib import contextmanager

from engine.v2.ops.errors import fail


def rss_bytes() -> int:
    """Read VmRSS in bytes; an unreadable /proc refuses rather than guessing."""
    try:
        with open("/proc/self/status", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    raise fail("RESOURCE_UNAVAILABLE", "resident memory is unreadable")


def start_rss_watchdog(limit_bytes, interval=0.05):
    """Start the RSS watchdog for this process; returns a stop() callable."""
    main_thread = threading.get_ident()
    rss_bytes()
    stopped = threading.Event()
    active = threading.Event()
    active.set()

    def watch():
        while not stopped.wait(interval):
            try:
                exceeded = rss_bytes() > limit_bytes
            except Exception:
                exceeded = True
            if exceeded:
                signal.pthread_kill(main_thread, signal.SIGUSR1)
                return

    def exhausted(signum, frame):
        if active.is_set():
            raise MemoryError

    previous = signal.signal(signal.SIGUSR1, exhausted)
    thread = threading.Thread(target=watch, daemon=True)
    thread.start()

    def stop():
        active.clear()
        stopped.set()
        thread.join()
        if previous is signal.SIG_DFL:
            signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        else:
            signal.signal(signal.SIGUSR1, previous)

    return stop


@contextmanager
def rss_watchdog(limit_bytes, interval=0.05):
    """Interrupt the calling thread with MemoryError once RSS exceeds limit_bytes."""
    stop_watchdog = start_rss_watchdog(limit_bytes, interval)
    try:
        yield
    finally:
        stop_watchdog()

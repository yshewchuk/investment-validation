"""Retain bounded pytest failures from mutmut's otherwise-silent stats child.

Loaded through pytest's supported plugin interface, not a mutmut patch. It
never changes test outcomes or runs tests again, and ignores mutant runs.
"""
import os
import sys
from pathlib import Path

LIMIT_BYTES = 64 * 1024
TRUNCATED = b"\n[stats diagnostic truncated]\n"


def warn_unavailable(error):
    # Even a closed original stderr must not replace the underlying outcome.
    try:
        if sys.__stderr__ is not None:
            print(f"[mutation_pilot] stats diagnostic unavailable: {error}", file=sys.__stderr__)
    except (OSError, ValueError):
        pass


def pytest_configure(config):
    destination = os.environ.get("MUTATION_PILOT_STATS_DIAGNOSTIC")
    if destination and os.environ.get("MUTANT_UNDER_TEST") == "stats":
        config.pluginmanager.register(StatsDiagnostics(Path(destination)), "stats-diagnostics")


class StatsDiagnostics:
    def __init__(self, destination):
        self.destination = destination
        self.unavailable = False

    def _record(self, text):
        if self.unavailable:
            return
        try:
            self._write(text)
        except (OSError, ValueError) as error:
            self.unavailable = True
            # Telemetry must not replace the original pytest outcome. Bypass
            # mutmut's fork-local capture so loss of diagnostics is explicit.
            warn_unavailable(error)

    def _write(self, text):
        # The stats runner is serial. Keep a hard byte cap across reports and
        # sessions; the driver removes stale evidence before each backend run.
        with self.destination.open("ab") as stream:
            remaining = LIMIT_BYTES - stream.tell()
            if remaining <= 0:
                return
            payload = text.encode("utf-8", "replace")
            if len(payload) > remaining:
                available = max(0, remaining - len(TRUNCATED))
                head = available // 2
                # Keep the test identity AND the final exception detail.
                payload = (payload[:head] + TRUNCATED[:remaining]
                           + payload[len(payload) - (available - head):])
            stream.write(payload)

    def pytest_runtest_logreport(self, report):
        if report.failed:
            # Typed refusals are often captured stdout, not the assertion's
            # traceback. Retain both in the same capped report.
            sections = "".join(f"\n--- {name} ---\n{content}" for name, content in report.sections)
            self._record(f"\nFAILED {report.nodeid} [{report.when}]\n{report.longreprtext}\n{sections}")

    def pytest_collectreport(self, report):
        if report.failed:
            sections = "".join(f"\n--- {name} ---\n{content}" for name, content in report.sections)
            self._record(f"\nCOLLECTION FAILED {report.nodeid}\n{report.longreprtext}\n{sections}")

    def pytest_internalerror(self, excrepr):
        self._record(f"\nPYTEST INTERNAL ERROR\n{excrepr}\n")

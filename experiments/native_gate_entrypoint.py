"""Fixed native smoke composition root; load pricing only after the launch gate."""
from engine.v2.ops.errors import fail
from engine.v2.ops.worker import main as run_worker


def dispatch(worker, parameters, root, *, envelope=None):
    """Admit only this server-owned kind; never dispatch legacy or arbitrary code."""
    if worker != "native_gate_pricing":
        raise fail("INVALID_EXPERIMENT_SPEC", "unsupported native gate worker")
    from experiments.native_gate_worker import run

    return run(parameters, root)


if __name__ == "__main__":
    raise SystemExit(run_worker(dispatcher=dispatch))

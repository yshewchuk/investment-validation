"""Fixed native composition root; load pricing/evaluation after the launch gate."""
from engine.v2.ops.errors import fail
from engine.v2.ops.worker import main as run_worker


def dispatch(worker, parameters, root, *, envelope=None):
    """Admit only this server-owned kind; never dispatch legacy or arbitrary code."""
    if worker == "native_gate_pricing":
        from experiments.native_gate_worker import run
    elif worker == "native_gate_evaluation":
        from experiments.native_gate_evaluation import run
    else:
        raise fail("INVALID_EXPERIMENT_SPEC", "unsupported native gate worker")

    return run(parameters, root)


if __name__ == "__main__":
    raise SystemExit(run_worker(dispatcher=dispatch))

"""Neutral containment boundary retains the installed strategy helper."""
import os
import subprocess
import sys

from experiments.distribution_containment import contained
from experiments.distribution_run_plan import contained as legacy_contained


def test_legacy_export_is_the_same_neutral_context_manager():
    assert legacy_contained is contained


def test_neutral_containment_import_refuses_strategy_dependencies_in_fresh_process():
    script = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname in ('experiments.distribution_run_request', 'experiments.distribution_market',
                        'experiments.distribution_book', 'engine.v2.research.magnitude_distribution_payoff'):
            raise AssertionError('economic import: ' + fullname)
sys.meta_path.insert(0, Block())
from experiments.distribution_containment import contained
from engine.v2.ops.errors import OpsError
try:
    with contained({'memory_bytes': 1, 'cpu_seconds': 1, 'wall_seconds': 1}, 0):
        raise AssertionError('invalid duration entered')
except OpsError as error:
    assert error.code == 'INVALID_EXPERIMENT_SPEC'
print('neutral')
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
        timeout=15, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'neutral'

"""tests.test_orchestrate_summary — --only must not clobber other cells' status.

Bug: each `orchestrate --only <cell>` invocation (the dynamic GPU-queue
pattern in run_scripts/*_queue.sh) only knows about the one cell it ran, but
write_summary() always reports the FULL queue's `names` list. Without
seeding results for out-of-scope cells from their own on-disk marker, every
--only invocation overwrote sweep_summary.csv with 'pending' placeholders
for every other cell — even ones that had already finished (or failed) —
making the CSV lie about cells the current invocation never touched.

Torch-free; orchestrate.py is stdlib-only.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(REPO_ROOT / 'experiments' / 'scripts'))
import orchestrate  # noqa: E402


def test_seed_results_reads_existing_markers_for_out_of_scope_cells(tmp_path):
    run_root = str(tmp_path)
    orchestrate.write_marker(run_root, 'done_cell', {'exit_code': 0, 'wall_seconds': 100.0})
    orchestrate.write_marker(run_root, 'failed_cell', {'exit_code': 1, 'wall_seconds': 5.0})
    # 'untouched_cell' has no marker at all — never ran.

    names = ['done_cell', 'failed_cell', 'untouched_cell', 'in_scope_cell']
    results = orchestrate.seed_results_from_markers(run_root, names, in_scope={'in_scope_cell'})

    assert results['done_cell'] == {'status': 'done', 'exit_code': 0, 'wall_seconds': 100.0}
    assert results['failed_cell'] == {'status': 'failed', 'exit_code': 1, 'wall_seconds': 5.0}
    assert 'untouched_cell' not in results
    # in_scope_cell is left for the caller's own run loop to populate.
    assert 'in_scope_cell' not in results


def test_seed_results_empty_when_no_markers_exist(tmp_path):
    results = orchestrate.seed_results_from_markers(
        str(tmp_path), ['a', 'b'], in_scope=set()
    )
    assert results == {}

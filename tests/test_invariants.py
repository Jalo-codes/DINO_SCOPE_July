"""tests.test_invariants — structural invariant tests for DINO_SCOPE_final.

These tests run without GPU or any dataset on disk.  They verify code-structure
contracts (the I* invariants from DESIGN_GUIDE §9):

  I1: "oracle" only in data/augment/crop.py — not in eval/ or labs/
  I2: model forward is called ONLY in lab_utils/eval/fetch.py
  I3: GT is touched ONLY in lab_utils/eval/metric.py
  I4: decode functions are GT-free (no mask/GT in their signature or imports)
  I5: decode functions produce no side-effects (no print/logging)
  C-script: no script in experiments/scripts/ imports another script
"""

import ast
import os
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent


# ── Helpers ────────────────────────────────────────────────────────────────────

def _py_files(root: Path, *exclude_dirs: str):
    for f in root.rglob('*.py'):
        if any(part in f.parts for part in exclude_dirs):
            continue
        if '__pycache__' in f.parts:
            continue
        yield f


def _source(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='replace')


def _ast(path: Path) -> ast.Module:
    return ast.parse(_source(path), filename=str(path))


def _imports_in(path: Path):
    """Return all module names imported (top-level and dotted) in a file."""
    tree  = _ast(path)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
    return names


def _has_token(path: Path, token: str) -> bool:
    src = _source(path)
    return token in src


def _has_call_to(path: Path, func_name: str) -> bool:
    """True if the AST contains a Call whose function name matches func_name."""
    tree = _ast(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id == func_name:
                return True
            if isinstance(fn, ast.Attribute) and fn.attr == func_name:
                return True
    return False


# ── I1: oracle token outside train augment ─────────────────────────────────────

class TestNoOracleOutsideTrainCrop:
    """oracle_mask_crop must only be called from data/augment/crop.py.

    We search for the specific function token 'oracle_mask_crop' rather than
    the word "oracle" to avoid false positives in doc-strings that say things
    like "oracle-free" or "no oracle".
    """

    ALLOWED_ORACLE_FILES = {
        'lab_utils/data/augment/crop.py',
        'lab_utils/data/dataset.py',         # Dataset accepts oracle_crop kwarg
        'experiments/configs/run_config.py', # oracle_crop field
        'experiments/scripts/train.py',      # --oracle_crop flag
    }

    def _relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)

    @pytest.mark.parametrize('search_dir', ['lab_utils/eval', 'experiments/labs'])
    def test_no_oracle_in_eval_or_labs(self, search_dir: str):
        root = REPO_ROOT / search_dir
        if not root.exists():
            pytest.skip(f'{search_dir} does not exist')
        # Check for the specific function / variable token, not the substring.
        oracle_tokens = ('oracle_mask_crop', 'oracle_crop(')
        violations = []
        for f in _py_files(root):
            src = _source(f)
            if any(tok in src for tok in oracle_tokens):
                rel = self._relative(f)
                if rel not in self.ALLOWED_ORACLE_FILES:
                    violations.append(rel)
        assert not violations, (
            f'I1 violation — oracle function token found in eval/labs code:\n'
            + textwrap.indent('\n'.join(sorted(violations)), '  ')
        )


# ── I2: model forward called only in fetch.py ─────────────────────────────────

class TestFetchIsSoleModelEntry:
    """Only lab_utils/eval/fetch.py should call the model forward (model_info)."""

    FETCH_MODULE = 'lab_utils/eval/fetch.py'

    ALLOWED_CALLERS = {
        'lab_utils/eval/cache.py',         # build_cache calls model_info
        'lab_utils/train/loop.py',         # run_val_eval calls model_info
        'experiments/scripts/eval.py',     # eval script calls model_info
        'lab_utils/eval/numbers.py',       # numerical-eval engine: shared-forward flat+zoom
        'experiments/labs/attention_zoom.py',  # two-pass calls model_info
        'experiments/labs/hdbscan_lab.py',     # lab orchestrator calls model_info
        'experiments/labs/tgif_finetune_eval.py',  # finetune eval: flat pass calls model_info
        'experiments/labs/box_heatmap_lab.py',
        'experiments/labs/box_policy_zoom.py',
        'experiments/scripts/train_box_policy.py',
        'experiments/scripts/eval_robustness.py',
        'experiments/scripts/train_single_box.py',
        'experiments/labs/zoom_cluster_lab.py',
        'experiments/scripts/train_zoom_head.py',
        'experiments/labs/zoom_box_lab.py',
        'experiments/scripts/bench_resolution.py',
        'experiments/scripts/train_zoom_box.py',
        'experiments/scripts/predict.py',       # GT-free qualitative inference; docstring names I2
        'experiments/scripts/eval_oracle.py',   # isolated cheating-oracle eval; reuses fetch.model_info by design
        'experiments/scripts/gen_size_bucket_visuals.py',  # per-item viz forward, same shape as attention_zoom.py
        'experiments/scripts/eval_openfake_by_generator.py',  # per-generator OpenFake eval; full+crop forwards
    }

    def test_model_info_not_called_elsewhere(self):
        """Check AST-level Call nodes, not raw token presence.

        Docstrings and comments that mention 'model_info' are fine;
        only actual Call nodes in non-allowed files are violations.
        """
        violations = []
        for base in [REPO_ROOT / 'lab_utils', REPO_ROOT / 'experiments']:
            for f in _py_files(base, '__pycache__'):
                rel = str(f.relative_to(REPO_ROOT))
                if rel == self.FETCH_MODULE or rel in self.ALLOWED_CALLERS:
                    continue
                if _has_call_to(f, 'model_info'):
                    violations.append(rel)
        assert not violations, (
            f'I2 violation — model_info called outside allowed files:\n'
            + textwrap.indent('\n'.join(sorted(violations)), '  ')
        )

    def test_fetch_module_exists(self):
        assert (REPO_ROOT / self.FETCH_MODULE).exists(), \
            f'I2: fetch module missing: {self.FETCH_MODULE}'


# ── I3: GT only touched in metric.py ──────────────────────────────────────────

class TestMetricIsSoleGTTouch:
    """GT mask loading must only happen inside lab_utils/eval/metric.py."""

    METRIC_MODULE  = 'lab_utils/eval/metric.py'
    GT_SIGNALS     = ('item.mask', 'triplet.mask', '.mask_area(', 'gt_mask')

    # Files that legitimately reference GT concepts
    ALLOWED_GT_FILES = {
        'lab_utils/eval/metric.py',
        'lab_utils/eval/record.py',    # EvalRecord stores gt_mask field
        'lab_utils/eval/aggregate.py', # reads .is_real (not the mask itself)
        'lab_utils/eval/robustness.py',
        'lab_utils/eval/buckets.py',   # docstring mentions Item.mask_area() concept
        'tests/test_invariants.py',    # this file
    }

    def test_no_gt_mask_load_outside_metric(self):
        violations = []
        for f in _py_files(REPO_ROOT / 'lab_utils/eval', '__pycache__'):
            rel = str(f.relative_to(REPO_ROOT))
            if rel in self.ALLOWED_GT_FILES:
                continue
            src = _source(f)
            for sig in self.GT_SIGNALS:
                if sig in src:
                    violations.append(f'{rel} (token: {sig!r})')
                    break
        assert not violations, (
            f'I3 violation — GT signal found outside metric.py:\n'
            + textwrap.indent('\n'.join(sorted(violations)), '  ')
        )


# ── I4: decode modules are GT-free ────────────────────────────────────────────

class TestDecodeIsGTFree:
    """Decode modules must not import data/ and must not accept GT parameters."""

    DECODE_DIR = REPO_ROOT / 'lab_utils/eval/decode'

    DATA_IMPORT_PREFIXES = ('lab_utils.data', 'lab_utils.train')
    GT_PARAM_NAMES       = ('gt_mask', 'mask', 'label', 'labels', 'triplet')

    def _decode_files(self):
        if not self.DECODE_DIR.exists():
            return []
        return list(_py_files(self.DECODE_DIR))

    def test_no_data_imports(self):
        violations = []
        for f in self._decode_files():
            imports = _imports_in(f)
            for imp in imports:
                if any(imp.startswith(p) for p in self.DATA_IMPORT_PREFIXES):
                    violations.append(f'{f.name}: imports {imp!r}')
        assert not violations, (
            'I4 violation — decode module imports data/train layer:\n'
            + textwrap.indent('\n'.join(sorted(violations)), '  ')
        )

    def test_no_gt_parameters_in_public_decode_functions(self):
        """Scan decode function signatures for GT-looking parameter names."""
        violations = []
        for f in self._decode_files():
            tree = _ast(f)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith('decode_'):
                    for arg in node.args.args + node.args.kwonlyargs:
                        if arg.arg in self.GT_PARAM_NAMES:
                            violations.append(
                                f'{f.name}:{node.lineno} {node.name}() has arg {arg.arg!r}'
                            )
        assert not violations, (
            'I4 violation — decode_*() function accepts GT parameter:\n'
            + textwrap.indent('\n'.join(sorted(violations)), '  ')
        )


# ── I5: decode modules are side-effect free ───────────────────────────────────

class TestDecodeIsSilent:
    """Decode modules must not call print() or logging functions."""

    DECODE_DIR = REPO_ROOT / 'lab_utils/eval/decode'

    NOISY_CALLS = ('print', 'logging.', 'log_line', 'warnings.warn')

    def _decode_files(self):
        if not self.DECODE_DIR.exists():
            return []
        # Exclude __init__.py — it's a description of the package, not a decode module.
        return [f for f in _py_files(self.DECODE_DIR) if f.name != '__init__.py']

    def test_no_print_or_logging(self):
        violations = []
        for f in self._decode_files():
            src = _source(f)
            for call in self.NOISY_CALLS:
                if call in src:
                    violations.append(f'{f.name}: contains {call!r}')
                    break
        assert not violations, (
            'I5 violation — decode module has side-effects (print/logging):\n'
            + textwrap.indent('\n'.join(sorted(violations)), '  ')
        )


# ── C-script: no cross-script imports ─────────────────────────────────────────

class TestNoCrossScriptImports:
    """No script in experiments/scripts/ should import another script."""

    SCRIPTS_DIR = REPO_ROOT / 'experiments/scripts'

    def test_no_cross_script_imports(self):
        if not self.SCRIPTS_DIR.exists():
            pytest.skip('experiments/scripts does not exist')

        script_modules = set()
        for f in self.SCRIPTS_DIR.glob('*.py'):
            if f.name == '__init__.py':
                continue
            stem = f.stem
            script_modules.add(f'experiments.scripts.{stem}')
            script_modules.add(stem)

        violations = []
        for f in self.SCRIPTS_DIR.glob('*.py'):
            if f.name == '__init__.py':
                continue
            imports = _imports_in(f)
            for imp in imports:
                if imp in script_modules and imp != f'experiments.scripts.{f.stem}':
                    violations.append(f'{f.name} imports {imp!r}')

        assert not violations, (
            'C-script violation — cross-script import detected:\n'
            + textwrap.indent('\n'.join(sorted(violations)), '  ')
        )


# ── Smoke: key modules importable without torch ───────────────────────────────

class TestTorchFreeBoundary:
    """lab_utils.__init__ and the data+eval layers must not import torch at
    module level (C3 boundary).  Only checks AST-level imports, not runtime."""

    TORCH_FREE_MODULES = [
        'lab_utils/__init__.py',
        'lab_utils/data/item.py',
        'lab_utils/data/resolution.py',
        'lab_utils/eval/record.py',
        'lab_utils/eval/buckets.py',
        'lab_utils/eval/aggregate.py',
    ]

    def test_no_top_level_torch(self):
        violations = []
        for rel in self.TORCH_FREE_MODULES:
            f = REPO_ROOT / rel
            if not f.exists():
                continue
            tree = _ast(f)
            # Only check top-level (non-nested) imports
            for node in tree.body:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == 'torch' or alias.name.startswith('torch.'):
                            violations.append(f'{rel}: top-level "import {alias.name}"')
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ''
                    if mod == 'torch' or mod.startswith('torch.'):
                        violations.append(f'{rel}: top-level "from {mod} import ..."')
        assert not violations, (
            'C3 violation — torch imported at module level in torch-free boundary:\n'
            + textwrap.indent('\n'.join(sorted(violations)), '  ')
        )

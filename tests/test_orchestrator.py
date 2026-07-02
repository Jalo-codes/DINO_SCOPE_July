import os
import json
import tempfile
import numpy as np
from pathlib import Path
from lab_utils.eval.aggregate import save_summary_json
from experiments.scripts.orchestrate import build_argv, last_metrics_row

def test_save_summary_json():
    # Make some dummy summaries
    summaries = {
        "kmeans": {
            "image_auc": 0.95,
            "splices": {
                "f1": {"n": 5, "median": 0.85, "mean": 0.83, "std": 0.05, "p25": 0.8, "p75": 0.9},
                "iou": {"n": 5, "median": 0.75, "mean": 0.73, "std": 0.05, "p25": 0.7, "p75": 0.8}
            },
            "reals": {
                "accuracy": {"n": 3, "mean": 0.98}
            },
            "by_bucket": {
                "100px": {
                    "f1": {"n": 2, "median": 0.80}
                }
            }
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "summary.json")
        save_summary_json(path, summaries)
        
        with open(path) as f:
            data = json.load(f)
            
        # Since there is only 1 decoder, keys should not have a prefix.
        assert data["image_auc"] == 0.95
        assert data["f1_median"] == 0.85
        assert data["f1_mean"] == 0.83
        assert data["reals_acc"] == 0.98
        assert data["bucket_100px_f1_median"] == 0.80
        
def test_save_summary_json_multi_decoder():
    summaries = {
        "kmeans": {
            "image_auc": 0.95,
            "splices": {"f1": {"n": 5, "median": 0.85}}
        },
        "hdbscan": {
            "image_auc": 0.92,
            "splices": {"f1": {"n": 5, "median": 0.80}}
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "summary.json")
        save_summary_json(path, summaries)
        
        with open(path) as f:
            data = json.load(f)
            
        # Multiple decoders -> prefixed keys
        assert data["kmeans_image_auc"] == 0.95
        assert data["kmeans_f1_median"] == 0.85
        assert data["hdbscan_image_auc"] == 0.92
        assert data["hdbscan_f1_median"] == 0.80

def test_orchestrate_build_argv():
    entry_eval = {
        "name": "eval_job",
        "module": "experiments.scripts.eval",
        "args": {
            "checkpoint": "/path/to/best.pt"
        }
    }
    entry_train = {
        "name": "train_job",
        "module": "experiments.scripts.train",
        "args": {
            "lr": 1e-4
        }
    }
    base_args = {"device": "cuda"}
    run_root = "/tmp/runs"
    
    argv_eval = build_argv(entry_eval, base_args, run_root)
    assert "--summary_out" in argv_eval
    assert argv_eval[argv_eval.index("--summary_out") + 1] == "/tmp/runs/eval_job/eval_summary.json"
    assert "--checkpoint" in argv_eval
    
    argv_train = build_argv(entry_train, base_args, run_root)
    assert "--checkpoint_root" in argv_train
    assert argv_train[argv_train.index("--checkpoint_root") + 1] == "/tmp/runs/train_job"
    assert "--lr" in argv_train

def test_orchestrate_substitution(monkeypatch):
    entry = {
        "name": "job",
        "module": "experiments.scripts.eval",
        "args": {
            "checkpoint": "{env:TEST_RUNS_DIR,/fallback}/model.pt"
        }
    }
    # Case 1: env var not set, fallback should be used
    monkeypatch.delenv("TEST_RUNS_DIR", raising=False)
    argv = build_argv(entry, {}, "/tmp/runs")
    assert argv[argv.index("--checkpoint") + 1] == "/fallback/model.pt"

    # Case 2: env var set, value should be substituted
    monkeypatch.setenv("TEST_RUNS_DIR", "/my/runs")
    argv = build_argv(entry, {}, "/tmp/runs")
    assert argv[argv.index("--checkpoint") + 1] == "/my/runs/model.pt"

def test_orchestrate_last_metrics_row():
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Test when eval_summary.json exists
        run_name = "job1"
        job_dir = os.path.join(tmpdir, run_name)
        os.makedirs(job_dir, exist_ok=True)
        
        eval_data = {"f1_median": 0.87, "image_auc": 0.96}
        with open(os.path.join(job_dir, "eval_summary.json"), "w") as f:
            json.dump(eval_data, f)
            
        metrics = last_metrics_row(tmpdir, run_name)
        assert metrics == eval_data

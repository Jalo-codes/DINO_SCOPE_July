import pytest
import numpy as np

# We import or skip torch since zoom_cluster_lab imports torch
pytest.importorskip('torch')

from experiments.labs.zoom_cluster_lab import cluster_regions

def test_cluster_regions_overlap_suppression(monkeypatch):
    # Mock HDBSCAN to return predictable labels.
    # Let's say we have a grid of 14x14 patches (196 patches).
    grid_hw = (14, 14)
    n_patches = grid_hw[0] * grid_hw[1]
    
    # Construct mock labels
    labels = np.full(n_patches, -1)
    
    # Cluster 0: patches at (1, 1), (1, 2), (2, 1), (2, 2)
    # Cluster 1: patches at (2, 2), (2, 3), (3, 2), (3, 3) (heavily overlapping)
    # Let's place cluster 0 at row 1..2, col 1..2
    for r in range(1, 3):
        for c in range(1, 3):
            labels[r * 14 + c] = 0
            
    # Cluster 1: row 2..3, col 2..3
    for r in range(2, 4):
        for c in range(2, 4):
            labels[r * 14 + c] = 1

    class MockHDB:
        def __init__(self, **kwargs):
            pass
        def fit_predict(self, X):
            return labels

    monkeypatch.setattr('experiments.labs.zoom_cluster_lab._load_hdbscan', lambda: ('mock', MockHDB))

    # Dummy zp array
    zp = np.zeros((n_patches, 32))
    
    # 1. Run without overlap suppression (overlap_kill_frac = 0.0)
    regions_no_kill = cluster_regions(
        zp, grid_hw,
        min_cluster_size=2,
        min_patches=2,
        overlap_kill_frac=0.0,
        dilate=0
    )
    
    # Both clusters should be present
    assert len(regions_no_kill) == 2

    # 2. Run with overlap suppression (overlap_kill_frac = 0.30)
    regions_with_kill = cluster_regions(
        zp, grid_hw,
        min_cluster_size=2,
        min_patches=2,
        overlap_kill_frac=0.30,
        dilate=0
    )
    
    # The smaller box should be killed due to heavy overlap with the larger one
    assert len(regions_with_kill) == 1

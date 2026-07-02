import pytest
pytest.importorskip('torch')

import argparse
from pathlib import Path
from lab_utils.data.item import Item
from lab_utils.eval.val_sources import collect_val_items_by_source
from lab_utils.data.datasets.registry import REGISTRY

class DummyArgs:
    def __init__(self, sources=None, max_items=None, tgif_eval_per_cell=None, subgroup=None):
        self.sources = sources
        self.max_items = max_items
        self.tgif_eval_per_cell = tgif_eval_per_cell
        self.subgroup = subgroup
        self.tgif2_root = "./dummy_tgif2_root"  # won't exist but we can mock REGISTRY

def test_tgif_eval_per_cell_kwarg_forwarding(monkeypatch):
    called_args = []
    
    # Mock registry for tgif2 to record kwargs
    def dummy_tgif2_build(root, res, **kwargs):
        called_args.append((root, res, kwargs))
        from lab_utils.data.dataset import Dataset
        return Dataset([], res=res, augment=False), Dataset([], res=res, augment=False)
        
    monkeypatch.setitem(REGISTRY, 'tgif2', dummy_tgif2_build)
    
    # Mock Path.exists to return True for dummy_tgif2_root
    monkeypatch.setattr(Path, "exists", lambda self: True)
    
    from lab_utils.data.resolution import Resolution
    res = Resolution(512, 14)
    
    # Run with tgif_eval_per_cell set
    args = DummyArgs(sources=['tgif2'], tgif_eval_per_cell=300)
    collect_val_items_by_source(args, res)
    
    assert len(called_args) == 1
    assert called_args[0][2].get('eval_per_cell') == 300


def test_subgroup_filtration():
    # Construct mock items
    items = [
        Item(image=Path("fake1.png"), authentic=None, mask=Path("mask1.png"), source="tgif2", item_id="1",
             meta={"tgif_subcat": "flux-dev|sp|semantic"}),
        Item(image=Path("fake2.png"), authentic=None, mask=Path("mask2.png"), source="tgif2", item_id="2",
             meta={"tgif_subcat": "sd3|sp|semantic"}),
        Item(image=Path("real.png"), authentic=None, mask=None, source="tgif2", item_id="3",
             meta={"tgif_subcat": "real"}),
    ]
    items[2].is_real = True  # mark third item as real
    
    # Simulate the filtering logic in eval.py:
    subgroups = ["flux-dev|sp|semantic"]
    filtered_items = [
        item for item in items
        if item.is_real or (item.meta.get('generator') or item.meta.get('tgif_subcat')) in subgroups
    ]
    
    assert len(filtered_items) == 2
    assert filtered_items[0].item_id == "1"
    assert filtered_items[1].item_id == "3"

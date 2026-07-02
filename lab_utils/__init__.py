"""lab_utils — shared building blocks for the DINO_SCOPE rebuild.

TORCH-FREE IMPORT BOUNDARY (GAMEPLAN C3)
----------------------------------------
Importing ``lab_utils`` must NOT import torch. The torch-free layers — data
(``item``, ``verify``, ``buckets``), eval records/aggregation, run config — have
to import and run on a CPU dev box and on Colab-CPU. Only modules that genuinely
need tensors (``eval.fetch``, ``model.*``, ``train.*``, tensor-bound decoders)
may import torch, and they do so lazily / on their own import, never from this
top-level ``__init__``.

So: do NOT add eager ``from lab_utils.<x> import ...`` re-exports here for any
module that pulls torch. The previous tree's top-level eagerly imported a
torch-bound eval module, which made the entire package (and its whole test
suite) unimportable without torch — that is the regression this boundary exists
to prevent. Re-exports, if any are added later, must stay torch-free.

This file is intentionally minimal during the rebuild; the public surface is
assembled phase by phase per GAMEPLAN, each addition kept torch-free.
"""

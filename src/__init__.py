"""
Pirate Pain — multivariate time-series classification (AN2DL 2025/26, Challenge 1).

Reference implementation of the submitted solution, extracted from
`notebooks/01_full_pipeline.ipynb` so that the model and pipeline code can be
read, imported and diffed without opening a 400 kB notebook.

The notebooks remain the authoritative record of what was actually executed on
Colab: they carry the training logs and metrics of the submitted run.
"""

__all__ = ["config", "data", "models", "training", "ensemble", "gate", "submission"]
__version__ = "1.0.0"

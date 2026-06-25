"""tfbs -- Transcription-factor binding-site prediction with an embedding-gated
Mixture-of-Experts and a heterogeneous expert zoo.

Importable library for the pipeline. Submodules:
- :mod:`tfbs.constants` -- TF lists, OOD stratification, filesystem path roots
- :mod:`tfbs.data`      -- ChIP-seq loaders, one-hot encoding, dinuc-shuffle negatives
- :mod:`tfbs.models`    -- ConvNet expert, FeatureProbeExpert, MixtureOfExperts gate
- :mod:`tfbs.utils`     -- seeding, early stopping, file discovery

Runnable entry points live under ``experiments/`` (thin CLIs over this library).
"""

__version__ = "0.1.0"

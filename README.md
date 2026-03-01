# SHARP

SHARP is a network- and transcriptomics-based framework for drug prioritization.

## Installation

pip install git+https://github.com/BnayaGross/sharp-aging.git

## Quick example

import sharp
results = sharp.run_sharp(...)


## Extensibility

SHARP is modular and supports arbitrary perturbation transcriptomic datasets.
The pAGE component is dataset-agnostic and can be applied to RNA-seq,
CRISPR perturbations, or LINCS/CMap profiles.

## Citation

If you use SHARP, please cite:
Gross, B., Ehlert, J., Gladyshev, V. N., Loscalzo, J., & Barabási, A. L. (2025). Network-driven discovery of repurposable drugs targeting hallmarks of aging. ArXiv, arXiv-2509.‏

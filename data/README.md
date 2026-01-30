# Data

Datasets used for research, experiments, and analysis.

Recommended structure:
- raw/        → original, immutable datasets
- processed/  → cleaned / transformed data
- interim/    → intermediate pipeline outputs
- results/    → generated outputs, metrics, simulations

Notes:
- Large datasets should NOT be committed to Git.
- Use .gitignore to exclude large or sensitive data.

This folder should primarily contain:
- small sample datasets
- metadata
- schema definitions
- documentation

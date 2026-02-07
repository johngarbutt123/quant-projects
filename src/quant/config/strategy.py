# ===========================
# ======= DATA TIMING =======
# ===========================

TRADING_DAYS = 252

# ===========================
# == LAYER 1: PER-ASSET RISK (04)
# ===========================

# Rolling vol estimation
VOL_WINDOW = 60
MIN_PERIODS = 30

# Per-asset vol targeting
TARGET_VOL_ANN = 0.10  # 10% per-asset target vol
VOL_FLOOR_ANN = 0.02  # floor to avoid leverage blow-ups

# Position limits at the asset level
WEIGHT_CAP = 2.0  # per-asset cap
MAX_GROSS = 3.0  # hard leverage ceiling for layer 1

# ===========================
# == LAYER 2: PORTFOLIO RISK (05)
# ===========================

# Covariance estimation for portfolio risk
COV_WINDOW = 60  # typically same as VOL_WINDOW

# Portfolio vol targeting
PORTFOLIO_TARGET_VOL_ANN = 0.10  # target vol for the whole book

# Safety guardrail on scaling
SCALE_CAP = 3.0  # don't lever up more than 3x via portfolio scaling

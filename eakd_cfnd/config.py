"""Experiment configuration. Values here match the manuscript's Implementation
Details section (main_minor_revison1.tex, subsec:implementation) so that runs
reproduce what's already reported, and extend it with the reviewer-requested
sweeps (sensitivity, calibration) that the paper currently lacks.
"""
from dataclasses import dataclass, field


# --- Fixed hyperparameters reported in the manuscript ---
BACKBONE = "bert-base-uncased"
LEARNING_RATE = 2e-5
BATCH_SIZE = 32
EPOCHS_PER_TASK = 10
DEFAULT_SEEDS = [13, 42, 123, 2024, 31415]

# --- Method-specific defaults, also from the manuscript ---
DEFAULT_UNCERTAINTY_THRESHOLD = 0.7   # theta_uncertainty
DEFAULT_BETA = 1.0                    # beta in alpha(omega(x)) = max(0, 1 - beta * normalize(omega(x)))
DER_BUFFER_SIZE = 200                 # matches Table \ref{tab:performance_revised}, "DER (Buffer=200)"
KD_TEMPERATURE = 2.0                  # tau in L_KD; not stated numerically in the paper, standard default

DATASETS = ("PHEME-Event", "FNN-Poli-Time", "FNN-Gossip-Time")
METHODS = ("FT", "EWC", "LwF", "DER", "LUD", "EAKD-CFND")


@dataclass
class RunConfig:
    dataset: str
    method: str
    seed: int
    uncertainty_threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD
    beta: float = DEFAULT_BETA
    uncertainty_signal: str = "entropy"     # "entropy" | "msp" | "mc_dropout" | "ensemble"
    mc_dropout_passes: int = 20             # only used when uncertainty_signal == "mc_dropout"
    ensemble_size: int = 5                  # only used when uncertainty_signal == "ensemble"
    epochs_per_task: int = EPOCHS_PER_TASK
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    der_buffer_size: int = DER_BUFFER_SIZE
    kd_temperature: float = KD_TEMPERATURE
    log_api_cost: bool = False   # set True for the cost/latency logging phase
    output_dir: str = "runs"

    def run_id(self) -> str:
        return f"{self.dataset}_{self.method}_seed{self.seed}"

# EAKD-CFND experiment pipeline

Reference implementation and full experiment pipeline for **EAKD-CFND**:
class-incremental knowledge distillation with per-instance uncertainty
weighting and external fact-check verification, offered in two
configurations — an exemplar-free default (**EAKD-CFND**) and a
rehearsal-augmented configuration (**EAKD-CFND-R**) for deployments with a
storage budget and severe drift risk. Implements six baselines (FT, EWC, LwF,
DER, LUD) alongside EAKD-CFND/EAKD-CFND-R, and every experiment reported in
the paper: the core benchmark comparison, ablations, calibration validation,
hyperparameter sensitivity, external-verification cost accounting, the
EAKD-CFND-R rehearsal diagnostic, and a task-similarity/isolated-ceiling
analysis explaining why one benchmark split shows no separation between
methods.

## Layout

```
eakd_cfnd/          the library
  config.py           RunConfig — hyperparameters, matches Implementation Details
  data.py             PHEME / FakeNewsNet loading + class-incremental task construction
  uncertainty.py      entropy, MSP, MC Dropout, deep-ensemble uncertainty signals
  verification.py     External Verification module = Algorithm 1, verbatim;
                       supports API-key or service-account OAuth auth
  methods.py          FT, EWC, LwF, DER, LUD, EAKD-CFND, EAKD-CFNDReplay (EAKD-CFND-R)
  train.py            shared CIL training loop (same loop for every method)
  evaluate.py         ACC / F1 / BWT / FWT (Lopez-Paz & Ranzato 2017 definitions)
  calibration.py      Expected Calibration Error + reliability-diagram data
  stats.py            Welch's t-test (paired/seed-matched, matches the manuscript's significance tables)
scripts/             one script per experiment phase, see below
tests/test_core.py   runnable self-check, no GPU/data needed: `python -m tests.test_core`
```

## Setup

```
pip install -r requirements.txt
```

External verification needs one of:
```
export GOOGLE_FACT_CHECK_API_KEY=...              # simple API key (recommended)
export GOOGLE_FACT_CHECK_SERVICE_ACCOUNT=/path/to/service-account.json   # OAuth alternative
```
Only needed for EAKD-CFND/EAKD-CFND-R runs and the cost-logging phase; every
other method runs without it.

## Getting the data

Not bundled (see the manuscript's Data Availability section for the canonical links):

- **PHEME**: https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078
  Extract so you have `data/pheme/<event_name>/{rumours,non-rumours}/<id>/source-tweet/<id>.json`.
  Also works directly against the Kaggle re-upload
  https://www.kaggle.com/datasets/manuelcecerepalazzo/pheme-dataset (nested
  `all-rnr-annotated-threads_1/<event>-all-rnr-threads/...`, plural
  `source-tweets/`) — `data.py` detects either layout and skips the `._*`
  macOS junk files in that re-upload.
- **FakeNewsNet**: https://github.com/KaiDMML/FakeNewsNet
  Their pipeline produces `politifact_fake.csv` / `politifact_real.csv` /
  `gossipcop_fake.csv` / `gossipcop_real.csv`. Put them at
  `data/fakenewsnet/<subset>_<fake|real>.csv`. Kaggle mirror with the same
  filenames: https://www.kaggle.com/datasets/mohamedgreshamahdi/fakenewsnet.
  That raw release has no `publish_date` column (only `id, news_url, title,
  tweet_ids`) — `data.py` falls back to the first `tweet_id`'s Twitter
  snowflake-ID timestamp as a publication-year proxy in that case. If you
  have a source with real `publish_date` (e.g. Kaggle's
  `mdepak/fakenewsnet` content CSVs — PolitiFact/BuzzFeed only, no
  GossipCop), that column is used instead and takes precedence.

`eakd_cfnd/data.py` raises a clear `FileNotFoundError` pointing back here if
the layout doesn't match — check that first if a run fails immediately.

## Running the self-check first

Before spending any GPU time, confirm the logic that matters most is
correct — this runs in seconds, no data or GPU needed:

```
python -m tests.test_core
```

It regression-tests `stats.welch_t_test` against the four t-values already
published in the manuscript's Table `tab:significance`, and checks the
External Verification module's fallback/conflict branching (Algorithm 1)
and the ECE computation.

## Experiment phases

Each phase is a standalone script; GPU-hour estimates are for BERT-base on a
Kaggle T4/P100, 5 seeds unless noted, and assume the pipeline is correct on
the first pass (add ~30-50% for debugging/failed runs in practice).

| # | Script | What | Est. GPU-hrs |
|---|---|---|---|
| 1a-c | `scripts/run_core_table.py --dataset <PHEME-Event\|FNN-Poli-Time\|FNN-Gossip-Time>` | Core 6-method comparison, one of three splits | ~10-25 each |
| 2 | `scripts/run_ablation.py` | Ablation table, 6 configs on FNN-Poli-Time | ~15 |
| 3 | `scripts/run_calibration.py` | ECE for entropy/MSP/MC-Dropout/ensemble | ~20 |
| 4 | `scripts/run_sensitivity.py` | theta_uncertainty and beta sweeps | ~15 |
| 5 | `scripts/run_cost_logging.py` | API cost/latency/failure-rate + train/inference time + memory, all 6 methods | ~5-10 (mostly non-GPU-bound; API-rate-limited) |
| 6 | `scripts/run_hybrid_experiment.py --dataset <name> --variants ...` | EAKD-CFND-R (rehearsal-augmented configuration): plain replay, `+PP` (DER++-style label replay), `+UW` (uncertainty-weighted replay), any combination, per dataset | varies with dataset size; FNN-Gossip-Time is the largest split and can take 30+ GPU-hrs per variant set, so budget for a Kaggle session-limit kill-and-resume cycle (checkpointed, safe to interrupt) |
| 7 | `scripts/analyze_task_drift.py` | TF-IDF task-to-task topical similarity + vocabulary overlap across all three benchmarks (no GPU needed) | minutes |
| 8 | `scripts/run_isolated_task_ceiling.py` | Per-task accuracy ceiling: fresh non-continual FT model trained on each task alone, no forgetting possible by construction | ~1 |
| — | `scripts/compute_review_stats.py` | Paired seed-matched significance tests (t, Cohen's $d_z$, Holm-Bonferroni correction) from already-collected per-seed results, no new training | seconds |

Example invocation for phase 1a:

```
python -m scripts.run_core_table --dataset PHEME-Event --data_root data --out runs/core_pheme.json
```

Every script is resumable across killed/restarted sessions via
`scripts/common.run_with_checkpoint` — pass `--ckpt_dir` (or accept the
default) and a re-run picks up exactly where a prior one stopped, keyed by
`(method, seed)`. Each script writes a JSON file under `runs/`; merge the
`run_core_table` outputs (`scripts/merge_results.py`) plus the ablation/
calibration/sensitivity/cost-logging/hybrid outputs to reproduce the
manuscript's tables from these JSON files.

## Results

Core 6-method comparison (5 seeds, BERT-base), from `runs/core_*.json` via
`scripts/merge_results.py`:

| Method | Dataset | Acc. (%) | F1 (%) | BWT (%) |
|---|---|---|---|---|
| Fine-tuning (FT) | PHEME-Event | 16.1 ± 1.2 | 3.2 ± 0.2 | -85.3 ± 2.0 |
| Fine-tuning (FT) | FNN-Poli-Time | 27.4 ± 2.8 | 7.1 ± 2.5 | -80.5 ± 3.4 |
| Fine-tuning (FT) | FNN-Gossip-Time | 29.0 ± 0.2 | 8.9 ± 0.1 | -84.6 ± 2.0 |
| EWC | PHEME-Event | 16.5 ± 0.5 | 3.3 ± 0.1 | -86.3 ± 1.6 |
| EWC | FNN-Poli-Time | 28.6 ± 3.4 | 7.9 ± 2.9 | -82.1 ± 2.7 |
| EWC | FNN-Gossip-Time | 28.9 ± 0.3 | 8.9 ± 0.1 | -84.8 ± 1.6 |
| LwF | PHEME-Event | 16.5 ± 0.3 | 3.3 ± 0.1 | -87.1 ± 1.0 |
| LwF | FNN-Poli-Time | 31.2 ± 1.0 | 10.1 ± 0.5 | -81.4 ± 2.4 |
| LwF | FNN-Gossip-Time | 28.9 ± 0.5 | 8.9 ± 0.2 | -84.3 ± 1.5 |
| DER (Buffer=200) | PHEME-Event | 84.8 ± 1.5 | 16.5 ± 0.2 | -3.2 ± 2.1 |
| DER (Buffer=200) | FNN-Poli-Time | 30.4 ± 0.9 | 9.9 ± 0.3 | -80.1 ± 4.0 |
| DER (Buffer=200) | FNN-Gossip-Time | 29.4 ± 0.3 | 9.1 ± 0.1 | -85.4 ± 0.5 |
| LUD | PHEME-Event | 16.7 ± 0.3 | 3.3 ± 0.1 | -87.1 ± 0.9 |
| LUD | FNN-Poli-Time | 31.2 ± 0.5 | 10.1 ± 0.3 | -82.2 ± 2.3 |
| LUD | FNN-Gossip-Time | 29.0 ± 0.3 | 8.8 ± 0.2 | -84.8 ± 1.5 |
| **EAKD-CFND (Ours)** | PHEME-Event | **16.8 ± 0.4** | **3.3 ± 0.1** | **-87.2 ± 1.1** |
| **EAKD-CFND (Ours)** | FNN-Poli-Time | **29.8 ± 2.8** | **9.0 ± 2.4** | **-81.6 ± 2.0** |
| **EAKD-CFND (Ours)** | FNN-Gossip-Time | **29.1 ± 0.4** | **8.9 ± 0.1** | **-84.6 ± 1.7** |

PHEME-Event is the one benchmark where the exemplar-free default loses badly
to DER's replay buffer (16.8% vs 84.8% accuracy) — see EAKD-CFND-R (phase 6
above) for the rehearsal-augmented configuration that closes this gap.
Paired significance tests for these comparisons are in
`scripts/compute_review_stats.py`'s output and the manuscript's Table
`tab:significance`.

### EAKD-CFND-R (rehearsal-augmented configuration)

Same protocol, reservoir buffer added on top of EAKD-CFND (`scripts/run_hybrid_experiment.py`, 5 seeds). Rehearsal benefit is real but not uniform across datasets — see `scripts/analyze_task_drift.py` for why FNN-Gossip-Time's tasks are too topically similar to each other for any method to show separation:

| Dataset | Variant | Acc. (%) | BWT (%) |
|---|---|---|---|
| PHEME-Event | plain replay | 85.5 ± 1.1 | -1.8 ± 1.0 |
| PHEME-Event | +PP | 85.0 ± 0.7 | -1.9 ± 0.8 |
| PHEME-Event | +UW | 84.3 ± 1.1 | -3.1 ± 1.5 |
| PHEME-Event | +PP+UW | 84.4 ± 0.5 | -2.6 ± 1.2 |
| PHEME-Event | no-verification | 85.4 ± 1.1 | -2.0 ± 1.0 |
| FNN-Poli-Time | plain replay | 34.9 ± 3.2 | -72.0 ± 8.0 |
| FNN-Poli-Time | +PP | 37.2 ± 2.6 | -56.5 ± 12.7 |
| FNN-Poli-Time | +UW | 32.8 ± 1.6 | -77.8 ± 4.2 |
| FNN-Poli-Time | +PP+UW | 37.8 ± 2.5 | -64.3 ± 8.1 |
| FNN-Poli-Time | no-verification | 34.1 ± 2.5 | -72.3 ± 7.9 |
| FNN-Gossip-Time | +PP | 29.4 ± 0.2 | -85.4 ± 0.9 |
| FNN-Gossip-Time | +UW | 29.2 ± 0.1 | -85.4 ± 0.8 |
| FNN-Gossip-Time | no-verification | 29.1 ± 0.3 | -85.6 ± 0.7 |
| FNN-Gossip-Time | plain replay | *pending* | *pending* |
| FNN-Gossip-Time | +PP+UW | *pending* | *pending* |

On PHEME-Event, every variant closes almost the entire gap to DER's 84.8%
(EAKD-CFND alone gets 16.8%). On FNN-Poli-Time, rehearsal adds a real,
measurable benefit over EAKD-CFND's 29.8%, with `+PP` and `+PP+UW` the
strongest variants (37.2% and 37.8%), not plain replay; paired seed-matched
t-tests put both under p<0.05 against EAKD-CFND and DER
(`scripts/compute_r_variant_stats.py`). On FNN-Gossip-Time, every R-variant
lands within noise of plain EAKD-CFND's 29.1%, consistent with that
dataset's tasks being near-duplicates of each other (TF-IDF task-similarity
0.816 vs. PHEME-Event's 0.199 and FNN-Poli-Time's 0.406) rather than a
weakness in the rehearsal mechanism itself.

## Known scope cuts

- No additional adaptive-KD baseline beyond LUD/DER/EWC/LwF is implemented —
  a new baseline needs to be implemented *correctly* to be worth trusting,
  and that's separate, larger work from instrumenting the existing six.
- `data.py`'s PHEME event list and FakeNewsNet year bins are a reasonable
  default reading of "task sequences ordered chronologically" but aren't
  pinned to a specific prior paper's exact split — document whatever split
  you actually use if you build on this.
- Forward Transfer (FWT) is not currently recorded by the training loop
  (needs a random-initialization baseline logged per task before training on
  it); every table that would include it reports BWT only.

## License

MIT — see `LICENSE`.

## Citation

If you use this code, please cite the accompanying paper (citation details
to be added on publication).

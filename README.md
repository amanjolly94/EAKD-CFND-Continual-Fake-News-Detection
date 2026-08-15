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

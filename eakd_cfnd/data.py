"""Dataset loading + class-incremental (CIL) task construction for PHEME and
FakeNewsNet, matching Table \\ref{tab:datasets} in main_minor_revison1.tex:

    Dataset          Source      #Cls(Init)  #Cls(Incr)  Split Criterion
    PHEME-Event      Twitter     2           1-2 / task  Event chronology
    FNN-Poli-Time    PolitiFact  2           2 / task    Publication year
    FNN-Gossip-Time  GossipCop   2           2 / task    Publication year

Each task introduces a DISJOINT set of classes (class-incremental, not
domain-incremental — see the manuscript's Problem Formulation section):
class = (task_index, veracity). Two instances from different tasks that are
both "fake" are still different classes, because the model is evaluated per
task-conditioned class, matching the CIL protocol the paper describes.

Raw data is NOT bundled here (see the manuscript's Data Availability
section for the source links). This module expects the raw files to already
be downloaded locally; see README.md "Getting the data" for exact steps.
Swap RAW_PHEME_DIR / RAW_FNN_DIR for wherever Kaggle mounts the dataset.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Instance:
    text: str
    label: int          # global class id, unique across the whole task sequence
    task_id: int         # which task this instance belongs to
    veracity: int         # 0 = real, 1 = fake (before task-conditioning)


@dataclass
class Task:
    task_id: int
    classes: list[int]           # global class ids introduced by this task
    train: list[Instance]
    test: list[Instance]


def _assign_class_id(task_id: int, veracity: int) -> int:
    """(task_id, veracity) -> a globally unique class id, class-incremental style."""
    return task_id * 2 + veracity


def load_pheme(raw_dir: str | Path) -> list[Task]:
    """PHEME: one task per event, ordered chronologically. Each event's
    'rumours' folder -> veracity=1 (fake/unverified rumour), 'non-rumours' ->
    veracity=0, matching the manuscript's "PHEME dataset for Rumour Detection
    and Veracity Classification" framing.

    Expected layout (standard PHEME release):
        raw_dir/<event_name>/rumours/<id>/source-tweet/<id>.json
        raw_dir/<event_name>/non-rumours/<id>/source-tweet/<id>.json

    Also accepts the common Kaggle re-upload layout, which nests everything
    under an extra "all-rnr-annotated-threads_*" directory, suffixes each
    event with "-all-rnr-threads", and pluralizes "source-tweet(s)":
        raw_dir/all-rnr-annotated-threads_1/<event_name>-all-rnr-threads/rumours/<id>/source-tweets/<id>.json

    Event chronological order is fixed here to match real event dates; if a
    different PHEME release is used, update EVENT_ORDER accordingly.
    """
    raw_dir = Path(raw_dir)
    EVENT_ORDER = [
        "charliehebdo", "ferguson", "germanwings-crash",
        "ottawashooting", "sydneysiege",
    ]
    tasks: list[Task] = []
    for task_id, event in enumerate(EVENT_ORDER):
        event_dir = _find_pheme_event_dir(raw_dir, event)
        instances: list[Instance] = []
        for veracity, subdir in [(1, "rumours"), (0, "non-rumours")]:
            for thread_dir in sorted((event_dir / subdir).glob("*")):
                source_dir = thread_dir / "source-tweet"
                if not source_dir.exists():
                    source_dir = thread_dir / "source-tweets"
                for f in source_dir.glob("*.json"):
                    if f.name.startswith("._"):
                        continue  # macOS resource-fork junk in some re-uploads
                    tweet = json.loads(f.read_text(encoding="utf-8"))
                    text = tweet.get("text", "")
                    if not text:
                        continue
                    instances.append(Instance(
                        text=text,
                        label=_assign_class_id(task_id, veracity),
                        task_id=task_id,
                        veracity=veracity,
                    ))
        train, test = _split_train_test(instances)
        classes = sorted({inst.label for inst in instances})
        tasks.append(Task(task_id=task_id, classes=classes, train=train, test=test))
    return tasks


def _find_pheme_event_dir(raw_dir: Path, event: str) -> Path:
    candidates = [raw_dir / event, *raw_dir.glob(f"*/{event}-all-rnr-threads"), raw_dir / f"{event}-all-rnr-threads"]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Expected PHEME event directory for {event!r} under {raw_dir} "
        f"(checked {[str(c) for c in candidates]}). "
        "See README.md 'Getting the data' for the download steps."
    )


_TWITTER_EPOCH_MS = 1288834974657  # Twitter snowflake ID epoch (2010-11-04)


def _snowflake_id_to_year(tweet_id: str) -> int | None:
    """First tweet_id in a FakeNewsNet row -> posting year, used as a publish
    date proxy when the CSV has no publish_date column (the standard
    KaiDMML/FakeNewsNet raw release doesn't ship one)."""
    import datetime

    try:
        raw = int(tweet_id)
        if raw < 10**13:
            # Too small to be a real Snowflake ID (those reach 10**13 within
            # ~40 min of the 2010-11-04 epoch); this is a pre-Snowflake
            # sequential tweet ID, and decoding it would coincidentally land
            # near 2010 rather than the tweet's real post date.
            return None
        ms = (raw >> 22) + _TWITTER_EPOCH_MS
        return datetime.datetime.fromtimestamp(ms / 1000, tz=datetime.timezone.utc).year
    except (ValueError, OverflowError, OSError):
        return None


def load_fakenewsnet(raw_dir: str | Path, subset: str, year_bins: list[tuple[int, int]]) -> list[Task]:
    """FakeNewsNet: one task per publication-year bin, ordered chronologically.
    subset: "politifact" or "gossipcop".

    Expected layout (standard FakeNewsNet release, github.com/KaiDMML/FakeNewsNet):
        raw_dir/<subset>_fake.csv   columns: id, news_url, title, tweet_ids[, publish_date]
        raw_dir/<subset>_real.csv   same columns

    The raw KaiDMML release has no publish_date column; when absent, the year
    is taken from the first tweet_id's Twitter-snowflake timestamp instead
    (see _snowflake_id_to_year). If you have a source with real publish_date
    (e.g. Kaggle's mdepak/fakenewsnet content CSVs, PolitiFact-only), that
    column is used directly and takes precedence.

    year_bins: list of (start_year, end_year) inclusive tuples defining each
    task's window, e.g. [(2015, 2016), (2017, 2018), (2019, 2020)] for
    FNN-Poli-Time (3 tasks matching "2 / task" class growth in Table
    \\ref{tab:datasets}: each bin contributes one fake + one real class).
    """
    import csv

    raw_dir = Path(raw_dir)
    fake_csv = raw_dir / f"{subset}_fake.csv"
    real_csv = raw_dir / f"{subset}_real.csv"
    for p in (fake_csv, real_csv):
        if not p.exists():
            raise FileNotFoundError(
                f"Expected FakeNewsNet CSV at {p}. "
                "See README.md 'Getting the data' for the download steps."
            )

    def _read(path: Path, veracity: int) -> list[dict]:
        csv.field_size_limit(10_000_000)  # tweet_ids can list thousands of IDs, past the 131072 default
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["_veracity"] = veracity
        return rows

    rows = _read(fake_csv, 1) + _read(real_csv, 0)

    tasks: list[Task] = []
    for task_id, (start, end) in enumerate(year_bins):
        instances: list[Instance] = []
        for r in rows:
            year = _safe_year(r.get("publish_date") or "")
            if year is None:
                # Raw KaiDMML/FakeNewsNet CSVs (id, news_url, title, tweet_ids)
                # ship no publish_date; fall back to the first sharing tweet's
                # snowflake-ID timestamp as a chronology proxy.
                tweet_ids = (r.get("tweet_ids") or "").split()
                if tweet_ids:
                    year = _snowflake_id_to_year(tweet_ids[0])
            if year is None or not (start <= year <= end):
                continue
            title = r.get("title", "")
            if not title:
                continue
            instances.append(Instance(
                text=title,
                label=_assign_class_id(task_id, r["_veracity"]),
                task_id=task_id,
                veracity=r["_veracity"],
            ))
        train, test = _split_train_test(instances)
        classes = sorted({inst.label for inst in instances})
        tasks.append(Task(task_id=task_id, classes=classes, train=train, test=test))
    return tasks


def _safe_year(date_str: str) -> int | None:
    try:
        return int(date_str[:4])
    except (ValueError, IndexError):
        return None


def _split_train_test(instances: list[Instance], test_fraction: float = 0.2, seed: int = 0) -> tuple[list[Instance], list[Instance]]:
    import random
    rng = random.Random(seed)
    shuffled = instances[:]
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_fraction)) if shuffled else 0
    return shuffled[n_test:], shuffled[:n_test]


def load_dataset(name: str, root: str | Path = "data") -> list[Task]:
    """Dispatch by the dataset names used throughout the manuscript and config.py."""
    root = Path(root)
    if name == "PHEME-Event":
        return load_pheme(root / "pheme")
    if name == "FNN-Poli-Time":
        # Bins re-derived from the actual sharing-tweet year distribution in
        # the downloaded politifact_{fake,real}.csv (Kaggle
        # mohamedgreshamahdi/fakenewsnet): no tweets past 2018, so bins
        # reaching 2019+ would leave the last task's class permanently empty.
        return load_fakenewsnet(root / "fakenewsnet", "politifact",
                                 year_bins=[(2010, 2015), (2016, 2017), (2018, 2018)])
    if name == "FNN-Gossip-Time":
        # Same reasoning; gossipcop's tweet-sharing volume is concentrated in
        # 2017-2018 (crawl window), so bins are split accordingly rather than
        # evenly by calendar year.
        return load_fakenewsnet(root / "fakenewsnet", "gossipcop",
                                 year_bins=[(2010, 2016), (2017, 2017), (2018, 2018)])
    raise ValueError(f"Unknown dataset {name!r}; expected one of the manuscript's three splits.")


def cumulative_seen_classes(tasks: list[Task], up_to_task: int) -> list[int]:
    """All classes introduced by tasks[0..up_to_task] inclusive — this is the
    label space the model must be evaluated against after training on task
    `up_to_task`, per the CIL protocol (Problem Formulation, Section
    \\ref{subsec:problem})."""
    classes: list[int] = []
    for t in tasks[: up_to_task + 1]:
        classes.extend(t.classes)
    return sorted(set(classes))

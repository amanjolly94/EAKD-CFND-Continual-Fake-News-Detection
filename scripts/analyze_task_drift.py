"""Investigate why FNN-Gossip-Time is flat across every method tested
(Section subsec:interpretation): is there measurably less real topical/lexical
drift between its tasks than PHEME-Event's or FNN-Poli-Time's, using the exact
same task construction (eakd_cfnd.data.load_dataset) the experiments train on.

Two real, quotable measures per dataset:
  1. Task-centroid cosine similarity (TF-IDF, fit jointly per dataset) --
     higher = tasks look more alike to a bag-of-words model.
  2. Top-30 content-word Jaccard overlap between consecutive tasks.
Plus a concrete example: named/frequent tokens shared across ALL tasks.

Usage: python -m scripts.analyze_task_drift
"""
from __future__ import annotations

import re
from collections import Counter
from itertools import combinations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from eakd_cfnd.data import load_dataset

STOPWORDS = set("""
a an the of to in on at for and or is are was were be been being this that
these those it its as by with from into over under about after before during
he she they them his her their i you we us our your my me not no do does did
""".split())

TOKEN_RE = re.compile(r"[a-zA-Z']+")


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in TOKEN_RE.findall(text) if w.lower() not in STOPWORDS and len(w) > 2]


def analyze(dataset_name: str, root: str = "data"):
    tasks = load_dataset(dataset_name, root=root)
    task_texts = []
    for t in tasks:
        texts = [inst.text for inst in t.train] + [inst.text for inst in t.test]
        task_texts.append(texts)

    print(f"\n{'='*70}\n{dataset_name}: {len(tasks)} tasks, sizes {[len(x) for x in task_texts]}\n{'='*70}")

    # 1. TF-IDF task-centroid cosine similarity
    all_docs = [" ".join(texts) for texts in task_texts]  # one pseudo-doc per task
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vec.fit_transform(all_docs)
    sim = cosine_similarity(X)
    print("Task-centroid cosine similarity matrix (TF-IDF):")
    for i, row in enumerate(sim):
        print("  T%d: " % i, " ".join(f"{v:.3f}" for v in row))
    off_diag = [sim[i][j] for i, j in combinations(range(len(tasks)), 2)]
    print(f"  mean pairwise similarity (off-diagonal): {sum(off_diag)/len(off_diag):.3f}")
    print(f"  first-vs-last task similarity: {sim[0][-1]:.3f}")

    # 2. Top-30 content-word Jaccard overlap, consecutive tasks
    task_vocabs = []
    for texts in task_texts:
        counts = Counter()
        for txt in texts:
            counts.update(tokenize(txt))
        top30 = set(w for w, _ in counts.most_common(30))
        task_vocabs.append(top30)
    print("Top-30-word Jaccard overlap, consecutive tasks:")
    for i in range(len(task_vocabs) - 1):
        a, b = task_vocabs[i], task_vocabs[i + 1]
        jac = len(a & b) / len(a | b) if (a | b) else 0.0
        shared = sorted(a & b)
        print(f"  T{i} vs T{i+1}: Jaccard={jac:.3f}, shared top words: {shared}")

    # 3. Words appearing in EVERY task's top-30 (a concrete quotable example)
    common_all = set.intersection(*task_vocabs) if task_vocabs else set()
    print(f"Top-30 words common to ALL {len(tasks)} tasks: {sorted(common_all)}")

    return {
        "mean_pairwise_sim": sum(off_diag) / len(off_diag),
        "first_last_sim": sim[0][-1],
        "common_all_tasks_words": sorted(common_all),
    }


if __name__ == "__main__":
    results = {}
    for name in ["PHEME-Event", "FNN-Poli-Time", "FNN-Gossip-Time"]:
        results[name] = analyze(name)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for name, r in results.items():
        print(f"{name}: mean task-pair similarity={r['mean_pairwise_sim']:.3f}, "
              f"first-vs-last={r['first_last_sim']:.3f}, "
              f"words common to all tasks={r['common_all_tasks_words']}")

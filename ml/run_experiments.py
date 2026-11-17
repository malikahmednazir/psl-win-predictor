"""
Dynamic (ball-by-ball) T20 match outcome prediction: reproduction of Goel et al. (2021) on IPL,
application to the Pakistan Super League (PSL), and the proposed novelty.

    python ml/run_experiments.py --exp e1   # paper reproduction on IPL (11 classical + 10 LSTM/GRU, 5 phases)
    python ml/run_experiments.py --exp e2   # 2 paper models + 2 new models, IPL & PSL, season and 80/20 splits
    python ml/run_experiments.py --exp e3   # novelty: chase-state features + IPL->PSL league-aware transfer
    python ml/run_experiments.py --exp all

Target: 1 if the team batting second wins. Positive class = chasing team wins.
Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC (+ Brier score for probability quality).
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, TimeSeriesSplit
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from dl import DL_VARIANTS, SeqClassifier
from features import NOVEL_FEATURES, PAPER_FEATURES, build, load

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "public" / "research-data"
SEED = 42

# The proposed chase-state features (context features Elo / toss / venue are an ablation only)
CHASE_FEATURES = ["runs_needed", "current_run_rate", "required_run_rate", "rr_gap",
                  "resource_index", "runs_last_12", "wkts_last_12"]
CONTEXT_FEATURES = ["elo_diff", "chaser_won_toss", "venue_chase_rate"]

# >>> the two models taken from the paper, and the two new models <<<
PAPER_MODELS = ["SVC", "RandomForestClassifier"]
NEW_MODELS = ["LightGBM", "CatBoost"]
SVC_MAX_ROWS_E3 = 40000  # RBF-SVC is O(n^2); see note in e3()


def classical(name: str):
    return {
        "Logistic Regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "GaussianNB": lambda: GaussianNB(),
        "BernoulliNB": lambda: BernoulliNB(),
        "KNeighborsClassifier": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier()),
        "SVC": lambda: make_pipeline(StandardScaler(), SVC(random_state=SEED)),
        "DecisionTreeClassifier": lambda: DecisionTreeClassifier(random_state=SEED),
        "RandomForestClassifier": lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED),
        "AdaBoostClassifier": lambda: AdaBoostClassifier(random_state=SEED),
        "Bagging Classifier": lambda: BaggingClassifier(n_jobs=-1, random_state=SEED),
        "GradientBoostingClassifier": lambda: GradientBoostingClassifier(random_state=SEED),
        "XGBClassifier": lambda: XGBClassifier(eval_metric="logloss", n_jobs=-1, random_state=SEED),
        "LightGBM": lambda: LGBMClassifier(n_estimators=300, learning_rate=0.05, verbose=-1, random_state=SEED),
        "CatBoost": lambda: CatBoostClassifier(iterations=500, depth=6, verbose=0, random_seed=SEED, allow_writing_files=False,
                                               thread_count=2),
    }[name]()


CLASSICAL_PAPER = ["SVC", "RandomForestClassifier", "XGBClassifier", "GradientBoostingClassifier",
                   "AdaBoostClassifier", "Logistic Regression", "KNeighborsClassifier",
                   "Bagging Classifier", "DecisionTreeClassifier", "GaussianNB", "BernoulliNB"]

# Test scores reported in the paper (Tables 3-7), for the reproduction comparison
PAPER_TEST = {
    "1-20": {"SVC": 72.74, "RandomForestClassifier": 72.27, "XGBClassifier": 71.67,
             "GradientBoostingClassifier": 71.61, "AdaBoostClassifier": 71.49, "Logistic Regression": 71.22,
             "KNeighborsClassifier": 70.22, "Bagging Classifier": 69.20, "DecisionTreeClassifier": 68.71,
             "GaussianNB": 62.66, "BernoulliNB": 57.61, "o2o MF bidirec GRU": 72.30, "o2o MF Stacked GRU": 72.23,
             "o2o MF vanilla GRU": 72.11, "o2o MF bidirec lstm": 72.28, "o2o MF Stacked lstm": 72.20,
             "o2o MF vanilla lstm": 72.37, "m2o MF Stacked GRU": 72.25, "m2o MF vanilla GRU": 72.28,
             "m2o MF Stacked LSTM": 71.50, "m2o MF vanilla LSTM": 71.80},
    "1-5": {"Logistic Regression": 66.70, "RandomForestClassifier": 66.54, "SVC": 68.68, "XGBClassifier": 67.53,
            "m2o MF Stacked GRU": 68.59, "m2o MF Stacked LSTM": 68.65, "m2o MF vanilla GRU": 69.25,
            "m2o MF vanilla LSTM": 67.66, "o2o MF Stacked GRU": 68.52, "o2o MF Stacked lstm": 68.57,
            "o2o MF bidirec GRU": 68.19, "o2o MF bidirec lstm": 68.13, "o2o MF vanilla GRU": 68.68,
            "o2o MF vanilla lstm": 68.57},
    "6-10": {"Logistic Regression": 69.84, "RandomForestClassifier": 70.49, "SVC": 72.36, "XGBClassifier": 70.49,
             "m2o MF Stacked GRU": 71.51, "m2o MF Stacked LSTM": 70.96, "m2o MF vanilla GRU": 71.34,
             "m2o MF vanilla LSTM": 71.45, "o2o MF Stacked GRU": 72.03, "o2o MF Stacked lstm": 72.09,
             "o2o MF bidirec GRU": 71.92, "o2o MF bidirec lstm": 71.98, "o2o MF vanilla GRU": 71.87,
             "o2o MF vanilla lstm": 71.65},
    "11-15": {"Logistic Regression": 78.79, "RandomForestClassifier": 78.57, "SVC": 79.22, "XGBClassifier": 76.49,
              "m2o MF Stacked GRU": 78.76, "m2o MF Stacked LSTM": 78.65, "m2o MF vanilla GRU": 78.82,
              "m2o MF vanilla LSTM": 79.04, "o2o MF Stacked GRU": 78.90, "o2o MF Stacked lstm": 78.90,
              "o2o MF bidirec GRU": 79.22, "o2o MF bidirec lstm": 79.28, "o2o MF vanilla GRU": 79.28,
              "o2o MF vanilla lstm": 78.73},
    "16-20": {"Logistic Regression": 77.93, "RandomForestClassifier": 75.18, "SVC": 77.26, "XGBClassifier": 75.12,
              "m2o MF Stacked GRU": 75.62, "m2o MF Stacked LSTM": 73.94, "m2o MF vanilla GRU": 75.02,
              "m2o MF vanilla LSTM": 76.43, "o2o MF Stacked GRU": 75.65, "o2o MF Stacked lstm": 75.72,
              "o2o MF bidirec GRU": 77.13, "o2o MF bidirec lstm": 75.59, "o2o MF vanilla GRU": 73.44,
              "o2o MF vanilla lstm": 73.98},
}

# Paper's Models 1-5: (train overs, test overs)
PHASES = {"1-20": (20, (1, 20)), "1-5": (5, (1, 5)), "6-10": (10, (6, 10)),
          "11-15": (15, (11, 15)), "16-20": (20, (16, 20))}


# ------------------------------------------------------------------------------------------
def scores(y, p):
    pred = (p >= 0.5).astype(int)
    return {"Accuracy": accuracy_score(y, pred) * 100, "Precision": precision_score(y, pred, zero_division=0) * 100,
            "Recall": recall_score(y, pred, zero_division=0) * 100, "F1": f1_score(y, pred, zero_division=0) * 100,
            "ROC_AUC": roc_auc_score(y, p) * 100 if len(np.unique(y)) > 1 else np.nan,
            "Brier": brier_score_loss(y, np.clip(p, 0, 1))}


def fit_predict(name, tr, te, feats, svc_cap=None):
    """Returns probability (or a 0-1 score for SVC) of a chase win for every test row."""
    if name in DL_VARIANTS:
        m = SeqClassifier(**DL_VARIANTS[name]).fit(tr[feats].values, tr["result"].values, tr["match_id"].values)
        return m.predict_proba(te[feats].values, te["match_id"].values)[:, 1]
    if name == "SVC" and svc_cap and len(tr) > svc_cap:
        step = int(np.ceil(len(tr) / svc_cap))
        tr = tr.iloc[::step]
    m = classical(name).fit(tr[feats], tr["result"])
    if name == "SVC":  # decision function -> logistic squashing for AUC / Brier
        return 1 / (1 + np.exp(-m.decision_function(te[feats])))
    return m.predict_proba(te[feats])[:, 1]


def chrono(df, matches):
    order = matches.set_index("id")["date"]
    df = df.assign(_d=df["match_id"].map(order)).sort_values(["_d", "match_id", "ball"])
    return df.drop(columns="_d").reset_index(drop=True)


def ts_cv(name, tr, feats, folds=5):
    """KFold TimeSeriesSplit over whole matches in date order (as in the paper)."""
    mids = tr["match_id"].drop_duplicates().to_numpy()
    accs = []
    for a, b in TimeSeriesSplit(n_splits=folds).split(mids):
        f_tr = tr[tr["match_id"].isin(mids[a])]
        f_te = tr[tr["match_id"].isin(mids[b])]
        p = fit_predict(name, f_tr, f_te, feats)
        accs.append(accuracy_score(f_te["result"], p >= 0.5) * 100)
    return float(np.mean(accs)), float(np.std(accs))


def load_league(lg):
    m, d = load(DATA / lg / "matches.csv", DATA / lg / "deliveries.csv")
    cache = DATA / f"{lg}_features.csv"
    df = pd.read_csv(cache) if cache.exists() else build(m, d)
    if not cache.exists():
        df.to_csv(cache, index=False)
    return chrono(df, m), m


def save(df, name):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)
    print(f"  -> {name} ({len(df)} rows)")


# ------------------------------------------------------------------------------------------
def e1(family: str = "all"):
    """Reproduction of the paper on IPL: train 2008-2018, test 2019, Models 1-5.
    Models 1 (overs 1-20) and 5 (overs 16-20) share the same training data, so each model is
    trained once per training range and evaluated on every phase that uses it."""
    df, _ = load_league("ipl")
    names = {"classical": CLASSICAL_PAPER, "dl": list(DL_VARIANTS),
             "all": CLASSICAL_PAPER + list(DL_VARIANTS)}[family]
    by_train = {}
    for phase, (train_to, rng) in PHASES.items():
        by_train.setdefault(train_to, []).append((phase, rng))
    rows = []
    for train_to, phases in by_train.items():
        tr = df[(df.season <= 2018) & (df.over <= train_to)]
        te_all = df[(df.season == 2019) & (df.over <= train_to)]
        for name in names:
            t0 = time.time()
            cv_m, cv_s = ts_cv(name, tr, PAPER_FEATURES)
            p_all = fit_predict(name, tr, te_all, PAPER_FEATURES)
            for phase, (a, b) in phases:
                sel = te_all["over"].between(a, b).to_numpy()
                te = te_all[sel]
                s = scores(te["result"].to_numpy(), p_all[sel])
                rows.append({"Phase": phase, "Model": name,
                             "Family": "Deep learning" if name in DL_VARIANTS else "Classical",
                             "Train_rows": len(tr), "Test_rows": len(te), "CV_mean": cv_m, "CV_std": cv_s, **s,
                             "Paper_test_accuracy": PAPER_TEST.get(phase, {}).get(name, np.nan),
                             "Seconds": round(time.time() - t0, 1)})
                print(f"[e1 {phase:5s}] {name:28s} CV={cv_m:5.2f}+-{cv_s:4.2f} test={s['Accuracy']:5.2f} "
                      f"paper={rows[-1]['Paper_test_accuracy']}", flush=True)
            save(pd.DataFrame(rows), f"e1_part_{family}.csv")
    # merge the parts that exist into the final table
    parts = [OUT / f"e1_part_{f}.csv" for f in ("classical", "dl", "all")]
    merged = pd.concat([pd.read_csv(p) for p in parts if p.exists()], ignore_index=True)
    merged = merged.drop_duplicates(["Phase", "Model"], keep="last")
    order = {p: i for i, p in enumerate(PHASES)}
    merged = merged.sort_values(["Phase", "Family", "Model"], key=lambda c: c.map(order) if c.name == "Phase" else c)
    save(merged, "e1_paper_reproduction_ipl.csv")


def e2():
    """Assignment table: 2 paper models + 2 new models, IPL and PSL, season split and 80/20 split."""
    rows = []
    for lg, test_season in [("ipl", 2019), ("psl", 2026)]:
        df, _ = load_league(lg)
        mids = df["match_id"].drop_duplicates().to_numpy()
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(mids, groups=mids))
        splits = {
            f"Season split (test {test_season})": (df[df.season < test_season], df[df.season == test_season]),
            "Random 80/20 (by match)": (df[df.match_id.isin(mids[a])], df[df.match_id.isin(mids[b])]),
        }
        for split, (tr, te) in splits.items():
            for name in PAPER_MODELS + NEW_MODELS:
                t0 = time.time()
                p = fit_predict(name, tr, te, PAPER_FEATURES)
                s = scores(te["result"].to_numpy(), p)
                phase = {}
                for ph, (_, (lo, hi)) in list(PHASES.items())[1:]:
                    sel = te["over"].between(lo, hi).to_numpy()
                    phase[f"Acc_overs_{ph}"] = accuracy_score(te["result"][sel], p[sel] >= 0.5) * 100
                rows.append({"League": lg.upper(), "Split": split,
                             "Role": "Paper model" if name in PAPER_MODELS else "New model",
                             "Model": name, "Train_matches": tr.match_id.nunique(), "Test_matches": te.match_id.nunique(),
                             "Train_rows": len(tr), "Test_rows": len(te), **s, **phase,
                             "Seconds": round(time.time() - t0, 1)})
                print(f"[e2 {lg} {split:28s}] {name:24s} acc={s['Accuracy']:5.2f} auc={s['ROC_AUC']:5.2f}", flush=True)
    save(pd.DataFrame(rows), "e2_four_models.csv")


def e3():
    """Novelty. Problem: PSL has far fewer matches than IPL, so ball-by-ball predictors trained on
    PSL alone are inaccurate and over-confident. Proposed: chase-state features + league-aware
    transfer (IPL + PSL training data with a league indicator). Evaluated on six PSL seasons
    (2021-2026), each predicted using only earlier seasons.

    Note: RBF-SVC costs O(n^2); in this experiment its training set is capped at SVC_MAX_ROWS_E3 rows by
    uniform ball-stride subsampling (applied identically to every setting, so pooled settings with more
    data are, if anything, disadvantaged)."""
    ipl, _ = load_league("ipl")
    psl, _ = load_league("psl")
    ipl["league"], psl["league"] = 0, 1
    F_PAPER = PAPER_FEATURES
    F_CHASE = PAPER_FEATURES + CHASE_FEATURES
    settings = {
        "S1 Paper method: PSL data, paper features": ("psl", F_PAPER),
        "S2 PSL data + chase features": ("psl", F_CHASE),
        "S3 IPL+PSL transfer, paper features": ("both", F_PAPER),
        "S4 Proposed: IPL+PSL transfer + chase + league flag": ("both", F_CHASE + ["league"]),
        "A1 Ablation: S4 + context (Elo, toss, venue)": ("both", F_CHASE + CONTEXT_FEATURES + ["league"]),
        "A2 IPL only (zero-shot), chase features": ("ipl", F_CHASE),
    }
    rows = []
    for season in range(2021, 2027):
        p_tr, te = psl[psl.season < season], psl[psl.season == season]
        for sname, (src, feats) in settings.items():
            tr = {"psl": p_tr, "ipl": ipl, "both": pd.concat([ipl, p_tr], ignore_index=True)}[src]
            for name in PAPER_MODELS + NEW_MODELS:
                t0 = time.time()
                p = fit_predict(name, tr, te, feats, svc_cap=SVC_MAX_ROWS_E3)
                s = scores(te["result"].to_numpy(), p)
                rows.append({"Test_season": season, "Setting": sname, "Model": name,
                             "Role": "Paper model" if name in PAPER_MODELS else "New model",
                             "Train_rows": len(tr), "Test_matches": te.match_id.nunique(), **s,
                             "Seconds": round(time.time() - t0, 1)})
                print(f"[e3 {season}] {sname:52s} {name:24s} acc={s['Accuracy']:5.2f} "
                      f"auc={s['ROC_AUC']:5.2f} brier={s['Brier']:.3f}", flush=True)
        save(pd.DataFrame(rows), "e3_novelty_psl_rolling.csv")

    # IPL side: does the chase-state feature set also help on the paper's own IPL 2019 test?
    rows = []
    tr, te = ipl[ipl.season <= 2018], ipl[ipl.season == 2019]
    for fs_name, feats in [("Paper features", F_PAPER), ("Paper + chase features", F_CHASE),
                           ("Paper + chase + context", F_CHASE + CONTEXT_FEATURES)]:
        for name in PAPER_MODELS + NEW_MODELS:
            p = fit_predict(name, tr, te, feats, svc_cap=SVC_MAX_ROWS_E3)
            rows.append({"Feature_set": fs_name, "Model": name, **scores(te["result"].to_numpy(), p)})
            print(f"[e3 ipl] {fs_name:26s} {name:24s} acc={rows[-1]['Accuracy']:5.2f}", flush=True)
    save(pd.DataFrame(rows), "e3_novelty_ipl2019_features.csv")


def dataset_summary():
    rows = []
    for lg in ["ipl", "psl"]:
        df, m = load_league(lg)
        g = df.groupby("match_id").first()
        for season, gs in g.groupby("season"):
            rows.append({"League": lg.upper(), "Season": season, "Matches": len(gs),
                         "Second_innings_balls": int((df.season == season).sum()),
                         "Chasing_win_rate": gs["result"].mean() * 100})
    save(pd.DataFrame(rows), "dataset_summary.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", choices=["e1", "e2", "e3", "summary", "all"])
    ap.add_argument("--family", default="all", choices=["classical", "dl", "all"],
                    help="e1 only: run the classical or the deep-learning models (to run in parallel)")
    a = ap.parse_args()
    t = time.time()
    if a.exp in ("summary", "all"):
        dataset_summary()
    if a.exp in ("e2", "all"):
        e2()
    if a.exp in ("e3", "all"):
        e3()
    if a.exp in ("e1", "all"):
        e1(a.family)
    (OUT / "run_info.json").write_text(json.dumps({"last_run": a.exp, "seconds": round(time.time() - t)}, indent=2))

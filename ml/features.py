"""
Ball-by-ball second-innings features of Goel et al. (2021), "Dynamic cricket match outcome
prediction", Journal of Sports Analytics 7, 185-196 (Table 1), plus the novel chase-context
features proposed in this work.

Input : matches.csv + deliveries.csv (Kaggle IPL 2008-2019 schema; PSL converted to it)
Output: one row per delivery of the second innings, target `result` = 1 if the chasing team won.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---- Table 1 of the paper (30 attributes, match_id is only an identifier) ------------------
BMAN = [f"bman{i}" for i in range(1, 12)]
BOWL = [f"bowl{i}" for i in range(1, 11)]
PAPER_FEATURES = (["ball", "over", "balls_played"] + BMAN + BOWL +
                  ["extra_runs", "batsman_runs", "remaining_balls", "wickets_left",
                   "present_score", "target"])

# ---- Novel features (this work) ------------------------------------------------------------
NOVEL_FEATURES = [
    "runs_needed", "current_run_rate", "required_run_rate", "rr_gap",
    "resource_index",            # wickets_left x remaining_balls (DLS-style resources)
    "runs_last_12", "wkts_last_12",  # momentum over the last two overs
    "elo_diff",                  # pre-match team strength: chasing minus defending Elo
    "chaser_won_toss",
    "venue_chase_rate",          # historical chasing success at the venue (prior matches only)
]


def load(matches_csv: str, deliveries_csv: str):
    m = pd.read_csv(matches_csv)
    d = pd.read_csv(deliveries_csv)
    iso = pd.to_datetime(m["date"], format="%Y-%m-%d", errors="coerce")
    dmy = pd.to_datetime(m["date"], format="%d/%m/%y", errors="coerce")  # IPL 2018-2019 rows
    m["date"] = iso.fillna(dmy)
    assert m["date"].notna().all(), "unparsed dates"
    return m, d


def eligible_matches(m: pd.DataFrame) -> pd.DataFrame:
    """Matches with a decided result (normal or tie decided by super over), as in the paper."""
    return m[m["result"].isin(["normal", "tie"]) & m["winner"].notna()].copy()


def _elo_and_venue(m: pd.DataFrame, chasing: dict, defending: dict, k: float = 20.0):
    """Pre-match Elo difference and venue chase rate, using only matches played BEFORE each match."""
    elo: dict = {}
    venue_stats: dict = {}
    out_elo, out_venue = {}, {}
    for _, r in m.sort_values(["date", "id"]).iterrows():
        mid = r["id"]
        if mid not in chasing:
            continue
        c, f = chasing[mid], defending[mid]
        ec, ef = elo.get(c, 1500.0), elo.get(f, 1500.0)
        out_elo[mid] = ec - ef
        w, n = venue_stats.get(r["venue"], (0, 0))
        out_venue[mid] = (w + 0.5 * 5) / (n + 5)  # smoothed towards 50 %
        # update after the match
        s = 1.0 if r["winner"] == c else 0.0
        exp = 1 / (1 + 10 ** ((ef - ec) / 400))
        elo[c] = ec + k * (s - exp)
        elo[f] = ef - k * (s - exp)
        venue_stats[r["venue"]] = (w + s, n + 1)
    return out_elo, out_venue


def build(m: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    m = eligible_matches(m)
    d = d[(d["is_super_over"] == 0) & d["match_id"].isin(m["id"])]
    first = d[d["inning"] == 1].groupby("match_id")["total_runs"].sum()
    second = d[d["inning"] == 2]
    chasing = second.groupby("match_id")["batting_team"].first().to_dict()
    defending = second.groupby("match_id")["bowling_team"].first().to_dict()
    elo, venue = _elo_and_venue(m, chasing, defending)
    meta = m.set_index("id")

    rows = []
    for mid, g in second.groupby("match_id", sort=False):
        if mid not in first.index:
            continue
        info = meta.loc[mid]
        target = int(first[mid])
        chaser = chasing[mid]
        y = int(info["winner"] == chaser)
        bat_order, bowl_order = [], []
        bat_runs, bowl_runs = {}, {}
        score = wkts = legal = 0
        hist_runs, hist_wkts = [], []
        for i, b in enumerate(g.itertuples(index=False), start=1):
            for p in (b.batsman, b.non_striker):
                if p not in bat_runs:
                    bat_runs[p] = 0
                    bat_order.append(p)
            if b.bowler not in bowl_runs:
                bowl_runs[b.bowler] = 0
                bowl_order.append(b.bowler)
            bat_runs[b.batsman] += b.batsman_runs
            # runs charged to the bowler (byes / leg-byes are not)
            bowl_runs[b.bowler] += b.batsman_runs + b.wide_runs + b.noball_runs
            score += b.total_runs
            is_wkt = isinstance(b.player_dismissed, str) and b.player_dismissed != ""
            wkts += int(is_wkt)
            if b.wide_runs == 0 and b.noball_runs == 0:
                legal += 1
            hist_runs.append(b.total_runs)
            hist_wkts.append(int(is_wkt))
            remaining = max(120 - legal, 0)
            row = {"match_id": mid, "season": int(info["season"]), "result": y,
                   "ball": i, "over": int(b.over), "balls_played": legal,
                   "extra_runs": b.extra_runs, "batsman_runs": b.batsman_runs,
                   "remaining_balls": remaining, "wickets_left": 10 - wkts,
                   "present_score": score, "target": target}
            for j in range(11):
                row[BMAN[j]] = bat_runs[bat_order[j]] if j < len(bat_order) else 0
            for j in range(10):
                if j < 9:
                    row[BOWL[j]] = bowl_runs[bowl_order[j]] if j < len(bowl_order) else 0
                else:  # 10th slot also holds any further bowlers
                    row[BOWL[j]] = sum(bowl_runs[p] for p in bowl_order[9:])
            need = target + 1 - score
            overs_done = legal / 6
            overs_left = remaining / 6
            row["runs_needed"] = need
            row["current_run_rate"] = score / overs_done if overs_done > 0 else 0.0
            row["required_run_rate"] = min(need / overs_left, 36.0) if overs_left > 0 else (36.0 if need > 0 else 0.0)
            row["rr_gap"] = row["required_run_rate"] - row["current_run_rate"]
            row["resource_index"] = (10 - wkts) * remaining
            row["runs_last_12"] = sum(hist_runs[-12:])
            row["wkts_last_12"] = sum(hist_wkts[-12:])
            row["elo_diff"] = elo.get(mid, 0.0)
            row["chaser_won_toss"] = int(info["toss_winner"] == chaser)
            row["venue_chase_rate"] = venue.get(mid, 0.5)
            rows.append(row)
    df = pd.DataFrame(rows)
    return df

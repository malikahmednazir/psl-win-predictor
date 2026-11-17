"""Convert Cricsheet PSL csv2 files into the same matches.csv / deliveries.csv schema as the
Kaggle IPL 2008-2019 dataset used by Goel et al. (2021)."""
import csv, glob, os, sys
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
mrows, drows = [], []
for f in sorted(glob.glob(os.path.join(src, "*_info.csv"))):
    mid = int(os.path.basename(f).split("_")[0])
    info = {"team": [], "umpire": []}
    with open(f, newline="", encoding="utf-8") as fh:
        for r in csv.reader(fh):
            if len(r) < 3 or r[0] != "info":
                continue
            k, v = r[1], r[2:]
            if k in ("team", "umpire"):
                info[k].append(v[0])
            elif k not in info:
                info[k] = v
    g = lambda k, i=0: (info.get(k) or [None])[i] if info.get(k) and len(info.get(k)) > i else None
    winner = g("winner")
    outcome = g("outcome")  # 'tie' or 'no result'
    eliminator = g("eliminator")
    result = "normal"
    if outcome == "tie":
        result, winner = "tie", eliminator or winner
    elif outcome == "no result":
        result = "no result"
    date = g("date").replace("/", "-")
    mrows.append({
        "id": mid, "season": int(date[:4]), "city": g("city"), "date": date,
        "team1": info["team"][0], "team2": info["team"][1],
        "toss_winner": g("toss_winner"), "toss_decision": g("toss_decision"),
        "result": result, "dl_applied": int(g("method") in ("D/L", "DLS")),
        "winner": winner, "win_by_runs": int(g("winner_runs") or 0),
        "win_by_wickets": int(g("winner_wickets") or 0),
        "player_of_match": g("player_of_match"), "venue": g("venue"),
        "umpire1": info["umpire"][0] if info["umpire"] else None,
        "umpire2": info["umpire"][1] if len(info["umpire"]) > 1 else None, "umpire3": None,
    })
    d = pd.read_csv(os.path.join(src, f"{mid}.csv"))
    for _, b in d.iterrows():
        inn = int(b.innings)
        over = int(str(b.ball).split(".")[0]) + 1
        ball = int(str(b.ball).split(".")[1])
        num = lambda c: int(b[c]) if pd.notna(b[c]) else 0
        extras = num("extras")
        drows.append({
            "match_id": mid, "inning": inn if inn <= 2 else inn, "batting_team": b.batting_team,
            "bowling_team": b.bowling_team, "over": over, "ball": ball,
            "batsman": b.striker, "non_striker": b.non_striker, "bowler": b.bowler,
            "is_super_over": int(inn > 2), "wide_runs": num("wides"), "bye_runs": num("byes"),
            "legbye_runs": num("legbyes"), "noball_runs": num("noballs"), "penalty_runs": num("penalty"),
            "batsman_runs": num("runs_off_bat"), "extra_runs": extras,
            "total_runs": num("runs_off_bat") + extras,
            "player_dismissed": b.player_dismissed if pd.notna(b.player_dismissed) else None,
            "dismissal_kind": b.wicket_type if pd.notna(b.wicket_type) else None,
            "fielder": b.fielder_1 if pd.notna(b.fielder_1) else None,
        })
M = pd.DataFrame(mrows).sort_values(["date", "id"])
D = pd.DataFrame(drows)
M.to_csv(os.path.join(out, "matches.csv"), index=False)
D.to_csv(os.path.join(out, "deliveries.csv"), index=False)
print(M.shape, D.shape)
print(M.groupby("season").size().to_dict())
print(M.result.value_counts().to_dict(), "DL", M.dl_applied.sum())

# Dynamic T20 Match Outcome Prediction: from the IPL to the Pakistan Super League

**Author:** Sundas Maria Hayat (Roll No. 232015)

This repository reproduces **Goel et al. (2021), "Dynamic cricket match outcome prediction",
*Journal of Sports Analytics* 7, 185–196** on its original IPL dataset. It then applies the same
method to the **Pakistan Super League (PSL)** and proposes a novelty for data-scarce leagues.
An Angular dashboard publishes all results on GitHub Pages.

**Task.** After every ball of the second innings, predict whether the chasing team will win.

---

## Datasets

| League | Source | Matches used | Notes |
|---|---|---|---|
| IPL 2008–2019 | Kaggle IPL dataset (`matches.csv` + `deliveries.csv`), https://www.kaggle.com/nowke9/ipldata | 752 decided matches | The paper's dataset. Our 2019 test set has **59 matches / 6,960 balls, exactly as in the paper**. Training has 693 matches / 79,253 balls (paper: 692 / 79,129); the paper does not say which single match it dropped. |
| PSL 2016–2026 | Cricsheet, https://cricsheet.org/downloads/psl_csv2.zip (Open Data Commons Attribution License) | 354 decided matches | Converted by `ml/convert_psl.py` into exactly the same columns as the IPL files. |

"Decided" = normal result or tie decided by a super over. No-result matches are removed, as in the paper.

## Method of the paper (reproduced in `ml/features.py`, `ml/dl.py`)

* **Features (Table 1 of the paper, 30 attributes):** ball, over, balls played, runs of batsman 1–11 and
  bowler 1–10 (cumulative, in batting / bowling order), extra runs and batsman runs on the ball,
  remaining balls, wickets left, present score and target.
* **Models:** 11 classical models (Logistic Regression, Gaussian / Bernoulli NB, KNN, SVC, Decision Tree,
  Random Forest, AdaBoost, Bagging, Gradient Boosting, XGBoost) and 10 LSTM / GRU variants
  (vanilla, stacked, bidirectional; one-to-one and many-to-one).
* **Protocol:** train on seasons 2008–2018, test on 2019. "Mean train score" = KFold TimeSeriesSplit over
  matches in date order. The paper's Models 1–5 cover overs 1–20, 1–5, 6–10, 11–15 and 16–20.

## Repository structure

```
data/ipl/                  matches.csv, deliveries.csv (paper's IPL dataset)
data/psl/                  matches.csv, deliveries.csv (PSL, same format)
data/*_features.csv        ball-by-ball feature tables (rebuilt automatically if deleted)
ml/convert_psl.py          Cricsheet PSL -> IPL schema
ml/features.py             paper features (Table 1) + proposed chase-state features
ml/dl.py                   LSTM / GRU models of the paper (PyTorch)
ml/run_experiments.py      E1 reproduction, E2 four models, E3 novelty
ml/make_figures.py         figures for the paper (public/research-data/figures)
public/research-data/      all results (CSV) read by the dashboard
src/app/                   Angular dashboard
.github/workflows/         builds the dashboard and deploys it to GitHub Pages
```

## How to run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS / Linux: source .venv/bin/activate
pip install -r ml/requirements.txt

python ml/run_experiments.py --exp e2    # 4 models, IPL and PSL (~10 min)
python ml/run_experiments.py --exp e3    # novelty (~35 min)
python ml/run_experiments.py --exp e1 --family classical   # paper reproduction, classical (~30 min)
python ml/run_experiments.py --exp e1 --family dl          # paper reproduction, LSTM/GRU (~10 min)
python ml/make_figures.py

npm ci
npm start                                # dashboard at http://localhost:4200
```

To rebuild the PSL files from a fresh Cricsheet download:
`python ml/convert_psl.py <folder with unzipped psl_csv2> data/psl`

---

## E1. Reproduction of the paper on IPL (test season 2019, overs 1–20)

| Model | Mean train score (CV) | Our test acc. | Paper test acc. | Precision | Recall | F1 | ROC_AUC |
|---|---|---|---|---|---|---|---|
| AdaBoostClassifier | 69.91 (±3.78) | 71.97 | 71.49 | 80.19 | 70.67 | 75.13 | 82.86 |
| Bagging Classifier | 68.50 (±3.29) | 65.03 | 69.20 | 72.23 | 67.63 | 69.85 | 72.51 |
| BernoulliNB | 62.93 (±2.45) | 64.38 | 57.61 | 68.63 | 74.70 | 71.54 | 63.94 |
| DecisionTreeClassifier | 65.42 (±2.54) | 62.13 | 68.71 | 71.94 | 60.31 | 65.61 | 62.72 |
| GaussianNB | 59.95 (±1.58) | 64.25 | 62.66 | 63.10 | 97.15 | 76.51 | 79.63 |
| GradientBoostingClassifier | 71.65 (±5.08) | 72.20 | 71.61 | 79.98 | 71.49 | 75.50 | 82.49 |
| KNeighborsClassifier | 65.67 (±1.67) | 68.53 | 70.22 | 75.26 | 70.74 | 72.93 | 72.43 |
| Logistic Regression | 76.00 (±3.79) | 72.30 | 71.22 | 78.63 | 73.84 | 76.16 | 83.76 |
| RandomForestClassifier | 72.11 (±2.80) | 72.99 | 72.27 | 81.88 | 70.53 | 75.78 | 82.09 |
| SVC | 73.09 (±2.88) | 75.01 | 72.74 | 80.83 | 76.43 | 78.57 | 83.95 |
| XGBClassifier | 69.18 (±4.21) | 68.23 | 71.67 | 75.80 | 69.02 | 72.25 | 77.20 |
| m2o MF Stacked GRU | 71.35 (±3.06) | 74.21 | 72.25 | 81.00 | 74.41 | 77.57 | 82.85 |
| m2o MF Stacked LSTM | 70.96 (±2.66) | 73.62 | 71.50 | 80.36 | 74.08 | 77.09 | 81.62 |
| m2o MF vanilla GRU | 72.03 (±3.50) | 74.27 | 72.28 | 82.98 | 71.77 | 76.97 | 84.75 |
| m2o MF vanilla LSTM | 70.48 (±2.33) | 74.41 | 71.80 | 81.64 | 73.91 | 77.58 | 83.13 |
| o2o MF Stacked GRU | 73.50 (±4.22) | 73.56 | 72.23 | 80.10 | 74.34 | 77.11 | 83.10 |
| o2o MF Stacked lstm | 72.76 (±3.44) | 73.38 | 72.20 | 79.56 | 74.77 | 77.09 | 82.70 |
| o2o MF bidirec GRU | 74.38 (±3.33) | 76.42 | 72.30 | 82.17 | 77.46 | 79.74 | 86.27 |
| o2o MF bidirec lstm | 73.92 (±3.11) | 75.49 | 72.28 | 80.53 | 77.94 | 79.21 | 85.89 |
| o2o MF vanilla GRU | 74.53 (±3.38) | 75.17 | 72.11 | 80.99 | 76.52 | 78.69 | 86.01 |
| o2o MF vanilla lstm | 74.32 (±3.66) | 75.45 | 72.37 | 81.48 | 76.38 | 78.85 | 85.65 |

For the full innings our accuracy is close to the paper's (SVC 75.0 % vs 72.7 %, Random Forest 73.0 % vs 72.3 %,
LSTM / GRU 73.4–76.4 % vs 71.5–72.4 %). For the first five overs we are about 5 points lower than the paper,
and for overs 16–20 about 7 points higher; the paper's recall of 100 % in that phase suggests a different
decision threshold. Small differences are expected: the paper does not report its hyper-parameters,
network sizes or random seeds.

Accuracy by phase (our test accuracy / the paper's):

| Model | Overs 1-5 (ours / paper) | Overs 6-10 (ours / paper) | Overs 11-15 (ours / paper) | Overs 16-20 (ours / paper) |
|---|---|---|---|---|
| AdaBoostClassifier | 63.02 / — | 73.74 / — | 76.22 / — | 79.21 / — |
| Bagging Classifier | 59.51 / — | 66.43 / — | 71.41 / — | 74.25 / — |
| BernoulliNB | 61.21 / — | 68.08 / — | 70.86 / — | 55.20 / — |
| DecisionTreeClassifier | 55.16 / — | 70.11 / — | 65.66 / — | 71.63 / — |
| GaussianNB | 60.99 / — | 64.78 / — | 67.69 / — | 70.22 / — |
| GradientBoostingClassifier | 63.35 / — | 69.12 / — | 78.02 / — | 80.95 / — |
| KNeighborsClassifier | 62.09 / — | 68.24 / — | 69.82 / — | 75.32 / — |
| Logistic Regression | 64.01 / 66.70 | 69.07 / 69.84 | 79.33 / 78.79 | 79.61 / 77.93 |
| RandomForestClassifier | 63.63 / 66.54 | 71.70 / 70.49 | 78.13 / 78.57 | 81.09 / 75.18 |
| SVC | 63.46 / 68.68 | 70.93 / 72.36 | 81.08 / 79.22 | 81.42 / 77.26 |
| XGBClassifier | 60.05 / 67.53 | 66.48 / 70.49 | 69.11 / 76.49 | 82.49 / 75.12 |
| m2o MF Stacked GRU | 61.21 / 68.59 | 73.46 / 71.51 | 70.53 / 78.76 | 79.68 / 75.62 |
| m2o MF Stacked LSTM | 63.63 / 68.65 | 68.35 / 70.96 | 76.49 / 78.65 | 77.67 / 73.94 |
| m2o MF vanilla GRU | 65.11 / 69.25 | 73.41 / 71.34 | 80.48 / 78.82 | 83.90 / 75.02 |
| m2o MF vanilla LSTM | 63.74 / 67.66 | 70.16 / 71.45 | 78.95 / 79.04 | 81.35 / 76.43 |
| o2o MF Stacked GRU | 64.07 / 68.52 | 72.25 / 72.03 | 78.73 / 78.90 | 79.95 / 75.65 |
| o2o MF Stacked lstm | 64.89 / 68.57 | 69.84 / 72.09 | 77.97 / 78.90 | 77.33 / 75.72 |
| o2o MF bidirec GRU | 63.79 / 68.19 | 72.58 / 71.92 | 83.00 / 79.22 | 86.85 / 77.13 |
| o2o MF bidirec lstm | 62.69 / 68.13 | 70.11 / 71.98 | 81.52 / 79.28 | 87.66 / 75.59 |
| o2o MF vanilla GRU | 63.96 / 68.68 | 68.96 / 71.87 | 81.79 / 79.28 | 85.11 / 73.44 |
| o2o MF vanilla lstm | 64.07 / 68.57 | 68.96 / 71.65 | 82.39 / 78.73 | 86.05 / 73.98 |

## E2. Two paper models + two new models (paper features, five metrics)

The paper's two models used here are **SVC** and **Random Forest**; the paper names them as its best
classical models. The two new models, not used in the paper, are **LightGBM** and **CatBoost**.

* **Season split:** train on all earlier seasons, test on the latest season (IPL: 2019, PSL: 2026).
* **Random 80/20:** 80 % of matches for training, 20 % for testing (`random_state = 42`). All balls of a
  match stay on the same side, so the model never sees part of a test match.

| League | Split | Role | Model | Accuracy | Precision | Recall | F1 | ROC_AUC |
|---|---|---|---|---|---|---|---|---|
| IPL | Season split (test 2019) | Paper model | SVC | 75.01 | 80.83 | 76.43 | 78.57 | 83.95 |
| IPL | Season split (test 2019) | Paper model | RandomForestClassifier | 72.99 | 81.88 | 70.53 | 75.78 | 82.09 |
| IPL | Season split (test 2019) | New model | LightGBM | 70.24 | 77.82 | 70.41 | 73.93 | 79.04 |
| IPL | Season split (test 2019) | New model | CatBoost | 67.20 | 75.90 | 66.31 | 70.78 | 76.95 |
| IPL | Random 80/20 (by match) | Paper model | SVC | 78.68 | 79.69 | 79.73 | 79.71 | 85.37 |
| IPL | Random 80/20 (by match) | Paper model | RandomForestClassifier | 76.36 | 78.17 | 76.30 | 77.22 | 84.01 |
| IPL | Random 80/20 (by match) | New model | LightGBM | 78.03 | 79.80 | 77.87 | 78.83 | 87.33 |
| IPL | Random 80/20 (by match) | New model | CatBoost | 77.31 | 78.46 | 78.29 | 78.37 | 85.16 |
| PSL | Season split (test 2026) | Paper model | SVC | 76.84 | 90.33 | 71.15 | 79.60 | 88.43 |
| PSL | Season split (test 2026) | Paper model | RandomForestClassifier | 72.59 | 89.47 | 64.43 | 74.91 | 84.66 |
| PSL | Season split (test 2026) | New model | LightGBM | 72.73 | 88.65 | 65.45 | 75.30 | 83.28 |
| PSL | Season split (test 2026) | New model | CatBoost | 71.55 | 89.41 | 62.64 | 73.67 | 83.96 |
| PSL | Random 80/20 (by match) | Paper model | SVC | 77.87 | 81.54 | 79.42 | 80.47 | 83.98 |
| PSL | Random 80/20 (by match) | Paper model | RandomForestClassifier | 76.39 | 79.31 | 79.64 | 79.47 | 82.84 |
| PSL | Random 80/20 (by match) | New model | LightGBM | 76.81 | 80.49 | 78.66 | 79.57 | 82.59 |
| PSL | Random 80/20 (by match) | New model | CatBoost | 77.80 | 83.01 | 77.10 | 79.94 | 80.84 |

On the paper's own features the two new models do **not** beat the paper's SVC. Their gains come with
the proposed method in E3.

---

## E3. Novelty: league-aware transfer learning with chase-state features

### Problem
The PSL is a much smaller league than the IPL: 354 decided matches since 2016, against 752 IPL matches in
the paper's dataset alone. A ball-by-ball win predictor trained on PSL data alone has little to learn
from. With the paper's method, the boosting models reach only about 67–68 % accuracy, and their Brier
score is close to or worse than 0.25, which is what always predicting 50 % would score. The paper's raw
features (runs of batsman 1–11, bowler 1–10) also never state the quantities that decide a chase.

### Proposed method
1. **Chase-state features (7):** runs needed, current run rate, required run rate, their gap,
   resources (wickets left × balls left) and momentum (runs and wickets in the last 12 balls).
2. **League-aware transfer:** train on IPL + all earlier PSL seasons together, with a league flag.
3. **Honest evaluation:** six PSL seasons (2021–2026), each predicted using **only earlier seasons**.
   Results are reported as mean ± sd, with the Brier score added to the five metrics.

### Results (PSL, mean over test seasons 2021–2026)

| Setting | Model | Accuracy | F1 | ROC-AUC | Brier |
|---|---|---|---|---|---|
| S1 Paper method: PSL data, paper features | SVC | 73.62 ± 2.83 | 69.99 | 81.51 | 0.178 |
| S1 Paper method: PSL data, paper features | RandomForestClassifier | 71.11 ± 4.51 | 67.81 | 79.51 | 0.189 |
| S1 Paper method: PSL data, paper features | LightGBM | 67.10 ± 7.15 | 62.19 | 74.64 | 0.258 |
| S1 Paper method: PSL data, paper features | CatBoost | 67.94 ± 4.85 | 64.56 | 76.21 | 0.246 |
| S2 PSL data + chase features | SVC | 73.99 ± 3.23 | 70.31 | 81.94 | 0.176 |
| S2 PSL data + chase features | RandomForestClassifier | 75.10 ± 4.03 | 71.91 | 84.23 | 0.165 |
| S2 PSL data + chase features | LightGBM | 69.84 ± 5.88 | 65.02 | 79.04 | 0.232 |
| S2 PSL data + chase features | CatBoost | 70.19 ± 6.10 | 65.23 | 79.18 | 0.228 |
| S3 IPL+PSL transfer, paper features | SVC | 75.64 ± 5.36 | 71.28 | 83.92 | 0.164 |
| S3 IPL+PSL transfer, paper features | RandomForestClassifier | 72.48 ± 6.47 | 66.75 | 80.82 | 0.179 |
| S3 IPL+PSL transfer, paper features | LightGBM | 73.80 ± 6.62 | 68.53 | 81.51 | 0.179 |
| S3 IPL+PSL transfer, paper features | CatBoost | 72.73 ± 5.05 | 68.11 | 80.26 | 0.198 |
| S4 Proposed: IPL+PSL transfer + chase + league flag | SVC | 76.09 ± 4.35 | 72.34 | 84.29 | 0.163 |
| S4 Proposed: IPL+PSL transfer + chase + league flag | RandomForestClassifier | 75.88 ± 3.88 | 71.68 | 85.52 | 0.155 |
| S4 Proposed: IPL+PSL transfer + chase + league flag | LightGBM | 76.42 ± 4.89 | 72.39 | 84.37 | 0.162 |
| S4 Proposed: IPL+PSL transfer + chase + league flag | CatBoost | 73.40 ± 4.29 | 69.51 | 82.12 | 0.194 |
| A1 Ablation: S4 + context (Elo, toss, venue) | SVC | 73.92 ± 3.75 | 68.16 | 83.04 | 0.170 |
| A1 Ablation: S4 + context (Elo, toss, venue) | RandomForestClassifier | 74.48 ± 3.58 | 69.51 | 85.31 | 0.158 |
| A1 Ablation: S4 + context (Elo, toss, venue) | LightGBM | 73.72 ± 2.58 | 69.34 | 82.56 | 0.194 |
| A1 Ablation: S4 + context (Elo, toss, venue) | CatBoost | 70.37 ± 4.33 | 65.60 | 78.52 | 0.238 |
| A2 IPL only (zero-shot), chase features | SVC | 76.16 ± 5.10 | 71.42 | 84.05 | 0.162 |
| A2 IPL only (zero-shot), chase features | RandomForestClassifier | 75.19 ± 4.20 | 71.02 | 84.95 | 0.159 |
| A2 IPL only (zero-shot), chase features | LightGBM | 74.41 ± 4.70 | 69.68 | 82.54 | 0.181 |
| A2 IPL only (zero-shot), chase features | CatBoost | 73.70 ± 4.99 | 69.71 | 82.03 | 0.193 |

**Proposed (S4) vs the paper's method (S1):**

| Model | Accuracy gain (pp) | F1 gain (pp) | ROC-AUC gain (pp) | Brier change | Seasons won |
|---|---|---|---|---|---|
| SVC | +2.47 | +2.35 | +2.78 | -0.015 | 5 / 6 |
| RandomForestClassifier | +4.77 | +3.87 | +6.01 | -0.035 | 6 / 6 |
| LightGBM | +9.32 | +10.20 | +9.73 | -0.096 | 6 / 6 |
| CatBoost | +5.46 | +4.96 | +5.90 | -0.052 | 6 / 6 |

* The proposed method improves **all four models**, and wins **23 of 24** season × model comparisons.
* Most of the gain comes from transfer. A model trained on IPL alone (A2, "zero-shot") already beats the
  PSL-only model, because cricket chases follow the same logic in both leagues.
* **Tested and rejected:** adding team strength (Elo), toss and venue history (A1) **hurts** in the PSL,
  because with few matches these statistics are noisy. In the much larger IPL it helps the tree models
  (table below). So they are not part of the proposed method.

### Supporting check: the chase-state features on the paper's own IPL 2019 test

| Feature_set | Model | Accuracy | Precision | Recall | F1 | ROC_AUC | Brier |
|---|---|---|---|---|---|---|---|
| Paper features | SVC | 74.94 | 80.36 | 77.00 | 78.64 | 84.06 | 0.166 |
| Paper features | RandomForestClassifier | 72.99 | 81.88 | 70.53 | 75.78 | 82.09 | 0.175 |
| Paper features | LightGBM | 70.24 | 77.82 | 70.41 | 73.93 | 79.04 | 0.207 |
| Paper features | CatBoost | 67.20 | 75.90 | 66.31 | 70.78 | 76.95 | 0.234 |
| Paper + chase features | SVC | 74.78 | 80.29 | 76.76 | 78.48 | 84.22 | 0.165 |
| Paper + chase features | RandomForestClassifier | 72.77 | 80.89 | 71.44 | 75.87 | 84.39 | 0.164 |
| Paper + chase features | LightGBM | 70.09 | 78.81 | 68.49 | 73.29 | 80.96 | 0.201 |
| Paper + chase features | CatBoost | 68.13 | 77.08 | 66.62 | 71.47 | 78.00 | 0.235 |
| Paper + chase + context | SVC | 73.39 | 79.58 | 74.77 | 77.10 | 82.20 | 0.174 |
| Paper + chase + context | RandomForestClassifier | 74.38 | 80.53 | 75.49 | 77.93 | 83.93 | 0.164 |
| Paper + chase + context | LightGBM | 72.59 | 80.49 | 71.61 | 75.79 | 82.12 | 0.196 |
| Paper + chase + context | CatBoost | 73.38 | 79.44 | 74.96 | 77.14 | 79.20 | 0.215 |

On IPL 2019 the chase-state features raise ROC-AUC for all four models. Accuracy changes are small,
because the IPL already has enough data for the raw features.

## Figures

`public/research-data/figures/`:
* `fig1_reproduction_ipl2019.png`: reproduction vs paper
* `fig2_novelty_psl_rolling.png`: novelty results
* `fig3_accuracy_by_phase.png`: accuracy by phase of the chase

## Limitations

* For the ~1 % of matches shortened by rain, the target is the first-innings total, as in the paper
  (the Kaggle data has no revised targets).
* In E3, the RBF-SVC is trained on at most 40,000 rows (uniform ball-stride subsampling, the same in every
  setting) because its cost grows with the square of the data size. E1 and E2 use the full data.
* The SVC score is turned into a probability with a logistic function, so its Brier score is approximate.
* Test seasons have 23–43 matches, so single-season numbers vary. That is why E3 averages six seasons.

## Deploying the dashboard (GitHub Pages)

The workflow in `.github/workflows/deploy.yml` builds and deploys on every push to `main`. The site
address is set from the repository name automatically.

1. On GitHub: **Settings → Pages → Source: GitHub Actions** (the repository must be public on a free account).
2. Push:
   ```bash
   git add -A
   git commit -m "Update"
   git push origin main
   ```
3. The site will be at `https://<your-username>.github.io/<repository-name>/`.

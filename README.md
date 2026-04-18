# Baseball Hall of Fame Ballot Prediction (PySpark)

Predicting whether an MLB player will ever appear on a National Baseball Hall of Fame (HoF) ballot using a scalable PySpark ML pipeline built on the Lahman Baseball Database.

## Project Overview

The National Baseball Hall of Fame ballot is tightly controlled by the BBWAA and a screening committee that only considers players with long, star-level careers.
This project builds a **binary classification model** that imitates the screening committee and predicts whether a player deserves to appear on a HoF ballot at least once (`ballot_appearance = 1` vs `0`).

Key challenges:

- **Extreme class imbalance** (only around 6% of players ever reach the ballot).
- **Multi-table joins** across thousands of player–season records.
- **Heterogeneous features** (batting, pitching, fielding, salary, All-Star appearances). 
- **Temporal dependencies** requiring **career-level aggregation**.

The project is implemented end-to-end in **PySpark**, from data loading and preprocessing to modeling, hyperparameter tuning, and evaluation.

## Objectives

The project was designed with the following objectives:

- Perform **comprehensive EDA** with 15–20 KPIs over player careers.  
- Engineer **career-aggregate features** (e.g. total AB, H, IPouts, salary, All-Star games, position tenure).  
- Train **5 classifiers** and fine‑tune the best 3 via grid-search hyperparameter optimization.  
- Evaluate models using **AUROC** and **macro F1-score**, plus confusion matrices on an 80/20 train–test split.  
- Demonstrate **big-data scalability** and interpretable insights into HoF ballot selection criteria.  

## Data

**Source:** Lahman Baseball Database (Kaggle), seasons 1871–2023.

Selected tables:

- `Batting`
- `Pitching`
- `Fielding`
- `Salaries`
- `All-Star`

Only **players** are considered in this version (no managers, coaches, etc.).

After joins and feature selection, the combined Spark DataFrame contained:

- **214,157 rows**  
- **75 columns** before final feature pruning.

### Target Variable

A custom binary target `ballot_appearance` is constructed:

- `1` – player appears on a Baseball Hall of Fame ballot at least once.  
- `0` – player never appears on a ballot.

Ballot eligibility rules (10+ MLB seasons, 5 years retired, screening committee) are reflected implicitly via historical ballot data rather than reimplemented from scratch.

## Features & Engineering

Features were chosen based on their relevance to HoF criteria, following sabermetric principles.

### Raw Feature Categories

- **Batting:** G, AB, H, HR, RBI, BB, SO.
- **Pitching:** W, L, ERA, IPouts, strikeouts (PitSO).  
- **Fielding:** PO, A, E (putouts, assists, errors).
- **General:** total career salary, primary position, total All-Star appearances.

### Position Handling

Players can appear at multiple positions across seasons. To derive a **primary position**:

1. For each year, select the position with the **most putouts (PO)** as the main position.  
2. Over the full career, the position that is main in **most years** becomes the player’s primary position.

### Career-Level Aggregation

The pipeline aggregates season-level stats to **career-level** per player using PySpark aggregations:

- Career totals for batting: G, AB, H, HR, RBI, BB, SO.  
- Career totals for pitching: W, L, IPouts, PitSO; average ERA.  
- Career totals for fielding: PO, A, E.  
- Total career salary.  
- Total All-Star games.  
- Primary position (as derived above).

This yields a **player-level** DataFrame with one row per player and associated `ballot_appearance` label.

### Missing Values & Type-Specific Stats

- Pitching stats (W, L, ERA, IPouts, PitSO) are set to **0 for non-pitchers**, since those columns do not apply to other positions.  
- Salary is missing for many historical players; a **median salary** imputation is used to avoid letting this feature dominate while still capturing informative signal for modern players. 
- Players without games have fielding statistics imputed as 0; unknown positions are labeled as `U` (unknown).

### Encoding & Scaling

For modeling, the pipeline includes:

- One‑hot encoding of the **position**.  
- `VectorAssembler` to combine numerical and categorical features into a feature vector.  
- Standard scaling to prevent large-magnitude features (e.g. salary) from dominating the model.  
- **Class weighting** to compensate for the severe underrepresentation of ballot players.

No outliers are removed, since extreme performance is exactly what defines many ballot and HoF players; dropping them would disproportionately remove positive examples.

## Exploratory Data Analysis (EDA)

Key descriptive findings:

- Only **~6%** of all players ever appear on a HoF ballot (strong class imbalance).  
- Pitchers are the **most common** position, followed by outfielders; some positions (LF, RF, CF, DH) are rare.
- Pitchers have comparatively **low ballot probabilities**; right fielders and some offensive positions show much **higher ballot appearance rates**, reflecting bias toward hitting stats over pure fielding. 
- Correlation analysis shows:
  - Games played, hits, RBI, and All-Star appearances have **strong positive correlation** with ballot appearance.  
  - Pure fielding stats (PO, A) are less strongly correlated.  
  - Salary appears weakly correlated overall, but provides important information for recent-era players.

## Modeling & Methodology

All models are implemented in **PySpark ML**.

### Models Evaluated

Five classifiers were initially benchmarked:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Gradient Boosted Trees (GBT)  
- Multilayer Perceptron (Neural Network / MLP)

The primary selection metric for model choice was **AUROC**, as it is robust for imbalanced classification. Macro F1-score and confusion matrices were used to assess class-wise performance.

### Hyperparameter Tuning

The top 3 models by AUROC (Random Forest, Gradient Boosting, MLP) were then fine‑tuned using **grid search with cross-validation**:

- **Random Forest:** number of trees (`numTrees` ∈ {50, 100, 200}) and maximum depth (`maxDepth` ∈ {5, 10, 15}).  
- **MLP:** network architecture (`layers`), maximum iterations (`maxIter` ∈ {100, 250}), and learning rate (`stepSize` ∈ {0.03, 0.01}).  
- **GBT:** number of iterations (`maxIter` ∈ {20, 50, 100}), tree depth (`maxDepth` ∈ {3, 5, 7}), and learning rate (`stepSize` ∈ {0.1, 0.05}).

Stratified 3‑fold cross‑validation is used to select hyperparameters that maximize AUROC.

## Results

### Overall Performance

- The **Random Forest classifier** achieves the best AUROC, around **0.97**, and a **macro F1-score > 0.84** after tuning.
- Hyperparameter tuning yields only marginal AUROC gains but slightly improves the macro F1 (≈ +0.5 percentage points).

### Feature Importance (Random Forest)

Top 10 most important features:

1. Games played (G)  
2. Hits (H)  
3. All-Star appearances  
4. At bats (AB)  
5. Runs batted in (RBI)  
6. Walks (BB)  
7. Putouts (PO)  
8. Wins (W)  
9. Strikeouts (SO – batting)  
10. Errors (E)

This confirms a strong bias toward **hitting performance** and visibility (e.g. All-Star appearances), with fielding and pitching stats playing a secondary role.

### Threshold Tuning and Practical Use

At the default decision threshold optimized for AUROC, the model still under-detects the rare positive class (ballot players) in a way that is not ideal for screening.

To make the model more useful as a **pre-screening tool**, the decision threshold was lowered to **0.1**, intentionally favoring recall of ballot-worthy players:

- At threshold 0.1, the Random Forest correctly flags **209 out of 226** ballot players as positive (high recall), accepting more false positives to reduce the risk of missing legitimate candidates.

The intended real-world use is thus **not** as the final arbiter, but as a **filter that reduces the pool** of players the committee needs to manually review, while retaining almost all ballot-worthy candidates.

## Project Structure

Current structure:

```text
.
├── README.md
├── ProjectReport.pdf
├── requirements.txt
├── .gitignore
└── notebooks/
    ├── EDA.ipynb          # Exploratory data analysis
    └── ML.ipynb     # Feature engineering & PySpark ML pipeline
```

> Note: The implementation is currently notebook-based. All PySpark data processing and modeling code lives in the notebooks.

## Limitations

- The model can only use **observable performance and career stats**; it cannot account for off-field issues (e.g. banning, scandals), which explains cases like **Pete Rose** (elite stats but never on a ballot).
- Historical salary coverage is incomplete, especially before the 1980s; median imputation dilutes its effect, but the feature is still more informative for recent players than for early-era ones.
- Position handling through PO-based primary position is a pragmatic heuristic and may introduce bias for utility players.

Overall, the pipeline demonstrates how **big data and ML can support, but not replace, human judgement** in Hall of Fame ballot selection.

## References

- Lahman Baseball Database (Kaggle dataset, 1871–2023). 
- Baseball Hall of Fame BBWAA election rules.
- SABR resources on sabermetrics and objective baseball analysis.

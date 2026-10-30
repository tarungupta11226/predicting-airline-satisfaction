# Gradient-Boosted Ensemble with Out-of-Fold Target Encoding for Airline Passenger Satisfaction

Kaggle competition entry: *Predicting Airline Satisfaction* (binary classification, metric: ROC-AUC).
**Public leaderboard ROC-AUC: 0.96028** (3-model blend).

![Overview](assets/fig_overview.png)

*Figure 1. (a) Pipeline. (b) Holdout ablation, axis truncated; single 80/20 split, one seed. (c) Learning curve with a fixed holdout. Public leaderboard ROC-AUC of the final blend: 0.96028.*

## Abstract

We study tabular binary classification on a synthetic airline-satisfaction dataset (699,635 labelled rows, 299,844 test rows, 22 features). A single gradient-boosted tree model reaches 0.9586 ROC-AUC on a held-out 20% split. Treating `Flight Distance`, which has only 3,474 distinct values, as a group identifier through out-of-fold target encoding, together with finer histogram binning, raises this to 0.9598. An equal-weight blend of LightGBM, XGBoost and CatBoost reaches 0.9604 on the holdout and 0.96028 on the public leaderboard. Several further ideas (external data, pseudo-labelling, exact-pattern encodings) did not help and are reported below.

## Task and metric

Each row describes one passenger (age, distance, 13 service ratings, delays, gender, customer type, travel type, class). The target is binary satisfaction. Submissions are probabilities scored by ROC-AUC, which equals the probability that a random positive example is ranked above a random negative one. Because AUC depends only on the ordering of scores, we submit probabilities and average models in probability space.

## Method

1. **Out-of-fold target encoding (TE).** For key sets `{Flight Distance}` and `{Flight Distance, Class, Type of Travel}`, each category is replaced by a smoothed mean of the target,
   `(sum_y + prior * s) / (count + s)` with `s = 5`. Training rows are encoded with 5-fold out-of-fold statistics so a row's own label is never used in its encoding; test rows use statistics from the full training set.
2. **Models.** All three are gradient-boosted decision trees trained on log-loss.
   - LightGBM (Ke et al., 2017): two configurations (31 and 127 leaves, learning rate 0.03, `max_bin=4000`), predictions averaged.
   - XGBoost (Chen & Guestrin, 2016): depth 7, histogram method, 1,024 bins, native categoricals.
   - CatBoost (Prokhorenkova et al., 2018): ordered target statistics on raw categorical keys (`Flight Distance`, and `Flight Distance x Class x Travel type`) instead of the hand-built TE.
3. **Ensembling.** Unweighted mean of the three probability vectors. Pairwise Spearman rank correlations between model outputs on the test set are 0.97-0.99; CatBoost is the least correlated with the other two.
4. **Iteration counts** were selected by early stopping on a holdout, then scaled up by about 10% for the final fit on all training data.

## Results

Single 80/20 split (n = 140,000 holdout rows), one random seed.

| Model | Holdout ROC-AUC |
|---|---|
| LightGBM, default binning | 0.9586 |
| + `max_bin = 4000` | 0.9593 |
| + TE (`Flight Distance`) | 0.9597 |
| + TE (`Flight Distance x Class x Travel`) | 0.9598 |
| XGBoost + TE | 0.9599 |
| CatBoost (native categorical) | 0.9603 |
| LightGBM + XGBoost | 0.9602 |
| **LightGBM + XGBoost + CatBoost** | **0.9604** |

5-fold cross-validation of the two-configuration LightGBM model (TE rebuilt inside each fold) gives pooled ROC-AUC 0.96058. Public leaderboard for the final 3-model blend: **0.96028**.

Learning curve (LightGBM + TE, fixed holdout): 140k rows 0.9584, 280k rows 0.9592, 560k rows 0.9598. Each doubling of data gives a diminishing gain.

### Ablations with no measurable gain

Changes in holdout ROC-AUC relative to the matching reference model:

| Idea | Change |
|---|---|
| Original (pre-synthetic) dataset appended as extra training rows | +0.00003 |
| Model trained on the original dataset used as a feature (0.995 CV AUC on original data) | -0.00001 |
| Soft pseudo-labels from test predictions | -0.00015 |
| Target encodings of exact rating-and-category patterns | -0.0003 to +0.0001 |
| Aggregate rating features (sum, mean, min, count of zeros), delay differences | about -0.0004 |

A check for near-duplicate rows inside the data found no memorisation structure (leave-one-out label agreement among exact duplicates is about 0.92, comparable to model accuracy).

## Limitations

- Holdout differences below about 1e-4 are within the noise of a single split and seed; only the larger steps (binning, TE, CatBoost, blending) are likely real. The blend's improvement on the leaderboard (+0.00017 over a LightGBM blend) was smaller than the holdout suggested.
- The final models are trained once on all data with fixed iteration counts, without cross-validated blend weights.
- The entry sits below the top of the leaderboard. We did not test neural models with learned embeddings, multi-seed averaging or stacking, which are plausible sources of further gains.

## Reproducing

```bash
pip install -r requirements.txt
# place train.csv and test.csv from the competition page in this directory
python model.py          # writes submission.csv (about 45 min on a single core)
```

The competition data is not redistributed here; download it from the competition page: https://www.kaggle.com/competitions/playground-series-s6e10.

## References

- Ke et al., LightGBM: A Highly Efficient Gradient Boosting Decision Tree, NeurIPS 2017.
- Chen and Guestrin, XGBoost: A Scalable Tree Boosting System, KDD 2016.
- Prokhorenkova et al., CatBoost: Unbiased Boosting with Categorical Features, NeurIPS 2018.
- Micci-Barreca, A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems, SIGKDD Explorations 2001.

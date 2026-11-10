Dataset : California Housing Prices (Kaggle) - housing.csv (20,640 rows)
Models  : 1) Multiple Linear Regression
          2) Decision Tree Regressor
          3) K-Means Clustering
Run     : python ml_project.py            (housing.csv in the same folder)
          python ml_project.py path/to/housing.csv
Output  : results/ folder with plots (PNG), metrics.csv and analysis.txt
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save plots to files, no window needed
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (davies_bouldin_score, mean_absolute_error,
                             mean_squared_error, r2_score, silhouette_score)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

DATA = sys.argv[1] if len(sys.argv) > 1 else "housing.csv"
OUT = "results"
SEED = 42
TARGET = "median_house_value"

if not os.path.exists(DATA):
    sys.exit(f"'{DATA}' not found. Download it from Kaggle (see README.md) "
             "and keep it next to this file.")
os.makedirs(OUT, exist_ok=True)


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=150)
    plt.close()


def scores(y_true, y_pred):
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": r2_score(y_true, y_pred)}


# ---------------------------------------------------------------
# 1. LOAD + CLEAN
# ---------------------------------------------------------------
df = pd.read_csv(DATA)
print(f"Loaded {DATA}: {df.shape[0]} rows x {df.shape[1]} columns")
print("Missing values:\n", df.isna().sum()[df.isna().sum() > 0].to_string(), "\n")

df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())

# the target is capped at 500,001 (censored values), so those rows are removed
before = len(df)
df = df[df[TARGET] < 500000].copy()
print(f"Removed {before - len(df)} rows with capped house values -> {len(df)} rows left\n")

# feature engineering
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
df["population_per_household"] = df["population"] / df["households"]
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True, dtype=float)

X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows | Features: {X.shape[1]}\n")

# ---------------------------------------------------------------
# 2. DATA VISUALS
# ---------------------------------------------------------------
plt.figure(figsize=(7, 4))
plt.hist(y, bins=50, color="#4750AC", edgecolor="white")
plt.title("Distribution of Median House Value")
plt.xlabel("Median house value ($)")
plt.ylabel("Districts")
save("01_target_distribution.png")

corr = df.corr()
plt.figure(figsize=(11, 9))
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
plt.yticks(range(len(corr)), corr.columns, fontsize=7)
for i in range(len(corr)):
    for j in range(len(corr)):
        plt.text(j, i, f"{corr.iloc[i, j]:.1f}", ha="center", va="center", fontsize=5)
plt.title("Correlation Heatmap")
save("02_correlation_heatmap.png")

results = {}

# ---------------------------------------------------------------
# 3. MODEL 1 - MULTIPLE LINEAR REGRESSION
# ---------------------------------------------------------------
lr = make_pipeline(StandardScaler(), LinearRegression())
t = time.time()
lr.fit(X_train, y_train)
lr_time = time.time() - t
lr_pred = lr.predict(X_test)
results["Multiple Linear Regression"] = {
    **scores(y_test, lr_pred),
    "CV R2 (5-fold)": cross_val_score(lr, X_train, y_train, cv=5, scoring="r2").mean(),
    "Train R2": lr.score(X_train, y_train),
    "Train time (s)": lr_time,
}

# ---------------------------------------------------------------
# 4. MODEL 2 - DECISION TREE REGRESSOR (with hyper-parameter tuning)
# ---------------------------------------------------------------
grid = GridSearchCV(
    DecisionTreeRegressor(random_state=SEED),
    {"max_depth": [3, 5, 7, 10, 15, None], "min_samples_leaf": [1, 10, 30]},
    cv=3, scoring="r2", n_jobs=-1)
grid.fit(X_train, y_train)
print("Decision Tree best parameters:", grid.best_params_)

dt = DecisionTreeRegressor(random_state=SEED, **grid.best_params_)
t = time.time()
dt.fit(X_train, y_train)
dt_time = time.time() - t
dt_pred = dt.predict(X_test)
results["Decision Tree"] = {
    **scores(y_test, dt_pred),
    "CV R2 (5-fold)": cross_val_score(dt, X_train, y_train, cv=5, scoring="r2").mean(),
    "Train R2": dt.score(X_train, y_train),
    "Train time (s)": dt_time,
}

# depth vs R2 curve (under-fitting vs over-fitting)
depths = list(range(2, 21))
tr_r2, te_r2 = [], []
for d in depths:
    m = DecisionTreeRegressor(max_depth=d, min_samples_leaf=grid.best_params_["min_samples_leaf"],
                              random_state=SEED).fit(X_train, y_train)
    tr_r2.append(m.score(X_train, y_train))
    te_r2.append(m.score(X_test, y_test))
plt.figure(figsize=(7, 4))
plt.plot(depths, tr_r2, "o-", label="Train R2", color="#4750AC")
plt.plot(depths, te_r2, "o-", label="Test R2", color="#FF4D6D")
plt.xlabel("max_depth")
plt.ylabel("R2 score")
plt.title("Decision Tree: depth vs performance")
plt.legend()
plt.grid(alpha=0.3)
save("04_tree_depth_curve.png")

# ---------------------------------------------------------------
# 5. MODEL 3 - K-MEANS CLUSTERING (unsupervised)
# ---------------------------------------------------------------
scaler = StandardScaler().fit(X_train)
Xs_train = scaler.transform(X_train)
Xs_test = scaler.transform(X_test)

rng = np.random.RandomState(SEED)
sample = rng.choice(len(Xs_train), size=min(5000, len(Xs_train)), replace=False)
ks = list(range(2, 11))
inertia, sil = [], []
for k in ks:
    km_k = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xs_train)
    inertia.append(km_k.inertia_)
    sil.append(silhouette_score(Xs_train[sample], km_k.labels_[sample]))
best_k = ks[int(np.argmax(sil))]
print(f"K-Means: best k by silhouette = {best_k}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ks, inertia, "o-", color="#4750AC")
ax[0].set_title("Elbow method")
ax[0].set_xlabel("k")
ax[0].set_ylabel("Inertia")
ax[1].plot(ks, sil, "o-", color="#FF4D6D")
ax[1].axvline(best_k, ls="--", color="gray")
ax[1].set_title("Silhouette score")
ax[1].set_xlabel("k")
ax[1].set_ylabel("Silhouette")
save("05_kmeans_k_selection.png")

t = time.time()
km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(Xs_train)
km_time = time.time() - t
train_labels = km.labels_
test_labels = km.predict(Xs_test)

# K-Means is not a predictor, so to compare it with the other two models each
# district is "predicted" as the average house value of its cluster.
cluster_mean = pd.Series(y_train.values).groupby(train_labels).mean()
km_pred = pd.Series(test_labels).map(cluster_mean).values
results["K-Means (cluster mean)"] = {
    **scores(y_test, km_pred),
    "CV R2 (5-fold)": np.nan,
    "Train R2": r2_score(y_train, pd.Series(train_labels).map(cluster_mean).values),
    "Train time (s)": km_time,
}
silhouette = silhouette_score(Xs_train[sample], train_labels[sample])
dbi = davies_bouldin_score(Xs_train, train_labels)

profile = X_train[["median_income", "housing_median_age", "population"]].copy()
profile["house_value"] = y_train.values
profile["cluster"] = train_labels
profile = profile.groupby("cluster").agg(
    districts=("house_value", "size"),
    avg_income=("median_income", "mean"),
    avg_age=("housing_median_age", "mean"),
    avg_population=("population", "mean"),
    avg_house_value=("house_value", "mean")).round(2)
profile.to_csv(os.path.join(OUT, "kmeans_cluster_profile.csv"))
print("\nK-Means cluster profile:\n", profile.to_string(), "\n")

plt.figure(figsize=(7, 6))
plt.scatter(X_train["longitude"], X_train["latitude"], c=train_labels, cmap="tab10", s=4, alpha=0.7)
plt.title(f"K-Means clusters (k={best_k}) on the California map")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
save("06_kmeans_clusters_map.png")

# ---------------------------------------------------------------
# 6. COMPARISON + REMAINING PLOTS
# ---------------------------------------------------------------
comp = pd.DataFrame(results).T[["MAE", "RMSE", "R2", "Train R2", "CV R2 (5-fold)", "Train time (s)"]]
comp.round(4).to_csv(os.path.join(OUT, "metrics.csv"))
print("MODEL COMPARISON (test set)\n", comp.round(4).to_string(), "\n")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for a, pred, name in [(ax[0], lr_pred, "Multiple Linear Regression"), (ax[1], dt_pred, "Decision Tree")]:
    a.scatter(y_test, pred, s=4, alpha=0.4, color="#4750AC")
    lim = [y_test.min(), y_test.max()]
    a.plot(lim, lim, "r--")
    a.set_title(f"{name}: actual vs predicted")
    a.set_xlabel("Actual value ($)")
    a.set_ylabel("Predicted value ($)")
save("03_actual_vs_predicted.png")

names = ["Linear\nRegression", "Decision\nTree", "K-Means"]
cols = ["#4750AC", "#FF4D6D", "#1DD1A1"]
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for a, metric in zip(ax, ["R2", "RMSE", "MAE"]):
    vals = comp[metric].values
    a.bar(names, vals, color=cols)
    a.set_title(metric + (" (higher is better)" if metric == "R2" else " (lower is better)"))
    for i, v in enumerate(vals):
        a.text(i, v, f"{v:.2f}" if metric == "R2" else f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
save("07_model_comparison.png")

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
imp = pd.Series(dt.feature_importances_, index=X.columns).sort_values().tail(10)
ax[0].barh(imp.index, imp.values, color="#FF4D6D")
ax[0].set_title("Decision Tree: top 10 feature importances")
coef = pd.Series(lr.named_steps["linearregression"].coef_, index=X.columns)
coef = coef.reindex(coef.abs().sort_values().tail(10).index)
ax[1].barh(coef.index, coef.values, color="#4750AC")
ax[1].set_title("Linear Regression: top 10 coefficients (scaled)")
save("08_feature_importance.png")

# ---------------------------------------------------------------
# 7. ANALYSIS
# ---------------------------------------------------------------
best_name = comp["R2"].idxmax()
gap = results["Decision Tree"]["Train R2"] - results["Decision Tree"]["R2"]
top_feat = imp.index[-1]
rich = profile["avg_house_value"].idxmax()
poor = profile["avg_house_value"].idxmin()
analysis = f"""ANALYSIS
========
Dataset: {len(df)} districts after cleaning, {X.shape[1]} features, target = {TARGET}.

1. Best model: {best_name} (test R2 = {comp.loc[best_name, 'R2']:.3f}, RMSE = ${comp.loc[best_name, 'RMSE']:,.0f}).

2. Multiple Linear Regression: R2 = {results['Multiple Linear Regression']['R2']:.3f}, RMSE = ${results['Multiple Linear Regression']['RMSE']:,.0f}.
   It is fast and easy to interpret, but it can only fit straight-line relationships.

3. Decision Tree (tuned: {grid.best_params_}): R2 = {results['Decision Tree']['R2']:.3f}, RMSE = ${results['Decision Tree']['RMSE']:,.0f}.
   Train R2 - Test R2 gap = {gap:.3f} ({'some over-fitting' if gap > 0.1 else 'little over-fitting'}).
   The most important feature is '{top_feat}'. Trees capture non-linear patterns and interactions.

4. K-Means (k = {best_k}): silhouette = {silhouette:.3f}, Davies-Bouldin = {dbi:.3f}.
   Cluster {rich} has the highest average house value (${profile.loc[rich, 'avg_house_value']:,.0f}) and
   cluster {poor} the lowest (${profile.loc[poor, 'avg_house_value']:,.0f}), so the clusters do separate price segments.
   K-Means is unsupervised, so used as a price predictor it scores R2 = {results['K-Means (cluster mean)']['R2']:.3f}.
   It is best used to discover market segments, not to predict exact prices.

5. Conclusion: use {best_name} for price prediction, and K-Means for customer / area segmentation.
"""
with open(os.path.join(OUT, "analysis.txt"), "w", encoding="utf-8") as f:
    f.write(analysis)
print(analysis)
print(f"All plots and files saved in the '{OUT}' folder.")

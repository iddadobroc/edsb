# %%
"""Classes 6 and 7, regression: what interest rate does a loan get?

The same steps as class_week06_week07_classification.py, on the same loans, but the
target is now a number.

Functions defined here:
    preprocess: prepares any set with what was learned from the training rows.
"""

# %% [markdown]
# ### Classes 6 & 7. The same steps for a number
# Enterprise Data Science Bootcamp, NOVA IMS, 2026-27
#
# João Azambuja

# %% [markdown]
# ----------------------------------------------------------------------
# # STAGE 0. GET READY
# ----------------------------------------------------------------------

# %% [markdown]
# **0.1** Open `class_week06_week07_regression.py`, in the same `notebooks` folder,
# and check that the bar at the bottom still shows `.venv`. If VS Code runs it in the
# Interactive Window of the classification file, that is fine, as long as you do not go
# back to the classification cells afterwards without running that file again from the
# top.

# %% [markdown]
# **0.2** This is the same check as in the classification file: it finds the project
# folder and prints it, and you should see `True` three times and `OK` four times.

# %%
import sys
from importlib.util import find_spec
from pathlib import Path

# Find the project folder: the first folder above us that has data/raw inside.
PROJECT = next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / "raw").exists()
)
CSV = PROJECT / "data" / "raw" / "lending-club.csv"
print("Project folder :", PROJECT)
print("Python 3.13    :", sys.version_info[:2] == (3, 13))
print("Running .venv  :", ".venv" in sys.executable)
print("Data in place  :", CSV.exists())
for package in ["pandas", "sklearn", "ydata_profiling", "plotly"]:
    print(f"{package:<15}:", "OK" if find_spec(package) else "MISSING")

# %% [markdown]
# ----------------------------------------------------------------------
# # STAGE 1. THE SAME THREE SETS, A NUMBER TO PREDICT
# ----------------------------------------------------------------------
# We use the same loans, the same sample and the same split as in the classification
# file. What we predict now is `int_rate`, the interest rate in percent. As before, the
# test set is profiled in step 1.3, but we choose nothing from it, and we score a model
# on it only once, at the end.

# %% [markdown]
# **1.1** We load the data and drop the same loans as before, so both examples work on
# the same rows.

# %%
from datetime import date

import pandas as pd
from sklearn import metrics
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv(CSV)
df = df.dropna(subset=["loan_status"])
df["issue_d"] = pd.to_datetime(df["issue_d"])
df.shape

# %% [markdown]
# **1.2** The same sample and split. This time the table shows the average rate in the
# training and validation sets.

# %%
SEED = 42
# We leave out sub_grade and installment: the lender sets them with the rate.
NUMERIC = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "fico_range_low",
    "revol_util",
    "revol_bal",
    "open_acc",
    "total_acc",
    "pub_rec",
    "pub_rec_bankruptcies",
    "mort_acc",
    "emp_length",
    "time_to_earliest_cr_line",
]
CATEGORICAL = [
    "term",
    "home_ownership",
    "purpose",
    "verification_status",
    "application_type",
]

dev = df[df["split"] == "train"].sample(20_000, random_state=SEED)
is_val = dev["issue_d"] >= "2016-01-01"
train, val = dev[~is_val], dev[is_val]
test = df[df["split"] == "test"]
y_train, y_val = train["int_rate"], val["int_rate"]

sets = []
for name, part in [("train", train), ("validation", val)]:
    sets.append(
        {
            "set": name,
            "loans": len(part),
            "first": part["issue_d"].min(),
            "last": part["issue_d"].max(),
            "mean_rate": part["int_rate"].mean(),
        }
    )
sets.append(
    {
        "set": "test (kept for the end)",
        "loans": len(test),
        "first": test["issue_d"].min(),
        "last": test["issue_d"].max(),
    }
)
pd.DataFrame(sets).round(2)

# %% [markdown]
# **1.3** The same four profiles, this time with `int_rate` among the columns. They go
# to `reports/generated/` as `profile-int-rate-all.html` and so on. This takes as long
# as in the classification file, and the cell prints nothing while it works.

# %%
from ydata_profiling import ProfileReport

(PROJECT / "reports" / "generated").mkdir(parents=True, exist_ok=True)
for name, part in [("all", df), ("train", train), ("validation", val), ("test", test)]:
    rows = part[NUMERIC + CATEGORICAL + ["int_rate"]].copy()
    profile = ProfileReport(
        rows, title=f"Classes 6 and 7, regression: {name}", progress_bar=False
    )
    profile.to_file(PROJECT / "reports" / "generated" / f"profile-int-rate-{name}.html")

# %% [markdown]
# **1.4** The same `preprocess` function, learned from the training rows of this file.

# %%
low = train[NUMERIC].quantile(0.01)
high = train[NUMERIC].quantile(0.99)
clipped = train[NUMERIC].clip(low, high, axis=1)
mean, std = clipped.mean(), clipped.std()
dummy_columns = pd.get_dummies(train[CATEGORICAL], dtype=float).columns


def preprocess(data):
    """Clip, scale and one-hot any set, using only what the training rows taught."""
    numbers = (data[NUMERIC].clip(low, high, axis=1) - mean) / std
    dummies = pd.get_dummies(data[CATEGORICAL], dtype=float)
    dummies = dummies.reindex(columns=dummy_columns, fill_value=0.0)
    return pd.concat([numbers, dummies], axis=1)


X_train, X_val = preprocess(train), preprocess(val)
X_train.shape

# %% [markdown]
# ----------------------------------------------------------------------
# # STAGE 2. SIX MODELS, ONE TABLE
# ----------------------------------------------------------------------
# The method is the same, only the columns change. Each model again gets a row for the
# training rows and a row for the validation rows.
#
# `mae` is the average miss and `median_ae` the middle one, both in percentage points.
# `mse` averages the squared misses and `rmse` is its square root, back in percentage
# points, so big misses count more. `mape` is the average miss as a share of the rate
# (0.2 means 20%), and `max_error` is the largest miss. `r2` is 1 for a perfect model
# and 0 for a model that always guesses the average of the rows it scores;
# `explained_variance` is the same idea, but it ignores a miss that is the same for
# every loan.

# %% [markdown]
# **2.1** The table starts with a model that does nothing: it predicts the average rate
# of the training rows for every loan. On the validation rows its `r2` lands a little
# below 0, because the validation loans have a slightly different average.

# %%
today = date.today().isoformat()
columns = [
    "date",
    "model",
    "dataset",
    "mae",
    "median_ae",
    "mse",
    "rmse",
    "mape",
    "max_error",
    "r2",
    "explained_variance",
    "note",
]
results = pd.DataFrame(columns=columns)
fitted = {}

name = "Baseline (training mean)"
model = DummyRegressor().fit(X_train, y_train)
fitted[name] = model
for dataset, X, y in [("train", X_train, y_train), ("validation", X_val, y_val)]:
    pred = model.predict(X)
    results.loc[len(results)] = [
        today,
        name,
        dataset,
        metrics.mean_absolute_error(y, pred),
        metrics.median_absolute_error(y, pred),
        metrics.mean_squared_error(y, pred),
        metrics.root_mean_squared_error(y, pred),
        metrics.mean_absolute_percentage_error(y, pred),
        metrics.max_error(y, pred),
        metrics.r2_score(y, pred),
        metrics.explained_variance_score(y, pred),
        "",
    ]
results.round(3)

# %% [markdown]
# **2.2** A linear regression of the ridge kind, which keeps the coefficients small.

# %%
name = "Ridge regression"
model = Ridge().fit(X_train, y_train)
fitted[name] = model
for dataset, X, y in [("train", X_train, y_train), ("validation", X_val, y_val)]:
    pred = model.predict(X)
    results.loc[len(results)] = [
        today,
        name,
        dataset,
        metrics.mean_absolute_error(y, pred),
        metrics.median_absolute_error(y, pred),
        metrics.mean_squared_error(y, pred),
        metrics.root_mean_squared_error(y, pred),
        metrics.mean_absolute_percentage_error(y, pred),
        metrics.max_error(y, pred),
        metrics.r2_score(y, pred),
        metrics.explained_variance_score(y, pred),
        "",
    ]
results.round(3)

# %% [markdown]
# **2.3** A decision tree limited to six levels.

# %%
name = "Decision tree (depth 6)"
model = DecisionTreeRegressor(max_depth=6, random_state=SEED).fit(X_train, y_train)
fitted[name] = model
for dataset, X, y in [("train", X_train, y_train), ("validation", X_val, y_val)]:
    pred = model.predict(X)
    results.loc[len(results)] = [
        today,
        name,
        dataset,
        metrics.mean_absolute_error(y, pred),
        metrics.median_absolute_error(y, pred),
        metrics.mean_squared_error(y, pred),
        metrics.root_mean_squared_error(y, pred),
        metrics.mean_absolute_percentage_error(y, pred),
        metrics.max_error(y, pred),
        metrics.r2_score(y, pred),
        metrics.explained_variance_score(y, pred),
        "",
    ]
results.round(3)

# %% [markdown]
# **2.4** A random forest, this time with 20 trees.

# %%
name = "Random forest (20 trees)"
model = RandomForestRegressor(n_estimators=20, random_state=SEED).fit(X_train, y_train)
fitted[name] = model
for dataset, X, y in [("train", X_train, y_train), ("validation", X_val, y_val)]:
    pred = model.predict(X)
    results.loc[len(results)] = [
        today,
        name,
        dataset,
        metrics.mean_absolute_error(y, pred),
        metrics.median_absolute_error(y, pred),
        metrics.mean_squared_error(y, pred),
        metrics.root_mean_squared_error(y, pred),
        metrics.mean_absolute_percentage_error(y, pred),
        metrics.max_error(y, pred),
        metrics.r2_score(y, pred),
        metrics.explained_variance_score(y, pred),
        "",
    ]
results.round(3)

# %% [markdown]
# **2.5** Gradient boosting with its default settings.

# %%
name = "Gradient boosting"
model = HistGradientBoostingRegressor(random_state=SEED).fit(X_train, y_train)
fitted[name] = model
for dataset, X, y in [("train", X_train, y_train), ("validation", X_val, y_val)]:
    pred = model.predict(X)
    results.loc[len(results)] = [
        today,
        name,
        dataset,
        metrics.mean_absolute_error(y, pred),
        metrics.median_absolute_error(y, pred),
        metrics.mean_squared_error(y, pred),
        metrics.root_mean_squared_error(y, pred),
        metrics.mean_absolute_percentage_error(y, pred),
        metrics.max_error(y, pred),
        metrics.r2_score(y, pred),
        metrics.explained_variance_score(y, pred),
        "",
    ]
results.round(3)

# %% [markdown]
# **2.6** The last model predicts the average rate of the 25 most similar loans in the
# training rows.

# %%
name = "k-nearest neighbors (k=25)"
model = KNeighborsRegressor(n_neighbors=25).fit(X_train, y_train)
fitted[name] = model
for dataset, X, y in [("train", X_train, y_train), ("validation", X_val, y_val)]:
    pred = model.predict(X)
    results.loc[len(results)] = [
        today,
        name,
        dataset,
        metrics.mean_absolute_error(y, pred),
        metrics.median_absolute_error(y, pred),
        metrics.mean_squared_error(y, pred),
        metrics.root_mean_squared_error(y, pred),
        metrics.mean_absolute_percentage_error(y, pred),
        metrics.max_error(y, pred),
        metrics.r2_score(y, pred),
        metrics.explained_variance_score(y, pred),
        "",
    ]
results.round(3)

# %% [markdown]
# **2.7** We sort the table by `r2` on the validation rows and take the top model as the
# winner. This is the only place where we choose. The table keeps every row ever added,
# so if you change a model, restart the kernel and run all cells before choosing here.

# %%
# The winner is chosen here, on validation only.
validation = results[results["dataset"] == "validation"]
validation = validation.sort_values("r2", ascending=False)
best = validation["model"].iloc[0]
validation.round(3)

# %% [markdown]
# ----------------------------------------------------------------------
# # STAGE 3. THE LOG, AND THE TEST SET ONCE
# ----------------------------------------------------------------------

# %% [markdown]
# **3.1** We save the table to `reports/experiments-int-rate.csv`, the log for this
# target.

# %%
(PROJECT / "reports").mkdir(exist_ok=True)
results.round(3).to_csv(PROJECT / "reports" / "experiments-int-rate.csv", index=False)
results.round(3)

# %% [markdown]
# **3.2** Now, and only now, we score the winner of step 2.7 on the test set. Its scores
# go into the log as one test row, and the file is saved again.

# %%
# best comes from step 2.7. Do not choose again here.
X_test = preprocess(test)
y_test = test["int_rate"]
pred = fitted[best].predict(X_test)
results.loc[len(results)] = [
    today,
    best,
    "test",
    metrics.mean_absolute_error(y_test, pred),
    metrics.median_absolute_error(y_test, pred),
    metrics.mean_squared_error(y_test, pred),
    metrics.root_mean_squared_error(y_test, pred),
    metrics.mean_absolute_percentage_error(y_test, pred),
    metrics.max_error(y_test, pred),
    metrics.r2_score(y_test, pred),
    metrics.explained_variance_score(y_test, pred),
    "test opened once",
]
results.round(3).to_csv(PROJECT / "reports" / "experiments-int-rate.csv", index=False)
results.round(3)

# %% [markdown]
# What each column of the table measures, in the words of the scikit-learn
# documentation, version 1.9 (the one in your `.venv`). As a rule, a function whose name
# ends in `_score` gives "a value to maximize, the higher the better", and one that ends
# in `_error` or `_loss` gives "a value to minimize, the lower the better" ([user guide
# 3.4.3.2.1][1]).
#
# - `mae`: "The mean_absolute_error function computes mean absolute error, a risk metric
#   corresponding to the expected value of the absolute error loss" ([user guide
#   3.4.6.2][2])
# - `median_ae`: "The median_absolute_error is particularly interesting because it is
#   robust to outliers. The loss is calculated by taking the median of all absolute
#   differences between the target and the prediction." ([user guide 3.4.6.6][3])
# - `mse`: "The mean_squared_error function computes mean squared error, a risk metric
#   corresponding to the expected value of the squared (quadratic) error or loss."
#   ([user guide 3.4.6.3][4])
# - `rmse`: "Taking the square root of the MSE, called the root mean squared error
#   (RMSE), is another common metric that provides a measure in the same units as the
#   target variable." ([user guide 3.4.6.3][4])
# - `mape`: "The idea of this metric is to be sensitive to relative errors. It is for
#   example not changed by a global scaling of the target variable." and "the percentage
#   in the range [0, 100] is converted to a relative value in the range [0, 1] by
#   dividing by 100. Thus, an error of 200% corresponds to a relative error of 2."
#   ([user guide 3.4.6.5][5])
# - `max_error`: "a metric that captures the worst case error between the predicted
#   value and the true value" ([user guide 3.4.6.7][6])
# - `r2`: "It represents the proportion of variance (of y) that has been explained by
#   the independent variables in the model." and "Best possible score is 1.0 and it can
#   be negative (because the model can be arbitrarily worse). A constant model that
#   always predicts the expected (average) value of y, disregarding the input features,
#   would get an R^2 score of 0.0." ([user guide 3.4.6.1][7])
# - `explained_variance`: "The best possible score is 1.0, lower values are worse." and
#   "the explained variance score does not account for systematic offset in the
#   prediction. For this reason, the R^2 score, the coefficient of determination should
#   be preferred in general." ([user guide 3.4.6.8][8])
#
# [1]: https://scikit-learn.org/1.9/modules/model_evaluation.html#scoring-adapt-metric
# [2]: https://scikit-learn.org/1.9/modules/model_evaluation.html#mean-absolute-error
# [3]: https://scikit-learn.org/1.9/modules/model_evaluation.html#median-absolute-error
# [4]: https://scikit-learn.org/1.9/modules/model_evaluation.html#mean-squared-error
# [5]: https://scikit-learn.org/1.9/modules/model_evaluation.html#regression-metrics
# [6]: https://scikit-learn.org/1.9/modules/model_evaluation.html#max-error
# [7]: https://scikit-learn.org/1.9/modules/model_evaluation.html#r2-score
# [8]:
# https://scikit-learn.org/1.9/modules/model_evaluation.html#explained-variance-score

# %% [markdown]
# **3.3** In Source Control, click the + next to `reports/experiments-int-rate.csv`
# only, commit it with the message `Add the regression experiment log`, and then click
# Sync Changes.

# %% [markdown]
# Like those of the classification file, the chart below is for the development phase
# only: it looks back at the models we compared and at how the winner did, and nothing
# is chosen from it. It is not a business report and not a way to watch a model in
# production. It follows the same [IBCS](https://www.ibcs.com/standards/) rules: the
# message first, then who, what and when; the three sets in gray, darker for the later
# periods; every bar carries its value instead of axes and gridlines; and panels share a
# scale where they can: `mae`, `median_ae` and `rmse` share one, `r2` and
# `explained_variance` another, and `mse`, `mape` and `max_error` have their own.

# %% [markdown]
# **3.4** Last, the log as a chart, one panel per score, drawn here and saved as
# `reports/generated/regression-interest-rate-metrics.html`, which opens in any browser.

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

log = pd.read_csv(PROJECT / "reports" / "experiments-int-rate.csv")
panels = log.columns[3:-1]
gray = {"train": "#C0C0C0", "validation": "#808080", "test": "#404040"}
points = [0, log[["mae", "median_ae", "rmse"]].max().max() * 1.3]
ranges = {
    "mae": points,
    "median_ae": points,
    "mse": [0, log["mse"].max() * 1.3],
    "rmse": points,
    "mape": [0, log["mape"].max() * 1.3],
    "max_error": [0, log["max_error"].max() * 1.3],
    "r2": [-0.3, 1.3],
    "explained_variance": [-0.3, 1.3],
}
names = {
    "mae": "Mean absolute error, points",
    "median_ae": "Median absolute error, points",
    "mse": "Mean squared error, points squared",
    "rmse": "Root mean squared error, points",
    "mape": "Mean absolute percentage error, share",
    "max_error": "Largest error, points",
    "r2": "R2, share of the variance explained",
    "explained_variance": "Explained variance, share",
}
lower = ["mae", "median_ae", "mse", "rmse", "mape", "max_error"]
panel_titles = []
for p in panels:
    better = "lower is better" if p in lower else "higher is better"
    panel_titles.append(f"<b>{names[p]}</b><br>{p}, {better}")
fig = make_subplots(
    rows=4,
    cols=2,
    shared_yaxes=True,
    subplot_titles=panel_titles,
    horizontal_spacing=0.25,
    vertical_spacing=0.08,
)
for i, panel in enumerate(panels):
    row, col = i // 2 + 1, i % 2 + 1
    for dataset, color in gray.items():
        part = log[log["dataset"] == dataset]
        bar = go.Bar(
            x=part[panel],
            y=part["model"],
            name=dataset,
            orientation="h",
            marker_color=color,
            texttemplate="%{x:.3f}",
            textposition="outside",
            textfont_size=10,
            constraintext="none",
            cliponaxis=False,
            showlegend=i == 0,
        )
        fig.add_trace(bar, row=row, col=col)
    fig.update_xaxes(range=ranges[panel], row=row, col=col)
fig.update_xaxes(
    showticklabels=False, ticks="", showline=False, zeroline=True, zerolinecolor="black"
)
fig.update_yaxes(
    showticklabels=True,
    categoryorder="array",
    categoryarray=list(validation["model"]),
    autorange="reversed",
    ticks="",
    showline=False,
)
fig.update_annotations(font_size=12)
chosen = validation["r2"].iloc[0]
tested = log.loc[log["dataset"] == "test", "r2"].iloc[0]
title_lines = [
    f"<b>{best} was chosen on validation (r2 {chosen:.3f})"
    f" and scores {tested:.3f} on test</b>",
    "Lending Club loans, interest rate models;"
    " points are percentage points of the rate",
    "<b>Eight scores</b> per model and set, sorted by validation r2;"
    " mse, mape and max_error on their own scales",
    "Train 2014-01 to 2015-12, validation 2016-01 to 2016-06,"
    " test 2016-07 to 2018-11",
]
fig.update_layout(
    title={"text": "<br>".join(title_lines), "x": 0.01, "y": 0.98, "yanchor": "top"},
    template="simple_white",
    font={"family": "Arial", "size": 11, "color": "black"},
    legend={"orientation": "h", "x": 0, "y": 1.04, "yanchor": "bottom"},
    height=1200,
    margin={"t": 150, "b": 20},
)
fig.write_html(
    PROJECT / "reports" / "generated" / "regression-interest-rate-metrics.html"
)
fig.show()

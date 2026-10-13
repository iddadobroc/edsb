# %%
"""Classes 6 and 7, classification: will a loan default?

We split the loans by date, profile the whole dataset and each set, prepare the data,
compare seven models in one table and score the winner on the test set once, at the end.
The regression version of the same steps is class_week06_week07_regression.py.

Functions defined here:
    preprocess: prepares any set with what was learned from the training rows.
"""

# %% [markdown]
# ### Classes 6 & 7. Model selection and evaluation metrics
# Enterprise Data Science Bootcamp, NOVA IMS, 2026-27
#
# João Azambuja

# %% [markdown]
# ----------------------------------------------------------------------
# # STAGE 0. GET READY
# ----------------------------------------------------------------------

# %% [markdown]
# **0.1** Copy `requirements.txt` into your `edsb` folder and the four class files into
# its `notebooks` folder, next to `explore.py`. If you are asked to replace
# `requirements.txt`, say yes: it is the Class 3 file with one more package, `plotly`.
# Open the `edsb` folder in VS Code, then this file, and check that the bar at the
# bottom shows `.venv`. If it does not, press Ctrl+Shift+P (Cmd+Shift+P on a Mac) and
# choose Python: Select Interpreter.

# %% [markdown]
# **0.2** Click in the cell below and press Shift+Enter. Like `explore.py`, it first
# finds the project folder, the first folder above this file that has `data/raw`
# inside, and prints it: every path in this file starts from there. Then it checks
# Python, your `.venv`, the data file and four packages, so you should see `True` three
# times and `OK` four times. If `plotly` says `MISSING`, run
# `python -m pip install -r requirements.txt` in the terminal, with `(.venv)` at the
# start of the line, and run this cell again.

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
# # STAGE 1. THREE SETS
# ----------------------------------------------------------------------
# Loans issued up to June 2016 are ours to work with. The later ones are the test set.
# We profile it in step 1.3, but we choose nothing from it, and we score a model on it
# only once, at the end of the class.

# %% [markdown]
# **1.1** We load the loans whose outcome we already know. The file marks a repaid loan
# with 1, so we flip it: from here on, `default` is 1 when a loan was not repaid.

# %%
from datetime import date

import pandas as pd
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv(CSV)
df = df.dropna(subset=["loan_status"])
df["issue_d"] = pd.to_datetime(df["issue_d"])
# loan_status is 1 for fully paid. We flip it so that default = 1.
df["default"] = (df["loan_status"] == 0).astype(int)
df.shape

# %% [markdown]
# **1.2** To keep things fast we take 20,000 of those loans. The last six months,
# January to June 2016, become the validation set, and the earlier loans are the
# training set.

# %%
SEED = 42
# We leave out the lender's pricing: int_rate, sub_grade and installment.
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
y_train, y_val = train["default"], val["default"]

sets = []
for name, part in [("train", train), ("validation", val)]:
    sets.append(
        {
            "set": name,
            "loans": len(part),
            "first": part["issue_d"].min(),
            "last": part["issue_d"].max(),
            "default_rate": part["default"].mean(),
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
pd.DataFrame(sets).round(3)

# %% [markdown]
# **1.3** We profile the whole dataset and then each of the three sets, so we can
# compare them. This takes about half a minute on a fast laptop and a few minutes on a
# slow one, and the cell prints nothing while it works. The four files go to
# `reports/generated/`: open them in your browser (right-click one, then Reveal in File
# Explorer or Finder) and ignore the pink warning about the package name, as in Class 3.

# %%
from ydata_profiling import ProfileReport

(PROJECT / "reports" / "generated").mkdir(parents=True, exist_ok=True)
for name, part in [("all", df), ("train", train), ("validation", val), ("test", test)]:
    rows = part[NUMERIC + CATEGORICAL + ["default"]].copy()
    profile = ProfileReport(
        rows, title=f"Classes 6 and 7, classification: {name}", progress_bar=False
    )
    profile.to_file(PROJECT / "reports" / "generated" / f"profile-default-{name}.html")

# %% [markdown]
# **1.4** `preprocess` learns from the training rows where to cut extreme values (the
# 1st and 99th percentiles), how to scale each number and which categories exist, and
# then prepares any set the same way. The table shows each cut next to the largest value
# in training and in validation.

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
pd.DataFrame(
    {
        "low": low,
        "high": high,
        "largest_in_train": train[NUMERIC].max(),
        "largest_in_validation": val[NUMERIC].max(),
    }
).round(1)

# %% [markdown]
# ----------------------------------------------------------------------
# # STAGE 2. SEVEN MODELS, ONE TABLE
# ----------------------------------------------------------------------
# Each cell below fits one model on the training rows and adds two rows to the table:
# its scores on the training rows and on the validation rows.
#
# The first five scores treat a loan as flagged when its probability is 0.5 or more.
# `accuracy` is the share of loans called right, and `balanced_accuracy` averages that
# share over the defaults and the paid loans. `precision` is the share of flagged loans
# that defaulted, `recall` the share of defaults we flagged, and `f1` combines the two.
# `roc_auc` and `pr_auc` use the probabilities themselves: how well the model ranks the
# defaults above the paid loans. A model that guesses gets 0.5 on `roc_auc` but the
# default rate of the set on `pr_auc`. `log_loss` and `brier` measure how far the
# probabilities are from what happened, so for these two lower is better.

# %% [markdown]
# **2.1** We create the table and give it its first rows: a model that does nothing and
# gives every loan the default rate of the training rows. We read the other models
# against it: on `roc_auc` they all beat it, but on `accuracy` it is hard to beat,
# because it flags no loan and so is right on every paid loan.

# %%
today = date.today().isoformat()
columns = [
    "date",
    "model",
    "dataset",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "log_loss",
    "brier",
    "note",
]
results = pd.DataFrame(columns=columns)
fitted = {}

name = "Baseline (base rate)"
model = DummyClassifier(strategy="prior").fit(X_train, y_train)
fitted[name] = model
for dataset, X, y in [("train", X_train, y_train), ("validation", X_val, y_val)]:
    p = model.predict_proba(X)[:, 1]
    flagged = p >= 0.5
    results.loc[len(results)] = [
        today,
        name,
        dataset,
        metrics.accuracy_score(y, flagged),
        metrics.balanced_accuracy_score(y, flagged),
        metrics.precision_score(y, flagged, zero_division=0),
        metrics.recall_score(y, flagged),
        metrics.f1_score(y, flagged, zero_division=0),
        metrics.roc_auc_score(y, p),
        metrics.average_precision_score(y, p),
        metrics.log_loss(y, p),
        metrics.brier_score_loss(y, p),
        "",
    ]
results.round(3)

# %% [markdown]
# **2.2** A logistic regression. The cell repeats the steps of the one above with a real
# model and adds two more rows.

# %%
name = "Logistic regression"
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
fitted[name] = model
for dataset, X, y in [("train", X_train, y_train), ("validation", X_val, y_val)]:
    p = model.predict_proba(X)[:, 1]
    flagged = p >= 0.5
    results.loc[len(results)] = [
        today,
        name,
        dataset,
        metrics.accuracy_score(y, flagged),
        metrics.balanced_accuracy_score(y, flagged),
        metrics.precision_score(y, flagged, zero_division=0),
        metrics.recall_score(y, flagged),
        metrics.f1_score(y, flagged, zero_division=0),
        metrics.roc_auc_score(y, p),
        metrics.average_precision_score(y, p),
        metrics.log_loss(y, p),
        metrics.brier_score_loss(y, p),
        "",
    ]
results.round(3)

# %% [markdown]
# **2.3** A decision tree limited to five levels, so it asks at most five questions
# about a loan.

# %%
name = "Decision tree (depth 5)"
model = DecisionTreeClassifier(max_depth=5, random_state=SEED).fit(X_train, y_train)
fitted[name] = model
for dataset, X, y in [("train", X_train, y_train), ("validation", X_val, y_val)]:
    p = model.predict_proba(X)[:, 1]
    flagged = p >= 0.5
    results.loc[len(results)] = [
        today,
        name,
        dataset,
        metrics.accuracy_score(y, flagged),
        metrics.balanced_accuracy_score(y, flagged),
        metrics.precision_score(y, flagged, zero_division=0),
        metrics.recall_score(y, flagged),
        metrics.f1_score(y, flagged, zero_division=0),
        metrics.roc_auc_score(y, p),
        metrics.average_precision_score(y, p),
        metrics.log_loss(y, p),
        metrics.brier_score_loss(y, p),
        "",
    ]
results.round(3)

# %% [markdown]
# **2.4** A random forest: 100 trees, each grown on a different random sample of the
# training rows, with their answers averaged.

# %%
name = "Random forest (100 trees)"
model = RandomForestClassifier(n_estimators=100, random_state=SEED).fit(
    X_train, y_train
)
fitted[name] = model
for dataset, X, y in [("train", X_train, y_train), ("validation", X_val, y_val)]:
    p = model.predict_proba(X)[:, 1]
    flagged = p >= 0.5
    results.loc[len(results)] = [
        today,
        name,
        dataset,
        metrics.accuracy_score(y, flagged),
        metrics.balanced_accuracy_score(y, flagged),
        metrics.precision_score(y, flagged, zero_division=0),
        metrics.recall_score(y, flagged),
        metrics.f1_score(y, flagged, zero_division=0),
        metrics.roc_auc_score(y, p),
        metrics.average_precision_score(y, p),
        metrics.log_loss(y, p),
        metrics.brier_score_loss(y, p),
        "",
    ]
results.round(3)

# %% [markdown]
# **2.5** Gradient boosting, which adds trees one after another, each one correcting
# what the earlier ones got wrong.

# %%
name = "Gradient boosting"
model = HistGradientBoostingClassifier(random_state=SEED).fit(X_train, y_train)
fitted[name] = model
for dataset, X, y in [("train", X_train, y_train), ("validation", X_val, y_val)]:
    p = model.predict_proba(X)[:, 1]
    flagged = p >= 0.5
    results.loc[len(results)] = [
        today,
        name,
        dataset,
        metrics.accuracy_score(y, flagged),
        metrics.balanced_accuracy_score(y, flagged),
        metrics.precision_score(y, flagged, zero_division=0),
        metrics.recall_score(y, flagged),
        metrics.f1_score(y, flagged, zero_division=0),
        metrics.roc_auc_score(y, p),
        metrics.average_precision_score(y, p),
        metrics.log_loss(y, p),
        metrics.brier_score_loss(y, p),
        "",
    ]
results.round(3)

# %% [markdown]
# **2.6** The next model gives each loan the default rate of the 25 most similar loans
# in the training rows.

# %%
name = "k-nearest neighbors (k=25)"
model = KNeighborsClassifier(n_neighbors=25).fit(X_train, y_train)
fitted[name] = model
for dataset, X, y in [("train", X_train, y_train), ("validation", X_val, y_val)]:
    p = model.predict_proba(X)[:, 1]
    flagged = p >= 0.5
    results.loc[len(results)] = [
        today,
        name,
        dataset,
        metrics.accuracy_score(y, flagged),
        metrics.balanced_accuracy_score(y, flagged),
        metrics.precision_score(y, flagged, zero_division=0),
        metrics.recall_score(y, flagged),
        metrics.f1_score(y, flagged, zero_division=0),
        metrics.roc_auc_score(y, p),
        metrics.average_precision_score(y, p),
        metrics.log_loss(y, p),
        metrics.brier_score_loss(y, p),
        "",
    ]
results.round(3)

# %% [markdown]
# **2.7** The same tree with no depth limit: it keeps asking questions until it gets
# every training loan right. Compare its two rows with those of the depth-5 tree.

# %%
name = "Decision tree (no depth limit)"
model = DecisionTreeClassifier(max_depth=None, random_state=SEED).fit(X_train, y_train)
fitted[name] = model
for dataset, X, y in [("train", X_train, y_train), ("validation", X_val, y_val)]:
    p = model.predict_proba(X)[:, 1]
    flagged = p >= 0.5
    results.loc[len(results)] = [
        today,
        name,
        dataset,
        metrics.accuracy_score(y, flagged),
        metrics.balanced_accuracy_score(y, flagged),
        metrics.precision_score(y, flagged, zero_division=0),
        metrics.recall_score(y, flagged),
        metrics.f1_score(y, flagged, zero_division=0),
        metrics.roc_auc_score(y, p),
        metrics.average_precision_score(y, p),
        metrics.log_loss(y, p),
        metrics.brier_score_loss(y, p),
        "",
    ]
results.round(3)

# %% [markdown]
# **2.8** We keep the validation rows, sort them by ROC-AUC and take the top model as
# the winner. This is the only place where we choose. Two models with the same score to
# three decimals are a tie, not a winner, but the code still takes the top row.
# The table keeps every row ever added, so if you change a model, restart the kernel
# and run all cells before choosing here.

# %%
# The winner is chosen here, on validation only.
validation = results[results["dataset"] == "validation"]
validation = validation.sort_values("roc_auc", ascending=False)
best = validation["model"].iloc[0]
validation.round(3)

# %% [markdown]
# **2.9** The winner gives each loan a probability, and to act on it we need a
# threshold. Here are its precision and recall at two thresholds, 0.5 and 0.2.

# %%
p = fitted[best].predict_proba(X_val)[:, 1]
for threshold in [0.5, 0.2]:
    flagged = p >= threshold
    precision = metrics.precision_score(y_val, flagged)
    recall = metrics.recall_score(y_val, flagged)
    print(
        f"{best}, threshold {threshold}:",
        f"precision {precision:.2f}, recall {recall:.2f}, flagged {flagged.mean():.1%}",
    )

# %% [markdown]
# ----------------------------------------------------------------------
# # STAGE 3. THE LOG, AND THE TEST SET ONCE
# ----------------------------------------------------------------------

# %% [markdown]
# **3.1** We save the table to `reports/experiments-default.csv`. This file is our
# experiment log.

# %%
(PROJECT / "reports").mkdir(exist_ok=True)
results.round(3).to_csv(PROJECT / "reports" / "experiments-default.csv", index=False)
results.round(3)

# %% [markdown]
# **3.2** Now, and only now, we score the winner of step 2.8 on the test set. Its scores
# go into the log as one test row, and the file is saved again.

# %%
# best comes from step 2.8. Do not choose again here.
X_test = preprocess(test)
y_test = test["default"]
p = fitted[best].predict_proba(X_test)[:, 1]
flagged = p >= 0.5
results.loc[len(results)] = [
    today,
    best,
    "test",
    metrics.accuracy_score(y_test, flagged),
    metrics.balanced_accuracy_score(y_test, flagged),
    metrics.precision_score(y_test, flagged, zero_division=0),
    metrics.recall_score(y_test, flagged),
    metrics.f1_score(y_test, flagged, zero_division=0),
    metrics.roc_auc_score(y_test, p),
    metrics.average_precision_score(y_test, p),
    metrics.log_loss(y_test, p),
    metrics.brier_score_loss(y_test, p),
    "test opened once",
]
results.round(3).to_csv(PROJECT / "reports" / "experiments-default.csv", index=False)
results.round(3)

# %% [markdown]
# What each column of the table measures, in the words of the scikit-learn
# documentation, version 1.9 (the one in your `.venv`). As a rule, a function whose name
# ends in `_score` gives "a value to maximize, the higher the better", and one that ends
# in `_error` or `_loss` gives "a value to minimize, the lower the better" ([user guide
# 3.4.3.2.1][1]).
#
# - `accuracy`: "The accuracy_score function computes the accuracy, either the fraction
#   (default) or the count (normalize=False) of correct predictions." ([user guide
#   3.4.4.2][2])
# - `balanced_accuracy`: "In the binary case, balanced accuracy is equal to the
#   arithmetic mean of sensitivity (true positive rate) and specificity (true negative
#   rate)" and "If the classifier performs equally well on either class, this term
#   reduces to the conventional accuracy" ([user guide 3.4.4.4][9])
# - `precision`: "precision is the ability of the classifier not to label as positive a
#   sample that is negative" ([user guide 3.4.4.9][3])
# - `recall`: "The recall is the ratio tp / (tp + fn) where tp is the number of true
#   positives and fn the number of false negatives. The recall is intuitively the
#   ability of the classifier to find all the positive samples. The best value is 1 and
#   the worst value is 0." ([recall_score][4])
# - `f1`: "The F1 score can be interpreted as a harmonic mean of the precision and
#   recall, where an F1 score reaches its best value at 1 and worst score at 0."
#   ([f1_score][5])
# - `roc_auc`: "It is created by plotting the fraction of true positives out of the
#   positives (TPR = true positive rate) vs. the fraction of false positives out of the
#   negatives (FPR = false positive rate), at various threshold settings." and "The
#   roc_auc_score function, denoted by ROC-AUC or AUROC, computes the area under the ROC
#   curve. By doing so, the curve information is summarized in one number." ([user guide
#   3.4.4.15][6])
# - `pr_auc`, the average precision: "The average_precision_score function computes the
#   average precision (AP) from prediction scores. The value is between 0 and 1 and
#   higher is better." and "With random predictions, the AP is the fraction of
#   positive samples." ([user guide 3.4.4.9][3])
# - `log_loss`: "Log loss, also called logistic regression loss or cross-entropy loss,
#   is defined on probability estimates." and "can be used to evaluate the probability
#   outputs (predict_proba) of a classifier instead of its discrete predictions." ([user
#   guide 3.4.4.12][7])
# - `brier`: "The brier_score_loss function computes the Brier score for binary and
#   multiclass probabilistic predictions and is equivalent to the mean squared error."
#   and "the lower the value the better the probability estimates are (the mean squared
#   difference is smaller)" ([user guide 3.4.4.18][8])
#
# [1]: https://scikit-learn.org/1.9/modules/model_evaluation.html#scoring-adapt-metric
# [2]: https://scikit-learn.org/1.9/modules/model_evaluation.html#accuracy-score
# [3]: https://scikit-learn.org/1.9/modules/model_evaluation.html#classification-metrics
# [4]: https://scikit-learn.org/1.9/modules/generated/sklearn.metrics.recall_score.html
# [5]: https://scikit-learn.org/1.9/modules/generated/sklearn.metrics.f1_score.html
# [6]: https://scikit-learn.org/1.9/modules/model_evaluation.html#roc-metrics
# [7]: https://scikit-learn.org/1.9/modules/model_evaluation.html#log-loss
# [8]: https://scikit-learn.org/1.9/modules/model_evaluation.html#brier-score-loss
# [9]:
# https://scikit-learn.org/1.9/modules/model_evaluation.html#balanced-accuracy-score

# %% [markdown]
# **3.3** In Source Control, click the + next to `reports/experiments-default.csv` only,
# commit it with the message `Add the classification experiment log`, and then click
# Sync Changes.

# %% [markdown]
# The two charts below are for the development phase only: they look back at the models
# we compared and at how the winner did, and nothing is chosen from them. They are not a
# business report and not a way to watch a model in production. We still draw them by
# the [IBCS](https://www.ibcs.com/standards/) rules for business charts: the message
# first, then who, what and when, and the three sets in gray, darker for the later
# periods. In the bar chart every bar carries its value instead of axes and gridlines,
# and all panels share one scale, except `log_loss`.

# %% [markdown]
# **3.4** The log as a chart, one panel per score, drawn here and saved as
# `reports/generated/classification-loan-default-metrics.html`, which opens in any
# browser.

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

log = pd.read_csv(PROJECT / "reports" / "experiments-default.csv")
panels = log.columns[3:-1]
gray = {"train": "#C0C0C0", "validation": "#808080", "test": "#404040"}
ranges = {panel: [0, 1.3] for panel in panels}
ranges["log_loss"] = [0, log["log_loss"].max() * 1.3]
names = {
    "accuracy": "Accuracy, threshold 0.5",
    "balanced_accuracy": "Balanced accuracy, threshold 0.5",
    "precision": "Precision, threshold 0.5",
    "recall": "Recall, threshold 0.5",
    "f1": "F1 score, threshold 0.5",
    "roc_auc": "ROC AUC",
    "pr_auc": "PR AUC (average precision)",
    "log_loss": "Log loss",
    "brier": "Brier score",
}
lower = ["log_loss", "brier"]
panel_titles = []
for p in panels:
    better = "lower is better" if p in lower else "higher is better"
    panel_titles.append(f"<b>{names[p]}</b><br>{p}, {better}")
fig = make_subplots(
    rows=3,
    cols=3,
    shared_yaxes=True,
    subplot_titles=panel_titles,
    horizontal_spacing=0.25,
    vertical_spacing=0.09,
)
for i, panel in enumerate(panels):
    row, col = i // 3 + 1, i % 3 + 1
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
chosen = validation["roc_auc"].iloc[0]
tested = log.loc[log["dataset"] == "test", "roc_auc"].iloc[0]
title_lines = [
    f"<b>{best} was chosen on validation (roc_auc {chosen:.3f})"
    f" and scores {tested:.3f} on test</b>",
    "Lending Club loans, loan default models",
    "<b>Nine scores</b> per model and set, sorted by validation roc_auc;"
    " log_loss on its own scale",
    "Train 2014-01 to 2015-12, validation 2016-01 to 2016-06,"
    " test 2016-07 to 2018-11",
]
fig.update_layout(
    title={"text": "<br>".join(title_lines), "x": 0.01, "y": 0.98, "yanchor": "top"},
    template="simple_white",
    font={"family": "Arial", "size": 11, "color": "black"},
    legend={"orientation": "h", "x": 0, "y": 1.04, "yanchor": "bottom"},
    height=1100,
    margin={"t": 150, "b": 20},
)
fig.write_html(
    PROJECT / "reports" / "generated" / "classification-loan-default-metrics.html"
)
fig.show()

# %% [markdown]
# **3.5** Last, the ROC curve of the winner on each set: the share of defaults it
# catches (true positive rate) against the share of paid loans it flags by mistake
# (false positive rate), at every threshold. The dotted diagonal is a coin, and the area
# under each curve is the `roc_auc` of that row in the table. The chart is also saved as
# `reports/generated/classification-loan-default-roc-curves.html`.

# %%
fig = go.Figure()
areas = {}
for dataset, X, y in [
    ("train", X_train, y_train),
    ("validation", X_val, y_val),
    ("test", X_test, y_test),
]:
    p = fitted[best].predict_proba(X)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y, p)
    areas[dataset] = metrics.roc_auc_score(y, p)
    line = {"color": gray[dataset], "width": 2}
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=dataset, line=line))
fig.add_shape(
    type="line",
    x0=0,
    y0=0,
    x1=1,
    y1=1,
    line={"dash": "dot", "width": 1, "color": "gray"},
)
fig.update_xaxes(
    range=[0, 1], constrain="domain", title="paid loans flagged (false positive rate)"
)
fig.update_yaxes(
    range=[0, 1],
    constrain="domain",
    scaleanchor="x",
    title="defaults caught (true positive rate)",
)
title_lines = [
    f"<b>{best}: roc_auc {areas['train']:.3f} on train,"
    f" {areas['validation']:.3f} on validation, {areas['test']:.3f} on test</b>",
    "Lending Club loans, the loan default model chosen in step 2.8",
    "<b>ROC curve</b>, defaults caught against paid loans flagged, at every threshold",
    "Train 2014-01 to 2015-12, validation 2016-01 to 2016-06,"
    " test 2016-07 to 2018-11",
]
fig.update_layout(
    title={"text": "<br>".join(title_lines), "x": 0.01, "y": 0.98, "yanchor": "top"},
    template="simple_white",
    font={"family": "Arial", "size": 11, "color": "black"},
    legend={"x": 0.98, "xanchor": "right", "y": 0.02, "yanchor": "bottom"},
    width=760,
    height=800,
    margin={"t": 120},
)
fig.write_html(
    PROJECT / "reports" / "generated" / "classification-loan-default-roc-curves.html"
)
fig.show()

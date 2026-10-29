"""
STEP 1 + 2: Explore the data, train a Random Forest, and save it.

Run:  python train.py

What this script does:
  1. Loads the "Online Shoppers Purchasing Intention" dataset (12,330 website visits)
  2. Prints a few simple facts about the data (exploration)
  3. Splits the data into training (80%) and testing (20%) parts
  4. Trains a Random Forest with 300 trees
  5. Checks how good the model is (accuracy, precision, recall)
  6. Saves the model + a small JSON file the API and web app will use
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # draw charts to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).parent
DATA_CSV = ROOT / "data" / "online_shoppers_intention.csv"
MODEL_DIR = ROOT / "model"
REPORT_DIR = ROOT / "reports"

# Columns with text values -> must be turned into numbers (one-hot encoding)
CATEGORICAL = ["Month", "VisitorType"]

# Columns that are already numbers
NUMERIC = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
    "OperatingSystems", "Browser", "Region", "TrafficType", "Weekend",
]
TARGET = "Revenue"  # y: did the visit end with a purchase?

MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul",
               "Aug", "Sep", "Oct", "Nov", "Dec"]


def to_int_bool(series: pd.Series) -> pd.Series:
    """Turn True/False (or 'TRUE'/'FALSE') into 1/0."""
    return series.astype(str).str.strip().str.lower().isin(["true", "1"]).astype(int)


def load_data() -> pd.DataFrame:
    """Load the CSV if you already have it, otherwise download it from UCI."""
    if DATA_CSV.exists():
        print(f"Loading data from {DATA_CSV}")
        df = pd.read_csv(DATA_CSV)
    else:
        print("Downloading the dataset from the UCI repository...")
        from ucimlrepo import fetch_ucirepo

        ds = fetch_ucirepo(id=468)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        DATA_CSV.parent.mkdir(exist_ok=True)
        df.to_csv(DATA_CSV, index=False)
        print(f"Saved a copy to {DATA_CSV}")

    df["Weekend"] = to_int_bool(df["Weekend"])
    df[TARGET] = to_int_bool(df[TARGET])
    return df


def explore(df: pd.DataFrame) -> None:
    """STEP 1: look at the data before training anything."""
    print("\n=== STEP 1: Explore the data ===")
    print(f"Rows (n):            {len(df):,}")
    print(f"Input columns (p):   {len(CATEGORICAL) + len(NUMERIC)}")
    buy_rate = df[TARGET].mean()
    print(f"Visits that bought:  {df[TARGET].sum():,} ({buy_rate:.1%})")

    print("\nBuy rate by visitor type:")
    print(df.groupby("VisitorType")[TARGET].mean().map("{:.1%}".format).to_string())

    print("\nBuy rate by month:")
    by_month = df.groupby("Month")[TARGET].mean()
    by_month = by_month.reindex([m for m in MONTH_ORDER if m in by_month.index])
    print(by_month.map("{:.1%}".format).to_string())

    print("\nAverage behaviour: buyers vs. non-buyers")
    cols = ["ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues"]
    summary = df.groupby(TARGET)[cols].mean().T
    summary.columns = ["didn't buy", "bought"]
    print(summary.round(3).to_string())


def build_model() -> Pipeline:
    """The model = (turn text into numbers) + (Random Forest)."""
    prep = ColumnTransformer(
        [("text_to_numbers", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)],
        remainder="passthrough",          # numeric columns go through unchanged
        verbose_feature_names_out=False,
    )
    forest = RandomForestClassifier(
        n_estimators=300,          # B = number of trees
        max_features="sqrt",       # m = sqrt(p) features per question
        min_samples_leaf=3,        # stop splitting tiny groups (less overfitting)
        class_weight="balanced",   # buyers are rare (~15%), so give them more weight
        oob_score=True,            # free "out-of-bag" test (Formula 8)
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline([("prep", prep), ("forest", forest)])


def grouped_importance(model: Pipeline) -> list[dict]:
    """Feature importance, with one-hot columns (Month_Feb, Month_Mar...) added back together."""
    names = model.named_steps["prep"].get_feature_names_out()
    values = model.named_steps["forest"].feature_importances_
    totals: dict[str, float] = {}
    for name, value in zip(names, values):
        base = next((c for c in CATEGORICAL if name.startswith(c + "_")), name)
        totals[base] = totals.get(base, 0.0) + float(value)
    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    return [{"feature": k, "importance": round(v, 4)} for k, v in ranked]


def main() -> None:
    df = load_data()
    explore(df)

    X = df[CATEGORICAL + NUMERIC]
    y = df[TARGET]

    # stratify=y keeps the same 15% buyers in both parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\n=== STEP 2: Train the Random Forest ===")
    print(f"Training on {len(X_train):,} visits, testing on {len(X_test):,} visits...")
    model = build_model()
    model.fit(X_train, y_train)
    print("Done!")

    # ---- Evaluate ----
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    baseline = 1 - y_test.mean()  # a lazy model that always says "won't buy"

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "baseline_accuracy": baseline,
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "oob_score": model.named_steps["forest"].oob_score_,
    }
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print("\n=== How good is the model? ===")
    print(f"Accuracy:   {metrics['accuracy']:.1%}   (lazy 'always no' model: {baseline:.1%})")
    print(f"Precision:  {metrics['precision']:.1%}   (when it says 'will buy', how often is it right?)")
    print(f"Recall:     {metrics['recall']:.1%}   (of all real buyers, how many did it find?)")
    print(f"F1 score:   {metrics['f1']:.3f}")
    print(f"ROC AUC:    {metrics['roc_auc']:.3f}   (1.0 = perfect ranking, 0.5 = random guessing)")
    print(f"OOB score:  {metrics['oob_score']:.1%}   (accuracy on 'left-out' data, no test set needed)")
    print("\nConfusion matrix (test set):")
    print(f"  Real buyers found:          {tp}")
    print(f"  Real buyers missed:         {fn}")
    print(f"  Non-buyers wrongly flagged: {fp}")
    print(f"  Non-buyers correctly ruled out: {tn}")

    importance = grouped_importance(model)
    print("\n=== What matters most to the model? (feature importance) ===")
    for row in importance[:8]:
        bar = "#" * int(row["importance"] * 100)
        print(f"  {row['feature']:<26}{row['importance']:.3f}  {bar}")

    # ---- Save everything the API and web app need ----
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "shopper_forest.joblib", compress=3)

    # Typical seconds per page (used when the app sends page counts without durations)
    sec_per_page = {}
    for count_col, dur_col in [("Administrative", "Administrative_Duration"),
                               ("Informational", "Informational_Duration"),
                               ("ProductRelated", "ProductRelated_Duration")]:
        visited = df[df[count_col] > 0]
        sec_per_page[count_col] = round(float((visited[dur_col] / visited[count_col]).median()), 1)

    defaults = {c: float(df[c].median()) for c in NUMERIC}
    defaults.update({c: str(df[c].mode()[0]) for c in CATEGORICAL})

    months = [m for m in MONTH_ORDER if m in set(df["Month"])]
    meta = {
        "n_rows": int(len(df)),
        "buy_rate": round(float(df[TARGET].mean()), 4),
        "n_trees": int(model.named_steps["forest"].n_estimators),
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "importance": importance,
        "defaults": defaults,
        "sec_per_page": sec_per_page,
        "options": {"Month": months, "VisitorType": sorted(df["VisitorType"].unique().tolist())},
    }
    (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    # ---- Save a feature importance chart ----
    REPORT_DIR.mkdir(exist_ok=True)
    top = importance[:10][::-1]
    plt.figure(figsize=(8, 5))
    plt.barh([r["feature"] for r in top], [r["importance"] for r in top], color="#2E6A4F")
    plt.title("What the forest looks at most")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "feature_importance.png", dpi=150)

    print(f"\nSaved model to   {MODEL_DIR / 'shopper_forest.joblib'}")
    print(f"Saved info to    {MODEL_DIR / 'meta.json'}")
    print(f"Saved chart to   {REPORT_DIR / 'feature_importance.png'}")
    print("\nNext step: start the API with   uvicorn api:app --reload")


if __name__ == "__main__":
    main()

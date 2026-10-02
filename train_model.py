import json
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from app.filing_service import load_filings


def main():
    df = load_filings()

    # Company grouping is unavailable; related-company leakage remains possible.
    train_ids, test_ids = train_test_split(
        df.index.to_numpy(),
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    assert df.index.is_unique
    assert set(train_ids).isdisjoint(test_ids)
    assert set(train_ids) | set(test_ids) == set(df.index)

    train = df.loc[train_ids]
    test = df.loc[test_ids]

    reports_dir = Path(__file__).resolve().parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "split_ids.json").write_text(
        json.dumps(
            {
                "random_state": 42,
                "train_ids": train_ids.tolist(),
                "test_ids": test_ids.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=10000,
                min_df=2,
                ngram_range=(1, 1),
            ),
        ),
        (
            "classifier",
            LogisticRegression(max_iter=1000, random_state=42),
        ),
    ])

    print("Training")
    model.fit(train["filing_text"], train["label"])

    # This baseline always predicts the most common training label.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit([[0]] * len(train), train["label"])

    predictions = model.predict(test["filing_text"])
    baseline_predictions = baseline.predict([[0]] * len(test))

    assert len(predictions) == len(test)
    assert set(predictions).issubset({0, 1})
    assert len(baseline_predictions) == len(test)
    assert set(baseline_predictions).issubset({0, 1})
    results = {}

    for name, predicted in [
        ("logistic_regression", predictions),
        ("majority_baseline", baseline_predictions),
    ]:

        print(f"\n{name}:")
        print(classification_report(
            test["label"],
            predicted,
            labels=[0, 1],
            target_names=["no", "yes"],
            zero_division=0,
        ))

        results[name] = classification_report(
            test["label"],
            predicted,
            labels=[0, 1],
            target_names=["no", "yes"],
            output_dict=True,
            zero_division=0,
        )

    (reports_dir / "metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    display = ConfusionMatrixDisplay.from_predictions(
        test["label"],
        predictions,
        labels=[0, 1],
        display_labels=["no", "yes"],
        cmap="Blues",
    )
    display.ax_.set_title("Held-out filing classification")
    display.figure_.tight_layout()
    display.figure_.savefig(reports_dir / "confusion_matrix.png")
    plt.close(display.figure_)

    # Save evaluation details without copying the large filing texts.
    review = test[["label"]].rename(columns={"label": "actual_label"}).copy()
    review["predicted_label"] = predictions
    review["baseline_prediction"] = baseline_predictions
    review.to_csv(reports_dir / "test_predictions.csv")

    artifacts_dir = Path(__file__).resolve().parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    model_path = artifacts_dir / "filing_classifier.joblib"

    joblib.dump(model, model_path)
    (artifacts_dir / "label_mapping.json").write_text(
        json.dumps({"0": "no", "1": "yes"}, indent=2),
        encoding="utf-8",
    )

    # Check that saving and loading preserves predictions.
    restored_model = joblib.load(model_path)
    restored_predictions = restored_model.predict(test["filing_text"])
    assert (restored_predictions == predictions).all()


if __name__ == "__main__":
    main()

from functools import lru_cache
from pathlib import Path
import pandas as pd
import re
import joblib

DATA_PATH = Path(__file__).resolve().parents[1] / "Data" / "Final_Dataset.csv"


@lru_cache(maxsize=1)
def load_filings() -> pd.DataFrame:
    """Load and validate the local dataset once per process."""
    df = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)


    missing = {"Fillings", "Fraud"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.rename(columns={"Fillings": "filing_text"})
    df["record_id"] = [
        f"filing_{number:03d}" for number in range(1, len(df) + 1)
    ]

    if df.empty:
        raise ValueError("The dataset contains no records.")

    empty_text = df["filing_text"].str.strip().eq("")
    if empty_text.any():
        ids = df.loc[empty_text, "record_id"].tolist()
        raise ValueError(f"Empty filing text in records: {ids}")

    labels = df["Fraud"].str.strip().str.lower()
    invalid_labels = ~labels.isin(["yes", "no"])
    if invalid_labels.any():
        ids = df.loc[invalid_labels, "record_id"].tolist()
        raise ValueError(f"Invalid fraud labels in records: {ids}")

    df["label"] = labels.map({"no": 0, "yes": 1})
    return df.set_index("record_id")[["filing_text", "label"]]


def get_filing(record_id: str) -> dict:

    if not isinstance(record_id, str) or not record_id.strip():
        raise ValueError("Provide a non-empty record ID.")
    record_id = record_id.strip()
    df = load_filings()
    if record_id not in df.index:
        raise ValueError(f"Unknown record ID: {record_id}")


    return {
        "record_id": record_id,
        "filing_text": df.at[record_id, "filing_text"],
    }

MODEL_PATH = DATA_PATH.parents[1] / "artifacts" / "filing_classifier.joblib"

@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Model missing. Run train_model.py first.")


    return joblib.load(MODEL_PATH)


def review_filing_data(record_id: str) -> dict:
    filing = get_filing(record_id)
    text = filing["filing_text"]
    record_id = filing["record_id"]
    model = load_model()

    vectorizer = model["tfidf"]
    classifier = model["classifier"]
    features = vectorizer.transform([text])
    prediction = int(classifier.predict(features)[0])

    # Binary coefficients point toward classes_[1].
    direction = 1 if prediction == classifier.classes_[1] else -1
    contributions = (
        features.data
        * classifier.coef_[0, features.indices]
        * direction
    )
    vocabulary = vectorizer.get_feature_names_out()

    ranked = sorted(
        zip(features.indices, contributions),
        key=lambda item: item[1],
        reverse=True,
    )
    terms = [
        str(vocabulary[index])
        for index, contribution in ranked
        if contribution > 0
    ][:5]

    excerpts = []
    if terms:
        pattern = r"\b(?:" + "|".join(re.escape(term) for term in terms) + r")\b"

        # ponytail: first non-overlapping matches; improve selection if
        # repeated boilerplate makes these excerpts unhelpful.
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            start = max(0, match.start() - 250)
            end = min(len(text), start + 1000)

            if excerpts and start < excerpts[-1]["end"]:
                continue

            excerpts.append({
                "excerpt_id": f"{record_id}:excerpt_{len(excerpts) + 1}",
                "start": start,
                "end": end,
                "text": text[start:end],
            })
            if len(excerpts) == 3:
                break

    return {
        "record_id": record_id,
        "predicted_label": "yes" if prediction == 1 else "no",
        "influential_terms": terms,
        "excerpts": excerpts,
        "limitation": (
            "Exploratory prediction of the dataset label, not proof of fraud. "
            "Terms support the model prediction; excerpts show their context. "
            "A no prediction does not establish absence of fraud."
        ),
    }

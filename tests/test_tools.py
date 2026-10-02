"""Check the real filing tool on a tiny synthetic dataset and model."""

import joblib
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app import filing_service as service
from app.tools.base import get_tool_definitions
from app.tools.filing_review import review_filing


def test_filing_review(tmp_path, monkeypatch):
    data = pd.DataFrame({
        "Fillings": ["stable audited cash", "stable audited sales", "irregular hidden debt", "irregular hidden loss"],
        "Fraud": ["no", "no", "yes", "yes"],
    })
    data_path = tmp_path / "data.csv"
    model_path = tmp_path / "model.joblib"
    data.to_csv(data_path, index=False)
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression()),
    ]).fit(data["Fillings"], [0, 0, 1, 1])
    joblib.dump(model, model_path)
    monkeypatch.setattr(service, "DATA_PATH", data_path)
    monkeypatch.setattr(service, "MODEL_PATH", model_path)
    service.load_filings.cache_clear()
    service.load_model.cache_clear()
    try:
        definitions = get_tool_definitions()
        assert [tool["name"] for tool in definitions] == ["review_filing"]
        assert "record_id" in definitions[0]["parameters"]["required"]
        for record_id, expected in [("filing_001", "no"), ("filing_003", "yes")]:
            result = review_filing.invoke({"record_id": record_id})
            assert result["predicted_label"] == expected
            assert set(result) == {"record_id", "predicted_label", "influential_terms", "excerpts", "limitation"}
            assert 0 < len(result["excerpts"]) <= 3
            source = service.get_filing(record_id)["filing_text"]
            for excerpt in result["excerpts"]:
                assert excerpt["text"] == source[excerpt["start"]:excerpt["end"]]
                assert len(excerpt["text"]) <= 1000
        for invalid in ["unknown_record", "", None]:
            with pytest.raises(ValueError):
                service.get_filing(invalid)
        service.load_model.cache_clear()
        monkeypatch.setattr(service, "MODEL_PATH", tmp_path / "missing.joblib")
        with pytest.raises(FileNotFoundError):
            service.load_model()
        service.load_filings.cache_clear()
        pd.DataFrame({"wrong": ["column"]}).to_csv(data_path, index=False)
        with pytest.raises(ValueError, match="Missing required columns"):
            service.load_filings()
    finally:
        service.load_filings.cache_clear()
        service.load_model.cache_clear()

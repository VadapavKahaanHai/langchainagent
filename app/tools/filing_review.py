import json
from langchain_core.tools import tool
from app.filing_service import get_filing, review_filing_data


@tool
def review_filing(record_id: str) -> dict:
    """Review a filing by ID, returning a model prediction and source excerpts.

    Predictions are exploratory dataset classifications, not proof of fraud.
    """
    return review_filing_data(record_id)


if __name__ == "__main__":
    result = review_filing.invoke({"record_id": "filing_001"})
    source = get_filing(result["record_id"])["filing_text"]

    assert result["predicted_label"] in {"yes", "no"}
    assert set(result) == {
        "record_id",
        "predicted_label",
        "influential_terms",
        "excerpts",
        "limitation",
    }
    assert len(result["excerpts"]) <= 3

    for excerpt in result["excerpts"]:
        assert excerpt["text"] == source[excerpt["start"]:excerpt["end"]]
        assert len(excerpt["text"]) <= 1000

    try:
        review_filing.invoke({"record_id": "unknown_record"})
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown record IDs must be rejected.")

    print(json.dumps(result, indent=2))
    print("Review tool checks passed.")

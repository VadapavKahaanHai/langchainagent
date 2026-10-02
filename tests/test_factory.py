"""Gemini configuration checks without network requests."""

from unittest.mock import patch
import pytest
from app.core import llm_factory


def test_gemini_configuration(monkeypatch):
    monkeypatch.setattr(llm_factory.settings, "GOOGLE_API_KEY", "test-key")
    with patch.object(llm_factory, "ChatGoogleGenerativeAI") as model:
        assert llm_factory.create_llm(model_name="test-model", temperature=0) is model.return_value
        model.assert_called_once_with(model="test-model", temperature=0, google_api_key="test-key")
        llm_factory.create_llm(api_key="override-key")
        assert model.call_args.kwargs["google_api_key"] == "override-key"


def test_missing_key(monkeypatch):
    monkeypatch.setattr(llm_factory.settings, "GOOGLE_API_KEY", None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key not found"):
        llm_factory.create_llm()

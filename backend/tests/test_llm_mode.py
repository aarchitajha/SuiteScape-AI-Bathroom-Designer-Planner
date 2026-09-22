from backend.config import Settings


def test_get_llm_mode_prefers_groq_when_configured(monkeypatch):
    settings = Settings()
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "", raising=False)
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gsk_test_key", raising=False)
    monkeypatch.setattr(settings, "is_ollama_available", lambda: False)

    assert settings.get_llm_mode() == "groq"


def test_groq_vision_model_support_guard_rejects_non_vision_model():
    settings = Settings()
    assert settings.is_groq_vision_model_supported("") is False
    assert settings.is_groq_vision_model_supported("openai/gpt-oss-120b") is False
    assert settings.is_groq_vision_model_supported("llama-3.2-11b-vision-preview") is True

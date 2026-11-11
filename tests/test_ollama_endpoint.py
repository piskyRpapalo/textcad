"""A CAD prompt may only be sent to an explicitly configured local Ollama."""
import pytest

from textcad.llm import _ollama_endpoint


@pytest.mark.parametrize("base", [
    "http://localhost:11434",
    "http://127.0.0.1:21434",
    "http://[::1]:11434/",
])
def test_loopback_endpoint(monkeypatch, base):
    monkeypatch.setenv("TEXTCAD_OLLAMA_URL", base)
    assert _ollama_endpoint() == base.rstrip("/") + "/api/generate"


@pytest.mark.parametrize("base", [
    "https://127.0.0.1:11434",
    "http://example.com:11434",
    "http://192.168.1.2:11434",
    "http://127.0.0.1:11434/not/ollama",
    "http://127.0.0.1:11434/?leak=1",
    "http://user:password@127.0.0.1:11434",
    "http://127.0.0.1",
])
def test_remote_and_malformed_endpoint_rejected(monkeypatch, base):
    monkeypatch.setenv("TEXTCAD_OLLAMA_URL", base)
    with pytest.raises(ValueError, match="loopback HTTP"):
        _ollama_endpoint()

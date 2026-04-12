from meshagent.anthropic.proxy import proxy


def test_get_client_reads_base_url_from_environment(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeAsyncAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://env.anthropic.example.test")
    monkeypatch.setattr(proxy, "AsyncAnthropic", _FakeAsyncAnthropic)

    client = proxy.get_client()

    assert isinstance(client, _FakeAsyncAnthropic)
    assert captured["base_url"] == "https://env.anthropic.example.test"


def test_get_client_explicit_base_url_overrides_environment(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeAsyncAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://env.anthropic.example.test")
    monkeypatch.setattr(proxy, "AsyncAnthropic", _FakeAsyncAnthropic)

    client = proxy.get_client(base_url="https://explicit.anthropic.example.test")

    assert isinstance(client, _FakeAsyncAnthropic)
    assert captured["base_url"] == "https://explicit.anthropic.example.test"


def test_get_client_uses_meshagent_defaults_when_provider_env_missing(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeAsyncAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("MESHAGENT_API_URL", "https://api.example.test")
    monkeypatch.setenv("MESHAGENT_TOKEN", "meshagent-token")
    monkeypatch.setattr(proxy, "AsyncAnthropic", _FakeAsyncAnthropic)

    client = proxy.get_client()

    assert isinstance(client, _FakeAsyncAnthropic)
    assert captured["base_url"] == "https://api.example.test/anthropic"
    assert captured["api_key"] == "meshagent-token"

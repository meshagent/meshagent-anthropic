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

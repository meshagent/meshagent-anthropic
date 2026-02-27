from meshagent.anthropic.thread_adapter import AnthropicThreadAdapter


def test_conversation_from_export_uses_text_blocks_and_skips_thinking() -> None:
    adapter = AnthropicThreadAdapter()

    conversation = adapter.conversation_from_export(
        raw_conversation={
            "uuid": "conv-1",
            "name": "Example",
            "created_at": "2026-02-01T00:00:00Z",
            "updated_at": "2026-02-01T00:10:00Z",
            "chat_messages": [
                {
                    "uuid": "msg-1",
                    "sender": "assistant",
                    "text": "thinking + answer merged in export",
                    "created_at": "2026-02-01T00:00:01Z",
                    "content": [
                        {"type": "thinking", "thinking": "internal chain of thought"},
                        {"type": "text", "text": "final answer"},
                    ],
                }
            ],
        }
    )

    assert conversation.conversation_id == "conv-1"
    assert len(conversation.messages) == 1
    assert conversation.messages[0].author_name == "assistant"
    assert conversation.messages[0].text == "final answer"


def test_conversation_from_export_uses_fallback_for_attachment_only_message() -> None:
    adapter = AnthropicThreadAdapter()

    conversation = adapter.conversation_from_export(
        raw_conversation={
            "uuid": "conv-1",
            "name": "Attachments",
            "created_at": "2026-02-01T00:00:00Z",
            "updated_at": "2026-02-01T00:10:00Z",
            "chat_messages": [
                {
                    "uuid": "msg-1",
                    "sender": "human",
                    "text": "",
                    "created_at": "2026-02-01T00:00:01Z",
                    "content": [],
                    "attachments": [{"file_name": "notes.txt"}],
                    "files": [],
                }
            ],
        }
    )

    assert len(conversation.messages) == 1
    assert conversation.messages[0].text == "attached file notes.txt"
    assert conversation.messages[0].author_name == "human"


def test_to_thread_json_contains_members_and_external_thread_id() -> None:
    adapter = AnthropicThreadAdapter()
    conversation = adapter.conversation_from_export(
        raw_conversation={
            "uuid": "conv-1",
            "name": "Thread Name",
            "created_at": "2026-02-01T00:00:00Z",
            "updated_at": "2026-02-01T00:10:00Z",
            "chat_messages": [
                {
                    "uuid": "msg-1",
                    "sender": "human",
                    "text": "",
                    "created_at": "2026-02-01T00:00:01Z",
                    "content": [{"type": "text", "text": "hello"}],
                },
                {
                    "uuid": "msg-2",
                    "sender": "assistant",
                    "text": "",
                    "created_at": "2026-02-01T00:00:02Z",
                    "content": [{"type": "text", "text": "hi"}],
                },
            ],
        }
    )

    thread_json = adapter.to_thread_json(conversation=conversation)
    messages = thread_json["thread"]["properties"][1]["messages"]
    members = thread_json["thread"]["properties"][0]["members"]["items"]

    assert thread_json["thread"]["name"] == "Thread Name"
    assert messages["external_thread_id"] == "conv-1"
    assert [item["member"]["name"] for item in members] == ["human", "assistant"]
    assert messages["items"][0]["message"]["text"] == "hello"

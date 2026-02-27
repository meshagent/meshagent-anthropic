from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _non_empty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text == "":
        return None
    return text


def _content_blocks(raw_message: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    blocks = list[Mapping[str, Any]]()
    content = raw_message.get("content")
    if not isinstance(content, list):
        return blocks

    for raw_block in content:
        if isinstance(raw_block, dict):
            blocks.append(raw_block)
    return blocks


def _text_from_content_blocks(blocks: list[Mapping[str, Any]]) -> str:
    parts = list[str]()
    for block in blocks:
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts).strip()


def _attachment_file_names(raw_message: Mapping[str, Any]) -> list[str]:
    names = list[str]()

    for key in ("attachments", "files"):
        items = raw_message.get(key)
        if not isinstance(items, list):
            continue

        for raw_item in items:
            if not isinstance(raw_item, dict):
                continue
            file_name = _non_empty_string(raw_item.get("file_name"))
            if file_name is not None:
                names.append(file_name)

    deduped = list[str]()
    seen = set[str]()
    for name in names:
        lookup = name.lower()
        if lookup in seen:
            continue
        seen.add(lookup)
        deduped.append(name)
    return deduped


def _message_created_at(
    *,
    raw_message: Mapping[str, Any],
    blocks: list[Mapping[str, Any]],
) -> str | None:
    created_at = _non_empty_string(raw_message.get("created_at"))
    if created_at is not None:
        return created_at

    for block in blocks:
        start_timestamp = _non_empty_string(block.get("start_timestamp"))
        if start_timestamp is not None:
            return start_timestamp
        stop_timestamp = _non_empty_string(block.get("stop_timestamp"))
        if stop_timestamp is not None:
            return stop_timestamp

    return None


@dataclass(frozen=True)
class AnthropicThreadMessage:
    message_id: str
    author_name: str
    created_at: str
    text: str


@dataclass(frozen=True)
class AnthropicThreadConversation:
    conversation_id: str
    name: str
    created_at: str
    updated_at: str
    messages: tuple[AnthropicThreadMessage, ...]


class AnthropicThreadAdapter:
    def __init__(
        self,
        *,
        human_author_name: str = "human",
        assistant_author_name: str = "assistant",
        unknown_author_name: str = "participant",
        include_empty_messages: bool = False,
    ):
        self._human_author_name = human_author_name
        self._assistant_author_name = assistant_author_name
        self._unknown_author_name = unknown_author_name
        self._include_empty_messages = include_empty_messages

    def _author_name_for_sender(self, *, sender: str, raw_sender: str | None) -> str:
        if sender == "human":
            return self._human_author_name
        if sender == "assistant":
            return self._assistant_author_name
        if raw_sender is not None:
            return raw_sender
        return self._unknown_author_name

    def _message_text(
        self,
        *,
        raw_message: Mapping[str, Any],
        blocks: list[Mapping[str, Any]],
    ) -> str:
        if len(blocks) > 0:
            text = _text_from_content_blocks(blocks)
            if text != "":
                return text

        if len(blocks) == 0:
            fallback = _non_empty_string(raw_message.get("text"))
            if fallback is not None:
                return fallback

        file_names = _attachment_file_names(raw_message)
        if len(file_names) == 1:
            return f"attached file {file_names[0]}"
        if len(file_names) > 1:
            return "attached files " + ", ".join(file_names)
        return ""

    def conversation_from_export(
        self,
        *,
        raw_conversation: Mapping[str, Any],
    ) -> AnthropicThreadConversation:
        conversation_id = _non_empty_string(raw_conversation.get("uuid"))
        if conversation_id is None:
            raise ValueError("conversation is missing uuid")

        conversation_name = (
            _non_empty_string(raw_conversation.get("name")) or conversation_id
        )
        conversation_created_at = (
            _non_empty_string(raw_conversation.get("created_at")) or ""
        )
        conversation_updated_at = (
            _non_empty_string(raw_conversation.get("updated_at"))
            or conversation_created_at
        )

        raw_messages = raw_conversation.get("chat_messages")
        if not isinstance(raw_messages, list):
            raise ValueError("conversation chat_messages must be an array")

        messages = list[AnthropicThreadMessage]()
        for index, raw_message in enumerate(raw_messages):
            if not isinstance(raw_message, dict):
                continue

            blocks = _content_blocks(raw_message)
            text = self._message_text(raw_message=raw_message, blocks=blocks)
            if text == "" and not self._include_empty_messages:
                continue

            raw_sender = _non_empty_string(raw_message.get("sender"))
            sender = raw_sender.lower() if raw_sender is not None else ""
            author_name = self._author_name_for_sender(
                sender=sender,
                raw_sender=raw_sender,
            )

            message_created_at = _message_created_at(
                raw_message=raw_message,
                blocks=blocks,
            )
            if message_created_at is None:
                message_created_at = (
                    conversation_updated_at
                    if conversation_updated_at != ""
                    else conversation_created_at
                )

            message_id = (
                _non_empty_string(raw_message.get("uuid"))
                or f"{conversation_id}:message:{index}"
            )
            messages.append(
                AnthropicThreadMessage(
                    message_id=message_id,
                    author_name=author_name,
                    created_at=message_created_at,
                    text=text,
                )
            )

        return AnthropicThreadConversation(
            conversation_id=conversation_id,
            name=conversation_name,
            created_at=conversation_created_at,
            updated_at=conversation_updated_at,
            messages=tuple(messages),
        )

    def to_thread_json(
        self,
        *,
        conversation: AnthropicThreadConversation,
    ) -> dict[str, Any]:
        member_names = list[str]()
        seen_members = set[str]()
        for message in conversation.messages:
            member_name = message.author_name.strip()
            if member_name == "":
                continue
            if member_name in seen_members:
                continue
            seen_members.add(member_name)
            member_names.append(member_name)

        if len(member_names) == 0:
            member_names = [
                self._human_author_name,
                self._assistant_author_name,
            ]

        member_items = [
            {"member": {"name": member_name}} for member_name in member_names
        ]
        message_items = [
            {
                "message": {
                    "id": message.message_id,
                    "text": message.text,
                    "created_at": message.created_at,
                    "author_name": message.author_name,
                }
            }
            for message in conversation.messages
        ]

        return {
            "thread": {
                "name": conversation.name,
                "properties": [
                    {"members": {"items": member_items}},
                    {
                        "messages": {
                            "external_thread_id": conversation.conversation_id,
                            "items": message_items,
                        }
                    },
                ],
            }
        }

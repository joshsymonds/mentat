"""Conversation entity streaming mentat turns into the chat log."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, Literal

import aiohttp

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import aiohttp_client, intent
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import CONF_BASE_URL, SESSION_PREFIX, TURN_EFFORT, TURN_META
from .stream import LineSplitter, TurnError, wire_line_to_delta

# A turn legitimately runs for minutes while the daemon uses tools; only the
# connect and the gap between chunks are bounded.
TIMEOUT = aiohttp.ClientTimeout(total=None, connect=10, sock_read=600)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up the conversation entity."""
    async_add_entities([MentatConversationEntity(config_entry)])


class MentatConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """A conversation agent proxying turns to the mentat daemon.

    The daemon owns conversation memory (one persistent session per
    conversation id), so only the current utterance is sent — never the chat
    log history.
    """

    _attr_name = "Mentat"
    _attr_supports_streaming = True

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self._attr_unique_id = config_entry.entry_id
        self._base_url = str(config_entry.data[CONF_BASE_URL]).rstrip("/")

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Stream one daemon turn into the chat log."""
        session = aiohttp_client.async_get_clientsession(self.hass)
        body = {
            "session_id": SESSION_PREFIX + chat_log.conversation_id,
            "text": user_input.text,
            "meta": TURN_META,
            "effort": TURN_EFFORT,
        }
        try:
            async with session.post(
                f"{self._base_url}/v1/conversation", json=body, timeout=TIMEOUT
            ) as response:
                if response.status != 200:
                    raise TurnError(f"daemon answered HTTP {response.status}")
                async for _content in chat_log.async_add_delta_content_stream(
                    self.entity_id, _deltas(response)
                ):
                    pass
        except (TurnError, aiohttp.ClientError, TimeoutError) as err:
            # Cancellation (Assist "stop") is deliberately not caught here:
            # it propagates and closes the request, which is exactly what
            # makes the daemon abort the in-flight turn.
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN, f"Mentat error: {err}"
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=chat_log.conversation_id
            )
        return conversation.async_get_result_from_chat_log(user_input, chat_log)


async def _deltas(response: aiohttp.ClientResponse) -> AsyncGenerator[dict[str, Any]]:
    """Translate the NDJSON response body into assistant content deltas."""
    yield {"role": "assistant"}
    splitter = LineSplitter()
    async for chunk in response.content.iter_any():
        for line in splitter.feed(chunk):
            if (delta := wire_line_to_delta(line)) is not None:
                yield delta

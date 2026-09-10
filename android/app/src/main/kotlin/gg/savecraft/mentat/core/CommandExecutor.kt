package gg.savecraft.mentat.core

import android.content.ActivityNotFoundException
import java.nio.charset.StandardCharsets
import java.time.Instant
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

sealed class SmsSendOutcome {
    data object Sent : SmsSendOutcome()
    data class Failed(val resultCode: Int) : SmsSendOutcome()
    data object Unknown : SmsSendOutcome()
    data object Expired : SmsSendOutcome()
}

interface SmsSender {
    val hasPermission: Boolean
    suspend fun send(number: String, body: String, deadline: Instant): SmsSendOutcome
}

data class ContactMatch(val displayName: String, val number: String)

interface ContactResolver {
    val hasPermission: Boolean
    fun resolve(name: String): List<ContactMatch>
}

interface IntentLauncher {
    fun canDrawOverlays(): Boolean
    fun launch(uri: String)
}

interface Clock {
    fun now(): Instant
}

object SystemClock : Clock {
    override fun now(): Instant = Instant.now()
}

class CommandExecutor(
    private val smsSender: SmsSender,
    private val contactResolver: ContactResolver,
    private val intentLauncher: IntentLauncher,
    private val messageStore: MessageStore,
    private val clock: Clock = SystemClock,
) {
    suspend fun execute(command: PhoneCommand): PhoneResult {
        return when (command) {
            is PhoneCommand.Sms -> executeSms(command)
            is PhoneCommand.Open -> executeOpen(command)
            is PhoneCommand.Conversations -> executeConversations(command)
            is PhoneCommand.Messages -> executeMessages(command)
            is PhoneCommand.Search -> executeSearch(command)
            PhoneCommand.Ping -> PhoneResult("", "ok", "ping")
        }
    }

    private suspend fun executeConversations(command: PhoneCommand.Conversations): PhoneResult {
        if (expired(command.expiresAt)) {
            return PhoneResult(command.id, "error", "expired")
        }
        if (!messageStore.hasPermission) {
            return PhoneResult(command.id, "error", "permission denied")
        }
        val rows = withContext(Dispatchers.IO) { messageStore.conversations(command.limit) }
        val conversations = JSONArray()
        rows.forEach { conversations.put(conversationJson(it)) }
        val payload = JSONObject().put("conversations", conversations)
        return PhoneResult(command.id, "ok", "${rows.size} conversations", payload)
    }

    private suspend fun executeMessages(command: PhoneCommand.Messages): PhoneResult {
        if (expired(command.expiresAt)) {
            return PhoneResult(command.id, "error", "expired")
        }
        if (!messageStore.hasPermission) {
            return PhoneResult(command.id, "error", "permission denied")
        }
        val threadId = when (val selection = classifyConversation(command.conversation)) {
            is ConversationSelection.Error -> return PhoneResult(command.id, "error", selection.detail)
            is ConversationSelection.Thread -> selection.threadId
        }
        val boundary = parseBoundary(command.before)
            ?: if (command.before == null) null else return PhoneResult(command.id, "error", "invalid before")
        val rows = withContext(Dispatchers.IO) { messageStore.messages(threadId, command.limit, boundary) }
        return pageResult(command.id, rows, command.limit, "messages")
    }

    private suspend fun executeSearch(command: PhoneCommand.Search): PhoneResult {
        if (expired(command.expiresAt)) {
            return PhoneResult(command.id, "error", "expired")
        }
        if (!messageStore.hasPermission) {
            return PhoneResult(command.id, "error", "permission denied")
        }
        val boundary = parseBoundary(command.before)
            ?: if (command.before == null) null else return PhoneResult(command.id, "error", "invalid before")
        val rows = withContext(Dispatchers.IO) { messageStore.search(command.query, command.limit, boundary) }
        return pageResult(command.id, rows, command.limit, "matches")
    }

    private suspend fun executeSms(command: PhoneCommand.Sms): PhoneResult {
        if (expired(command.expiresAt)) {
            return PhoneResult(command.id, "error", "expired")
        }

        val number = if (isDialableNumber(command.to)) {
            if (!smsSender.hasPermission) {
                return PhoneResult(command.id, "error", "permission denied")
            }
            command.to
        } else {
            if (!contactResolver.hasPermission || !smsSender.hasPermission) {
                return PhoneResult(command.id, "error", "permission denied")
            }
            val matches = contactResolver.resolve(command.to)
                .distinctBy { normalizeNumber(it.number) }
            when (matches.size) {
                0 -> return PhoneResult(command.id, "error", "not found")
                1 -> matches.single().number
                else -> {
                    val details = matches.joinToString(", ") { "${it.displayName} (${it.number})" }
                    return PhoneResult(command.id, "error", "ambiguous: $details")
                }
            }
        }

        return when (val outcome = try {
            withContext(Dispatchers.IO) {
                smsSender.send(number, command.body, command.expiresAt)
            }
        } catch (_: SecurityException) {
            return PhoneResult(command.id, "error", "permission denied")
        }) {
            SmsSendOutcome.Sent -> PhoneResult(command.id, "ok", "sent to $number")
            is SmsSendOutcome.Failed -> PhoneResult(command.id, "error", "send failed: ${outcome.resultCode}")
            SmsSendOutcome.Unknown -> PhoneResult(command.id, "error", "outcome unknown")
            SmsSendOutcome.Expired -> PhoneResult(command.id, "error", "expired")
        }
    }

    private fun executeOpen(command: PhoneCommand.Open): PhoneResult {
        if (expired(command.expiresAt)) {
            return PhoneResult(command.id, "error", "expired")
        }
        if (!intentLauncher.canDrawOverlays()) {
            return PhoneResult(command.id, "error", "overlay permission missing")
        }
        return try {
            intentLauncher.launch(command.uri)
            PhoneResult(command.id, "ok", "launched")
        } catch (_: ActivityNotFoundException) {
            PhoneResult(command.id, "error", "no handler")
        }
    }

    private fun classifyConversation(value: String): ConversationSelection {
        val explicit = EXPLICIT_ID.matchEntire(value)
        if (explicit != null) {
            return ConversationSelection.Thread(explicit.groupValues[1].toLong())
        }
        if (isDialableNumber(value)) {
            return directSelection(listOf(value))
        }
        if (!contactResolver.hasPermission) {
            return ConversationSelection.Error("permission denied")
        }
        return directSelection(contactResolver.resolve(value).map { it.number })
    }

    private fun directSelection(numbers: List<String>): ConversationSelection {
        val matches = messageStore.directThreadsFor(numbers)
        return when (matches.size) {
            0 -> ConversationSelection.Error("not found")
            1 -> ConversationSelection.Thread(matches.single().first)
            else -> ConversationSelection.Error(
                "ambiguous: " + matches.joinToString(", ") { (threadId, participant) ->
                    "sms:$threadId (${participant.name ?: participant.number})"
                },
            )
        }
    }

    private fun parseBoundary(value: String?): Boundary? {
        if (value == null) return null
        val cursor = CURSOR.matchEntire(value)
        if (cursor != null) {
            return Boundary(cursor.groupValues[1].toLong(), cursor.groupValues[2])
        }
        return runCatching { Boundary(Instant.parse(value).toEpochMilli(), null) }.getOrNull()
    }

    private fun pageResult(id: String, rows: List<MessageRow>, limit: Int, detailNoun: String): PhoneResult {
        val retained = mutableListOf<MessageRow>()
        for (row in rows) {
            if (retained.isEmpty()) {
                retained += row
                continue
            }
            val candidate = retained + row
            if (payloadBytes(candidate) <= PAGE_BUDGET_BYTES) {
                retained += row
            } else {
                break
            }
        }
        val messages = JSONArray()
        retained.asReversed().forEach { messages.put(messageJson(it)) }
        val payload = JSONObject().put("messages", messages)
        if (retained.isNotEmpty() && (retained.size < rows.size || rows.size >= limit)) {
            payload.put("next", cursorFor(retained.last()))
        }
        return PhoneResult(id, "ok", "${retained.size} $detailNoun", payload)
    }

    private fun payloadBytes(rows: List<MessageRow>): Int {
        val messages = JSONArray()
        rows.forEach { messages.put(messageJson(it)) }
        return JSONObject().put("messages", messages).toString().toByteArray(StandardCharsets.UTF_8).size
    }

    private fun conversationJson(row: ConversationRow): JSONObject {
        val participants = JSONArray()
        row.participants.forEach { participant ->
            participants.put(participantJson(participant))
        }
        return JSONObject()
            .put("id", "sms:${row.threadId}")
            .put("channel", "sms")
            .put("participants", participants)
            .put(
                "last_message",
                JSONObject()
                    .put("direction", row.lastDirection)
                    .put("body", row.lastBody.take(PREVIEW_LIMIT))
                    .put("at", Instant.ofEpochMilli(row.lastAtMs).toString()),
            )
            .put("message_count", row.messageCount)
            .put("unread_count", row.unreadCount)
    }

    private fun messageJson(row: MessageRow): JSONObject {
        val body = row.body.take(BODY_LIMIT)
        val message = JSONObject()
            .put("id", "sms:${row.id}")
            .put("conversation", "sms:${row.threadId}")
            .put("channel", "sms")
            .put("direction", row.direction)
            .put("body", body)
            .put("at", Instant.ofEpochMilli(row.atMs).toString())
        row.from?.let { message.put("from", participantJson(it)) }
        val attachments = JSONArray()
        row.attachments.forEach { attachment ->
            val json = JSONObject().put("content_type", attachment.contentType)
            attachment.name?.let { json.put("name", it) }
            attachments.put(json)
        }
        message.put("attachments", attachments)
        if (row.body.length > BODY_LIMIT) message.put("truncated", true)
        return message
    }

    private fun participantJson(participant: Participant): JSONObject = JSONObject().apply {
        participant.name?.let { put("name", it) }
        put("number", participant.number)
    }

    private fun cursorFor(row: MessageRow): String = "${row.atMs}:sms:${row.id}"

    private fun expired(deadline: Instant): Boolean = !clock.now().isBefore(deadline)

    private fun isDialableNumber(value: String): Boolean =
        value.matches(Regex("^\\+?[0-9][0-9 .()\\-]{2,}$"))

    private fun normalizeNumber(value: String): String =
        value.filter { it.isDigit() }.let { digits -> if (value.trim().startsWith('+')) "+$digits" else digits }

    private sealed class ConversationSelection {
        data class Thread(val threadId: Long) : ConversationSelection()
        data class Error(val detail: String) : ConversationSelection()
    }

    private companion object {
        const val BODY_LIMIT = 2_000
        const val PREVIEW_LIMIT = 200
        const val PAGE_BUDGET_BYTES = 512 * 1024
        val EXPLICIT_ID = Regex("^sms:(\\d+)$")
        val CURSOR = Regex("^(\\d+):sms:(.+)$")
    }
}

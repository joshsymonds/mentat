package gg.savecraft.mentat.core

import android.content.ActivityNotFoundException
import java.nio.charset.StandardCharsets
import java.time.Instant
import org.json.JSONArray
import org.json.JSONObject
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CommandExecutorTest {
    private val expiry = Instant.parse("2026-09-10T00:00:00Z")

    @Test
    fun expiredSmsIsNotSent() = runBlocking {
        val sms = FakeSmsSender()
        val result = executor(sms = sms, now = expiry).execute(smsCommand())
        assertEquals(PhoneResult("id", "error", "expired"), result)
        assertTrue(sms.calls.isEmpty())
    }

    @Test
    fun dialableNumberSendsDirectly() = runBlocking {
        val sms = FakeSmsSender()
        val result = executor(sms = sms).execute(smsCommand(to = "+1 (555) 1212"))
        assertEquals(PhoneResult("id", "ok", "sent to +1 (555) 1212"), result)
        assertEquals(listOf("+1 (555) 1212" to "hello"), sms.calls)
    }

    @Test
    fun contactNameWithOneNumberIsResolvedAndSent() = runBlocking {
        val sms = FakeSmsSender()
        val contacts = FakeContactResolver(listOf(ContactMatch("Mum", "+15551212")))
        val result = executor(sms = sms, contacts = contacts).execute(smsCommand(to = "Mum"))
        assertEquals(PhoneResult("id", "ok", "sent to +15551212"), result)
    }

    @Test
    fun noContactReturnsNotFound() = runBlocking {
        val result = executor(contacts = FakeContactResolver()).execute(smsCommand(to = "Nobody"))
        assertEquals(PhoneResult("id", "error", "not found"), result)
    }

    @Test
    fun severalDistinctContactsReturnsAmbiguous() = runBlocking {
        val contacts = FakeContactResolver(
            listOf(
                ContactMatch("Sarah K", "+15550001"),
                ContactMatch("Sarah M", "+15550002"),
            ),
        )
        val result = executor(contacts = contacts).execute(smsCommand(to = "Sarah"))
        assertEquals(
            PhoneResult("id", "error", "ambiguous: Sarah K (+15550001), Sarah M (+15550002)"),
            result,
        )
    }

    @Test
    fun missingSmsPermissionReturnsPermissionDenied() = runBlocking {
        val sms = FakeSmsSender(hasPermission = false)
        val result = executor(sms = sms).execute(smsCommand())
        assertEquals(PhoneResult("id", "error", "permission denied"), result)
        assertTrue(sms.calls.isEmpty())
    }

    @Test
    fun missingContactsPermissionReturnsPermissionDenied() = runBlocking {
        val result = executor(contacts = FakeContactResolver(hasPermission = false)).execute(smsCommand(to = "Mum"))
        assertEquals(PhoneResult("id", "error", "permission denied"), result)
    }

    @Test
    fun failedPartReturnsItsResultCode() = runBlocking {
        val result = executor(sms = FakeSmsSender(SmsSendOutcome.Failed(17))).execute(smsCommand())
        assertEquals(PhoneResult("id", "error", "send failed: 17"), result)
    }

    @Test
    fun expiredSmsOutcomeIsReportedAsExpired() = runBlocking {
        val result = executor(sms = FakeSmsSender(SmsSendOutcome.Expired)).execute(smsCommand())
        assertEquals(PhoneResult("id", "error", "expired"), result)
    }

    @Test
    fun unknownSmsOutcomeDoesNotReplayAndNextCommandCanProceed() = runBlocking {
        val sms = FakeSmsSender(SmsSendOutcome.Unknown)
        val first = executor(sms = sms).execute(smsCommand())
        sms.outcome = SmsSendOutcome.Sent
        val second = executor(sms = sms).execute(openCommand())
        assertEquals(PhoneResult("id", "error", "outcome unknown"), first)
        assertEquals(PhoneResult("open", "ok", "launched"), second)
        assertEquals(1, sms.calls.size)
    }

    @Test
    fun expiredOpenIsNotLaunched() = runBlocking {
        val launcher = FakeIntentLauncher()
        val result = executor(launcher = launcher, now = expiry).execute(openCommand())
        assertEquals(PhoneResult("open", "error", "expired"), result)
        assertTrue(launcher.uris.isEmpty())
    }

    @Test
    fun openWithoutOverlayPermissionIsRejected() = runBlocking {
        val result = executor(launcher = FakeIntentLauncher(overlay = false)).execute(openCommand())
        assertEquals(PhoneResult("open", "error", "overlay permission missing"), result)
    }

    @Test
    fun openLaunchesUri() = runBlocking {
        val launcher = FakeIntentLauncher()
        val result = executor(launcher = launcher).execute(openCommand())
        assertEquals(PhoneResult("open", "ok", "launched"), result)
        assertEquals(listOf("https://example.com"), launcher.uris)
    }

    @Test
    fun activityNotFoundIsReported() = runBlocking {
        val launcher = FakeIntentLauncher(exception = ActivityNotFoundException())
        val result = executor(launcher = launcher).execute(openCommand())
        assertEquals(PhoneResult("open", "error", "no handler"), result)
    }

    @Test
    fun expiredReadCommandDoesNotCallStore() = runBlocking {
        val store = FakeMessageStore()
        val result = executor(store = store, now = expiry).execute(
            PhoneCommand.Conversations("c", 20, expiry),
        )
        assertEquals(PhoneResult("c", "error", "expired"), result)
        assertEquals(0, store.conversationsCalls)
    }

    @Test
    fun missingReadPermissionIsRejectedBeforeStoreQuery() = runBlocking {
        val store = FakeMessageStore(hasPermission = false)
        val result = executor(store = store).execute(
            PhoneCommand.Conversations("c", 20, expiry),
        )
        assertEquals(PhoneResult("c", "error", "permission denied"), result)
        assertEquals(0, store.conversationsCalls)
    }

    @Test
    fun expiredMessagesAndSearchDoNotCallStore() = runBlocking {
        val store = FakeMessageStore()
        assertEquals(
            PhoneResult("m", "error", "expired"),
            executor(store = store, now = expiry).execute(
                PhoneCommand.Messages("m", "sms:42", 2, null, expiry),
            ),
        )
        assertEquals(
            PhoneResult("s", "error", "expired"),
            executor(store = store, now = expiry).execute(
                PhoneCommand.Search("s", "needle", 2, null, expiry),
            ),
        )
        assertEquals(0, store.messagesCalls)
        assertEquals(0, store.searchCalls)
    }

    @Test
    fun missingReadPermissionRejectsMessagesAndSearchBeforeStoreQuery() = runBlocking {
        val store = FakeMessageStore(hasPermission = false)
        assertEquals(
            PhoneResult("m", "error", "permission denied"),
            executor(store = store).execute(
                PhoneCommand.Messages("m", "sms:42", 2, null, expiry),
            ),
        )
        assertEquals(
            PhoneResult("s", "error", "permission denied"),
            executor(store = store).execute(
                PhoneCommand.Search("s", "needle", 2, null, expiry),
            ),
        )
        assertEquals(0, store.messagesCalls)
        assertEquals(0, store.searchCalls)
    }

    @Test
    fun overflowingExplicitThreadIdReturnsNotFoundInsteadOfThrowing() = runBlocking {
        val result = executor().execute(
            PhoneCommand.Messages("m", "sms:9223372036854775808", 2, null, expiry),
        )
        assertEquals(PhoneResult("m", "error", "not found"), result)
    }

    @Test
    fun overflowingCursorEpochReturnsInvalidBeforeInsteadOfThrowing() = runBlocking {
        val result = executor().execute(
            PhoneCommand.Messages("m", "sms:42", 2, "9223372036854775808:sms:s1", expiry),
        )
        assertEquals(PhoneResult("m", "error", "invalid before"), result)
    }

    @Test
    fun conversationsUseEnvelopeAndPreviewCap() = runBlocking {
        val store = FakeMessageStore(conversationRows = listOf(
            ConversationRow(42, listOf(Participant("Mum", "+15551212")), "in", "x".repeat(201), 1_000, 3, 2),
        ))
        val result = executor(store = store).execute(PhoneCommand.Conversations("c", 20, expiry))
        val conversation = result.payload!!.getJSONArray("conversations").getJSONObject(0)
        assertEquals("sms:42", conversation.getString("id"))
        assertEquals(200, conversation.getJSONObject("last_message").getString("body").length)
        assertEquals(PhoneResult("c", "ok", "1 conversations", result.payload), result)
    }

    @Test
    fun messagesAreOldestFirstAndBodiesAreTruncated() = runBlocking {
        val store = FakeMessageStore(messageRows = listOf(
            MessageRow("s2", 42, "in", Participant(null, "+15551212"), "new", 2_000, emptyList()),
            MessageRow("s1", 42, "out", null, "x".repeat(2_001), 1_000, emptyList()),
        ))
        val result = executor(store = store).execute(PhoneCommand.Messages("m", "sms:42", 50, null, expiry))
        val rows = result.payload!!.getJSONArray("messages")
        assertEquals("s1", rows.getJSONObject(0).getString("id").removePrefix("sms:"))
        assertEquals("s2", rows.getJSONObject(1).getString("id").removePrefix("sms:"))
        assertTrue(rows.getJSONObject(0).getBoolean("truncated"))
        assertEquals(2_000, rows.getJSONObject(0).getString("body").length)
        assertEquals("2 messages", result.detail)
    }

    @Test
    fun exactlyLimitRowsDoNotInventNextAndAnExtraRowProvidesNext() = runBlocking {
        val rows = listOf(
            MessageRow("s2", 42, "in", null, "new", 2_000, emptyList()),
            MessageRow("s1", 42, "in", null, "old", 1_000, emptyList()),
        )
        val store = FakeMessageStore(messageRows = rows)
        val exact = executor(store = store).execute(PhoneCommand.Messages("exact", "sms:42", 2, null, expiry))
        assertEquals(false, exact.payload!!.has("next"))

        store.messageRows = rows + MessageRow("s0", 42, "in", null, "older", 0, emptyList())
        val extra = executor(store = store).execute(PhoneCommand.Messages("extra", "sms:42", 2, null, expiry))
        assertEquals("1000:sms:s1", extra.payload!!.getString("next"))
    }

    @Test
    fun searchUsesSameExtraRowPagingRule() = runBlocking {
        val store = FakeMessageStore(messageRows = listOf(
            MessageRow("s2", 42, "in", null, "new", 2_000, emptyList()),
            MessageRow("s1", 42, "in", null, "old", 1_000, emptyList()),
        ))
        val exact = executor(store = store).execute(PhoneCommand.Search("exact", "needle", 2, null, expiry))
        assertEquals(false, exact.payload!!.has("next"))

        store.messageRows += MessageRow("s0", 42, "in", null, "older", 0, emptyList())
        val extra = executor(store = store).execute(PhoneCommand.Search("extra", "needle", 2, null, expiry))
        assertEquals("1000:sms:s1", extra.payload!!.getString("next"))
    }

    @Test
    fun finalBudgetIncludesNextAndKeepsTheSpilledRowReachable() = runBlocking {
        val second = MessageRow("s1", 42, "in", null, "old", 1_000, emptyList())
        var attachmentNameLength = 524_288
        fun firstRow() = MessageRow(
            "s2", 42, "in", null, "new", 2_000,
            listOf(Attachment("application/octet-stream", "x".repeat(attachmentNameLength))),
        )
        while (serializedMessages(listOf(firstRow(), second), null) > 512 * 1024) {
            attachmentNameLength -= 1
        }
        val rows = listOf(firstRow(), second)
        val withoutNext = serializedMessages(rows, null)
        val withNext = serializedMessages(rows, "1000:sms:s1")
        assertTrue(withoutNext <= 512 * 1024)
        assertTrue(withoutNext >= 512 * 1024 - 64)
        assertTrue(withNext > 512 * 1024)

        val store = FakeMessageStore(messageRows = rows)
        val first = executor(store = store).execute(PhoneCommand.Messages("a", "sms:42", 1, null, expiry))
        assertEquals(1, first.payload!!.getJSONArray("messages").length())
        assertEquals("2000:sms:s2", first.payload!!.getString("next"))
        assertTrue(first.payload.toString().toByteArray(StandardCharsets.UTF_8).size <= 512 * 1024)

        val secondPage = executor(store = store).execute(
            PhoneCommand.Messages("b", "sms:42", 1, first.payload.getString("next"), expiry),
        )
        assertEquals(listOf("sms:s1"), listOf(secondPage.payload!!.getJSONArray("messages").getJSONObject(0).getString("id")))
    }

    @Test
    fun oversizedFirstRecordIsRetainedEvenWhenAloneOverBudget() = runBlocking {
        val row = MessageRow(
            "s1", 42, "in", null, "body", 1_000,
            (1..300).map { Attachment("application/octet-stream", "x".repeat(2_000)) },
        )
        val result = executor(store = FakeMessageStore(messageRows = listOf(row))).execute(
            PhoneCommand.Messages("m", "sms:42", 1, null, expiry),
        )
        val payload = result.payload!!
        assertEquals(1, payload.getJSONArray("messages").length())
        assertTrue(payload.toString().toByteArray(StandardCharsets.UTF_8).size > 512 * 1024)
        assertEquals(false, payload.has("next"))
    }

    @Test
    fun emptyReadEnvelopesHaveNoNextCursor() = runBlocking {
        val store = FakeMessageStore()
        val conversations = executor(store = store).execute(PhoneCommand.Conversations("c", 20, expiry))
        val messages = executor(store = store).execute(PhoneCommand.Messages("m", "sms:42", 2, null, expiry))
        assertEquals("[]", conversations.payload!!.getJSONArray("conversations").toString())
        assertEquals(false, conversations.payload.has("next"))
        assertEquals("[]", messages.payload!!.getJSONArray("messages").toString())
        assertEquals(false, messages.payload.has("next"))
    }

    @Test
    fun searchContinuationUsesNextWithoutGapOrOverlap() = runBlocking {
        val store = FakeMessageStore(messageRows = listOf(
            MessageRow("s4", 42, "in", null, "four", 4_000, emptyList()),
            MessageRow("s3", 42, "in", null, "three", 3_000, emptyList()),
            MessageRow("s2", 42, "in", null, "two", 2_000, emptyList()),
            MessageRow("s1", 42, "in", null, "one", 1_000, emptyList()),
        ))
        val first = executor(store = store).execute(PhoneCommand.Search("a", "needle", 2, null, expiry))
        val firstIds = (0 until first.payload!!.getJSONArray("messages").length())
            .map { first.payload.getJSONArray("messages").getJSONObject(it).getString("id") }
        val next = first.payload.getString("next")
        val second = executor(store = store).execute(PhoneCommand.Search("b", "needle", 2, next, expiry))
        val secondIds = (0 until second.payload!!.getJSONArray("messages").length())
            .map { second.payload.getJSONArray("messages").getJSONObject(it).getString("id") }
        assertEquals(listOf("sms:s3", "sms:s4"), firstIds)
        assertEquals(listOf("sms:s1", "sms:s2"), secondIds)
        assertEquals(emptySet<String>(), firstIds.toSet().intersect(secondIds.toSet()))
        assertEquals(false, second.payload.has("next"))
    }

    @Test
    fun equalTimestampCursorPagesWithoutGap() = runBlocking {
        val rows = listOf(
            MessageRow("s3", 42, "in", null, "3", 1_000, emptyList()),
            MessageRow("s2", 42, "in", null, "2", 1_000, emptyList()),
            MessageRow("s1", 42, "in", null, "1", 1_000, emptyList()),
        )
        val store = FakeMessageStore(messageRows = rows)
        val first = executor(store = store).execute(PhoneCommand.Messages("a", "sms:42", 2, null, expiry))
        val firstMessages = first.payload!!.getJSONArray("messages")
        assertEquals(listOf("sms:s2", "sms:s3"), (0 until firstMessages.length()).map { firstMessages.getJSONObject(it).getString("id") })
        val next = first.payload!!.getString("next")
        assertEquals("1000:sms:s2", next)
        val second = executor(store = store).execute(PhoneCommand.Messages("b", "sms:42", 2, next, expiry))
        val secondMessages = second.payload!!.getJSONArray("messages")
        assertEquals(listOf("sms:s1"), (0 until secondMessages.length()).map { secondMessages.getJSONObject(it).getString("id") })
        assertEquals(false, second.payload!!.has("next"))
    }

    @Test
    fun isoBeforeExcludesRowsAtExactlyThatTime() = runBlocking {
        val store = FakeMessageStore(messageRows = listOf(
            MessageRow("s2", 42, "in", null, "same", 1_000, emptyList()),
            MessageRow("s1", 42, "in", null, "old", 999, emptyList()),
        ))
        val result = executor(store = store).execute(
            PhoneCommand.Messages("m", "sms:42", 50, "1970-01-01T00:00:01Z", expiry),
        )
        assertEquals("s1", result.payload!!.getJSONArray("messages").getJSONObject(0).getString("id").removePrefix("sms:"))
    }

    @Test
    fun searchPassesBoundaryAndUsesMessagesEnvelope() = runBlocking {
        val store = FakeMessageStore(messageRows = listOf(MessageRow("s1", 42, "in", null, "hello", 1_000, emptyList())))
        val result = executor(store = store).execute(PhoneCommand.Search("s", "hello", 30, "1000:sms:s2", expiry))
        assertEquals("hello", result.payload!!.getJSONArray("messages").getJSONObject(0).getString("body"))
        assertEquals(Boundary(1_000, "s2"), store.lastSearchBoundary)
    }

    @Test
    fun explicitIdPrecedesContactResolutionAndDialableNumberUsesDirectLookup() = runBlocking {
        val contacts = FakeContactResolver(listOf(ContactMatch("Mum", "+15551212")))
        val store = FakeMessageStore(directThreads = listOf(42L to Participant("Mum", "+15551212")))
        executor(store = store, contacts = contacts).execute(PhoneCommand.Messages("a", "sms:42", 50, null, expiry))
        assertTrue(contacts.resolvedNames.isEmpty())
        executor(store = store, contacts = contacts).execute(PhoneCommand.Messages("b", "+1 (555) 1212", 50, null, expiry))
        assertTrue(contacts.resolvedNames.isEmpty())
        assertEquals(listOf(listOf("+1 (555) 1212")), store.directCalls)
    }

    @Test
    fun contactNameFoundOnlyInGroupThreadsReturnsNotFound() = runBlocking {
        val contacts = FakeContactResolver(listOf(ContactMatch("Mum", "+15551212")))
        val store = FakeMessageStore(directThreads = emptyList())
        val result = executor(store = store, contacts = contacts).execute(
            PhoneCommand.Messages("m", "Mum", 50, null, expiry),
        )
        assertEquals(PhoneResult("m", "error", "not found"), result)
    }

    @Test
    fun nameResolvesDirectThreadsAndReportsNotFoundOrAmbiguous() = runBlocking {
        val contacts = FakeContactResolver(listOf(ContactMatch("Mum", "+15551212")))
        val store = FakeMessageStore(directThreads = listOf(42L to Participant("Mum", "+15551212")))
        executor(store = store, contacts = contacts).execute(PhoneCommand.Messages("a", "Mum", 50, null, expiry))
        assertEquals(listOf("Mum"), contacts.resolvedNames)
        store.directThreads = emptyList()
        assertEquals(PhoneResult("b", "error", "not found"), executor(store = store, contacts = contacts).execute(PhoneCommand.Messages("b", "Mum", 50, null, expiry)))
        store.directThreads = listOf(
            42L to Participant("Mum", "+15551212"),
            43L to Participant(null, "+15559876"),
        )
        assertEquals("ambiguous: sms:42 (Mum), sms:43 (+15559876)", executor(store = store, contacts = contacts).execute(PhoneCommand.Messages("c", "Mum", 50, null, expiry)).detail)
    }

    private fun executor(
        sms: FakeSmsSender = FakeSmsSender(),
        contacts: FakeContactResolver = FakeContactResolver(),
        launcher: FakeIntentLauncher = FakeIntentLauncher(),
        store: FakeMessageStore = FakeMessageStore(),
        now: Instant = Instant.parse("2026-09-09T00:00:00Z"),
    ) = CommandExecutor(sms, contacts, launcher, store, FixedClock(now))

    private fun smsCommand(to: String = "+15551212") = PhoneCommand.Sms("id", to, "hello", expiry)

    private fun serializedMessages(rows: List<MessageRow>, next: String?): Int {
        val messages = JSONArray()
        rows.forEach { row ->
            val message = JSONObject()
                .put("id", "sms:${row.id}")
                .put("conversation", "sms:${row.threadId}")
                .put("channel", "sms")
                .put("direction", row.direction)
                .put("body", row.body.take(2_000))
                .put("at", Instant.ofEpochMilli(row.atMs).toString())
            row.from?.let {
                val from = JSONObject().put("number", it.number)
                it.name?.let { name -> from.put("name", name) }
                message.put("from", from)
            }
            val attachments = JSONArray()
            row.attachments.forEach {
                val attachment = JSONObject().put("content_type", it.contentType)
                it.name?.let { name -> attachment.put("name", name) }
                attachments.put(attachment)
            }
            message.put("attachments", attachments)
            if (row.body.length > 2_000) message.put("truncated", true)
            messages.put(message)
        }
        val payload = JSONObject().put("messages", messages)
        next?.let { payload.put("next", it) }
        return payload.toString().toByteArray(StandardCharsets.UTF_8).size
    }
    private fun openCommand() = PhoneCommand.Open("open", "https://example.com", expiry)

    private class FixedClock(private val value: Instant) : Clock {
        override fun now(): Instant = value
    }

    private class FakeSmsSender(
        var outcome: SmsSendOutcome = SmsSendOutcome.Sent,
        override val hasPermission: Boolean = true,
    ) : SmsSender {
        val calls = mutableListOf<Pair<String, String>>()
        override suspend fun send(number: String, body: String, deadline: Instant): SmsSendOutcome {
            calls += number to body
            return outcome
        }
    }

    private class FakeContactResolver(
        private val matches: List<ContactMatch> = emptyList(),
        override val hasPermission: Boolean = true,
    ) : ContactResolver {
        val resolvedNames = mutableListOf<String>()
        override fun resolve(name: String): List<ContactMatch> {
            resolvedNames += name
            return matches
        }
    }

    private class FakeMessageStore(
        override val hasPermission: Boolean = true,
        var conversationRows: List<ConversationRow> = emptyList(),
        var messageRows: List<MessageRow> = emptyList(),
        var directThreads: List<Pair<Long, Participant>> = emptyList(),
    ) : MessageStore {
        var conversationsCalls = 0
        var messagesCalls = 0
        var searchCalls = 0
        var directCalls = mutableListOf<List<String>>()
        var lastSearchBoundary: Boundary? = null
        override fun conversations(limit: Int): List<ConversationRow> {
            conversationsCalls += 1
            return conversationRows.take(limit)
        }
        override fun messages(threadId: Long, limit: Int, before: Boundary?): List<MessageRow> {
            messagesCalls += 1
            return messageRows.filter { before == null || it.atMs < before.timeMs || (it.atMs == before.timeMs && before.id != null && it.id < before.id) }.take(limit)
        }
        override fun search(query: String, limit: Int, before: Boundary?): List<MessageRow> {
            searchCalls += 1
            lastSearchBoundary = before
            return messageRows.filter { before == null || it.atMs < before.timeMs || (it.atMs == before.timeMs && before.id != null && it.id < before.id) }.take(limit)
        }
        override fun directThreadsFor(numbers: List<String>): List<Pair<Long, Participant>> {
            directCalls += numbers
            return directThreads
        }
    }

    private class FakeIntentLauncher(
        private val overlay: Boolean = true,
        private val exception: Exception? = null,
    ) : IntentLauncher {
        val uris = mutableListOf<String>()
        override fun canDrawOverlays(): Boolean = overlay
        override fun launch(uri: String) {
            uris += uri
            exception?.let { throw it }
        }
    }
}

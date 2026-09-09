package gg.savecraft.mentat.core

import android.content.ActivityNotFoundException
import java.time.Instant
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

    private fun executor(
        sms: FakeSmsSender = FakeSmsSender(),
        contacts: FakeContactResolver = FakeContactResolver(),
        launcher: FakeIntentLauncher = FakeIntentLauncher(),
        now: Instant = Instant.parse("2026-09-09T00:00:00Z"),
    ) = CommandExecutor(sms, contacts, launcher, FixedClock(now))

    private fun smsCommand(to: String = "+15551212") = PhoneCommand.Sms("id", to, "hello", expiry)
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
        override fun resolve(name: String): List<ContactMatch> = matches
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

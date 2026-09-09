package gg.savecraft.mentat.session

import android.Manifest
import android.app.Activity
import android.app.Application
import android.app.PendingIntent
import android.os.Looper
import androidx.test.core.app.ApplicationProvider
import gg.savecraft.mentat.core.Clock
import gg.savecraft.mentat.core.SmsSendOutcome
import java.time.Instant
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows
import org.robolectric.annotation.Config

@Config(sdk = [35])
@RunWith(RobolectricTestRunner::class)
class AndroidSmsSenderTest {
    @Test
    fun successfulSentCallbacksAggregateAsSent() = runBlocking {
        val application = setupApplication()
        val gateway = FakeSmsGateway(arrayListOf("hello"))
        val pending = async(Dispatchers.Default) {
            AndroidSmsSender(application, gateway = gateway).send(
                "+15551212",
                "hello",
                Instant.now().plusSeconds(5),
            )
        }
        awaitSentIntents(gateway, expectedCalls = 1)
        gateway.sentIntentsByCall[0][0].send(application, Activity.RESULT_OK, null)
        Shadows.shadowOf(Looper.getMainLooper()).idle()
        assertEquals(SmsSendOutcome.Sent, pending.await())
    }

    @Test
    fun failedPartReturnsItsResultCode() = runBlocking {
        val application = setupApplication()
        val gateway = FakeSmsGateway(arrayListOf("one", "two"))
        val pending = async(Dispatchers.Default) {
            AndroidSmsSender(application, gateway = gateway).send(
                "+15551212",
                "long message",
                Instant.now().plusSeconds(5),
            )
        }
        awaitSentIntents(gateway, expectedCalls = 1)
        gateway.sentIntentsByCall[0][0].send(application, Activity.RESULT_OK, null)
        gateway.sentIntentsByCall[0][1].send(application, 17, null)
        Shadows.shadowOf(Looper.getMainLooper()).idle()
        assertEquals(SmsSendOutcome.Failed(17), pending.await())
    }

    @Test
    fun deadlineReachedAfterDivideDoesNotSend() = runBlocking {
        val application = setupApplication()
        val deadline = Instant.parse("2026-09-10T00:00:00Z")
        val clock = MutableClock(deadline.minusMillis(1))
        val gateway = FakeSmsGateway(arrayListOf("hello")) {
            clock.current = deadline
        }

        val result = AndroidSmsSender(application, clock = clock, gateway = gateway).send(
            "+15551212",
            "hello",
            deadline,
        )

        assertEquals(SmsSendOutcome.Expired, result)
        assertEquals(0, gateway.sendCalls)
        assertTrue(Shadows.shadowOf(application).registeredReceivers.isEmpty())
    }

    @Test
    fun outstandingPartHitsDeadlineAndUnregistersReceiver() = runBlocking {
        val application = setupApplication()
        val gateway = FakeSmsGateway(arrayListOf("one", "two"))
        val result = AndroidSmsSender(application, gateway = gateway).send(
            "+15551212",
            "long message",
            Instant.now().plusMillis(100),
        )
        assertEquals(SmsSendOutcome.Unknown, result)
        assertTrue(Shadows.shadowOf(application).registeredReceivers.isEmpty())
    }

    @Test
    fun timedOutSendDoesNotReplayWhenSameSenderSendsAgain() = runBlocking {
        val application = setupApplication()
        val gateway = FakeSmsGateway(arrayListOf("one", "two"))
        val sender = AndroidSmsSender(application, gateway = gateway)

        val first = async(Dispatchers.Default) {
            sender.send(
                "+15551212",
                "first message",
                Instant.now().plusMillis(100),
            )
        }
        awaitSentIntents(gateway, expectedCalls = 1)
        assertEquals(SmsSendOutcome.Unknown, first.await())
        assertTrue(Shadows.shadowOf(application).registeredReceivers.isEmpty())

        val second = async(Dispatchers.Default) {
            sender.send(
                "+15551212",
                "second message",
                Instant.now().plusSeconds(5),
            )
        }
        awaitSentIntents(gateway, expectedCalls = 2)
        gateway.sentIntentsByCall[1].forEach { intent ->
            intent.send(application, Activity.RESULT_OK, null)
        }
        Shadows.shadowOf(Looper.getMainLooper()).idle()

        assertEquals(SmsSendOutcome.Sent, second.await())
        assertEquals(2, gateway.sendCalls)
        assertTrue(Shadows.shadowOf(application).registeredReceivers.isEmpty())
    }

    private suspend fun awaitSentIntents(gateway: FakeSmsGateway, expectedCalls: Int) {
        repeat(100) {
            if (gateway.sentIntentsByCall.size >= expectedCalls) return
            delay(10)
        }
        error("SMS gateway was not called")
    }

    private fun setupApplication(): Application {
        val application = ApplicationProvider.getApplicationContext<Application>()
        Shadows.shadowOf(application).grantPermissions(Manifest.permission.SEND_SMS)
        Shadows.shadowOf(application).clearRegisteredReceivers()
        return application
    }

    private class MutableClock(var current: Instant) : Clock {
        override fun now(): Instant = current
    }

    private class FakeSmsGateway(
        private val parts: ArrayList<String>,
        private val onDivide: (() -> Unit)? = null,
    ) : SmsGateway {
        val sentIntentsByCall = mutableListOf<ArrayList<PendingIntent>>()
        val sendCalls: Int
            get() = sentIntentsByCall.size

        override fun divideMessage(body: String): ArrayList<String> {
            onDivide?.invoke()
            return parts
        }

        override fun send(number: String, parts: ArrayList<String>, sentIntents: ArrayList<PendingIntent>) {
            sentIntentsByCall += sentIntents
        }
    }
}

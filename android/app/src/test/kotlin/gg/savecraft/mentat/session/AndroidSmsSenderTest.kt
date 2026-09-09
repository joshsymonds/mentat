package gg.savecraft.mentat.session

import android.Manifest
import android.app.Activity
import android.app.Application
import android.app.PendingIntent
import android.os.Looper
import androidx.test.core.app.ApplicationProvider
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
        awaitSentIntents(gateway)
        gateway.sentIntents!![0].send(application, Activity.RESULT_OK, null)
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
        awaitSentIntents(gateway)
        gateway.sentIntents!![0].send(application, Activity.RESULT_OK, null)
        gateway.sentIntents!![1].send(application, 17, null)
        Shadows.shadowOf(Looper.getMainLooper()).idle()
        assertEquals(SmsSendOutcome.Failed(17), pending.await())
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

    private suspend fun awaitSentIntents(gateway: FakeSmsGateway) {
        repeat(100) {
            if (gateway.sentIntents != null) return
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

    private class FakeSmsGateway(
        private val parts: ArrayList<String>,
    ) : SmsGateway {
        var sentIntents: ArrayList<PendingIntent>? = null

        override fun divideMessage(body: String): ArrayList<String> = parts

        override fun send(number: String, parts: ArrayList<String>, sentIntents: ArrayList<PendingIntent>) {
            this.sentIntents = sentIntents
        }
    }
}

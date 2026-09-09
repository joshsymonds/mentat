package gg.savecraft.mentat.session

import android.app.Application
import android.app.NotificationManager
import android.content.Intent
import android.content.pm.ServiceInfo
import androidx.test.core.app.ApplicationProvider
import gg.savecraft.mentat.core.CommandExecutor
import gg.savecraft.mentat.core.CommandStream
import gg.savecraft.mentat.core.PhoneCommand
import gg.savecraft.mentat.core.PhoneResult
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows
import org.robolectric.annotation.Config

@Config(sdk = [35])
@RunWith(RobolectricTestRunner::class)
class PhoneCommandServiceTest {
    @Test
    fun startsForegroundOnPhoneBridgeChannelAndStopsStreamOnDestroy() {
        val stream = FakeCommandStream()
        FakePhoneCommandService.stream = stream
        val service = Robolectric.buildService(FakePhoneCommandService::class.java).create().get()

        service.onStartCommand(Intent(), 0, 1)

        val manager = service.getSystemService(NotificationManager::class.java)
        assertTrue(manager.notificationChannels.any { it.id == "phone-bridge" })
        assertTrue(Shadows.shadowOf(service).isForegroundStopped.not())
        service.onDestroy()
        assertTrue(stream.closed)
    }

    @Test
    fun manifestDeclaresSpecialUseForegroundService() {
        val application = ApplicationProvider.getApplicationContext<Application>()
        val info = application.packageManager.getServiceInfo(
            android.content.ComponentName(application, PhoneCommandService::class.java),
            0,
        )
        assertEquals(ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE, info.foregroundServiceType)
    }

    class FakeCommandStream : CommandStream {
        var closed = false
        override suspend fun run(handler: suspend (PhoneCommand) -> PhoneResult) = Unit
        override fun close() {
            closed = true
        }
    }

    class FakePhoneCommandService : PhoneCommandService() {
        override fun commandStream(): CommandStream = stream

        companion object {
            lateinit var stream: FakeCommandStream
        }
    }
}

package gg.savecraft.mentat.session

import android.app.Application
import android.content.Intent
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows
import org.robolectric.annotation.Config

@Config(sdk = [35])
@RunWith(RobolectricTestRunner::class)
class BootReceiverTest {
    @Test
    fun bootCompletedStartsPhoneCommandService() {
        val application = ApplicationProvider.getApplicationContext<Application>()
        BootReceiver().onReceive(application, Intent(Intent.ACTION_BOOT_COMPLETED))
        assertEquals(
            PhoneCommandService::class.java.name,
            Shadows.shadowOf(application).peekNextStartedService()?.component?.className,
        )

        val receivers = application.packageManager.queryBroadcastReceivers(
            Intent(Intent.ACTION_BOOT_COMPLETED).setPackage(application.packageName),
            0,
        )
        assertEquals(
            listOf(BootReceiver::class.java.name),
            receivers.mapNotNull { it.activityInfo?.name },
        )
    }
}

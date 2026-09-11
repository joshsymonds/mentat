package gg.savecraft.mentat

import android.Manifest
import android.app.Application
import android.content.ComponentName
import android.content.Intent
import android.content.pm.PackageManager
import android.provider.Settings
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertEquals
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows
import org.robolectric.annotation.Config

@Config(sdk = [35])
@RunWith(RobolectricTestRunner::class)
class MainActivityTest {
    private lateinit var application: Application

    @Before
    fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        Shadows.shadowOf(application).denyPermissions(
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION,
            Manifest.permission.ACCESS_BACKGROUND_LOCATION,
        )
    }

    @Test
    fun firstRunRequestsForegroundFineLocationOnly() {
        Shadows.shadowOf(application).grantPermissions(Manifest.permission.RECORD_AUDIO)

        val activity = Robolectric.buildActivity(MainActivity::class.java).create().start().get()
        val request = requireNotNull(Shadows.shadowOf(activity).lastRequestedPermission)

        assertArrayEquals(arrayOf(Manifest.permission.ACCESS_FINE_LOCATION), request.requestedPermissions)
    }

    @Test
    fun grantingForegroundLocationOffersBackgroundSettingsPath() {
        Shadows.shadowOf(application).grantPermissions(Manifest.permission.RECORD_AUDIO)

        val activity = Robolectric.buildActivity(MainActivity::class.java).create().start().get()
        val request = requireNotNull(Shadows.shadowOf(activity).lastRequestedPermission)
        activity.onRequestPermissionsResult(
            request.requestCode,
            request.requestedPermissions,
            intArrayOf(PackageManager.PERMISSION_GRANTED),
        )

        val settingsIntent = Shadows.shadowOf(activity).nextStartedActivity
        assertEquals(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, settingsIntent?.action)
        assertEquals("package:${application.packageName}", settingsIntent?.dataString)
    }

    /**
     * "Open Mentat" and the home screen both resolve the package's launch intent, which
     * only exists when some activity carries MAIN + LAUNCHER. That has to be the
     * unguarded entry: the assist activity's signature permission would refuse the
     * launcher and every assistant that tries to open the app.
     */
    @Test
    fun thePackageLaunchIntentResolvesToTheUnguardedMainActivity() {
        val launch = application.packageManager.getLaunchIntentForPackage(application.packageName)

        assertEquals(MainActivity::class.java.name, launch?.component?.className)
        val info = application.packageManager.getActivityInfo(
            ComponentName(application, MainActivity::class.java),
            0,
        )
        assertTrue(info.exported)
        assertNull(info.permission)
    }

    @Test
    fun launchingFromTheLauncherStartsTheVoiceService() {
        Shadows.shadowOf(application).grantPermissions(Manifest.permission.RECORD_AUDIO)
        val intent = Intent(Intent.ACTION_MAIN)
            .addCategory(Intent.CATEGORY_LAUNCHER)
            .setClass(application, MainActivity::class.java)

        Robolectric.buildActivity(MainActivity::class.java, intent).create()

        assertEquals(
            "gg.savecraft.mentat.session.VoiceSessionService",
            Shadows.shadowOf(application).peekNextStartedService()?.component?.className,
        )
    }
}

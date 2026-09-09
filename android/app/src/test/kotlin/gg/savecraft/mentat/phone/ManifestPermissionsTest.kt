package gg.savecraft.mentat.phone

import android.app.Application
import android.content.pm.PackageManager
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@Config(sdk = [35])
@RunWith(RobolectricTestRunner::class)
class ManifestPermissionsTest {
    @Test
    fun manifestDeclaresAlarmAndLocationPermissions() {
        val application = ApplicationProvider.getApplicationContext<Application>()
        val permissions = application.packageManager
            .getPackageInfo(application.packageName, PackageManager.GET_PERMISSIONS)
            .requestedPermissions
            ?.toSet()
            .orEmpty()

        assertTrue("alarm permission missing", permissions.contains("com.android.alarm.permission.SET_ALARM"))
        assertTrue("fine location permission missing", permissions.contains("android.permission.ACCESS_FINE_LOCATION"))
        assertTrue("coarse location permission missing", permissions.contains("android.permission.ACCESS_COARSE_LOCATION"))
    }
}

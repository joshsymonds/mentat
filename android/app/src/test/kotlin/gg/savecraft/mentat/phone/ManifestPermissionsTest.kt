package gg.savecraft.mentat.phone

import android.app.Application
import android.content.ComponentName
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import gg.savecraft.mentat.session.PhoneCommandService
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertEquals
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
        assertTrue("background location permission missing", permissions.contains("android.permission.ACCESS_BACKGROUND_LOCATION"))
        assertTrue("location foreground service permission missing", permissions.contains("android.permission.FOREGROUND_SERVICE_LOCATION"))

        val serviceInfo = application.packageManager.getServiceInfo(
            ComponentName(application, PhoneCommandService::class.java),
            0,
        )
        assertEquals(
            ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE or ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION,
            serviceInfo.foregroundServiceType,
        )
    }
}

package gg.savecraft.mentat.core

import android.location.Location
import android.os.SystemClock
import androidx.test.core.app.ApplicationProvider
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@Config(sdk = [35])
@RunWith(RobolectricTestRunner::class)
class PhoneLocationTest {
    @Test
    fun freshCurrentFixIsReturnedWithoutMainLooperDrain() {
        val location = location(51.5, -0.12, 8f, ageSeconds = 0)
        val source = FakeLocationSource(currentLocation = location)
        val phoneLocation = PhoneLocation(
            ApplicationProvider.getApplicationContext(),
            source,
        )
        grantLocation()

        val response = requireNotNull(phoneLocation.get())

        assertEquals(51.5, response.getDouble("lat"), 0.00001)
        assertEquals(-0.12, response.getDouble("lng"), 0.00001)
        assertEquals(8.0, response.getDouble("accuracy_m"), 0.00001)
        assertEquals(0L, response.getLong("age_s"))
        assertEquals(5_000L, source.currentTimeoutMs)
        assertTrue(source.currentCalled)
    }

    @Test
    fun timedOutCurrentFixFallsBackToRecentLastKnown() {
        val source = FakeLocationSource(
            currentLocation = null,
            lastKnownLocation = location(51.5, -0.12, 10f, ageSeconds = 300),
        )
        val phoneLocation = PhoneLocation(ApplicationProvider.getApplicationContext(), source)
        grantLocation()

        val response = requireNotNull(phoneLocation.get())

        assertEquals(300L, response.getLong("age_s"))
        assertEquals(51.5, response.getDouble("lat"), 0.00001)
    }

    @Test
    fun staleLastKnownLocationIsRejected() {
        val source = FakeLocationSource(lastKnownLocation = location(51.5, -0.12, 10f, ageSeconds = 900))
        val phoneLocation = PhoneLocation(ApplicationProvider.getApplicationContext(), source)
        grantLocation()

        assertEquals(null, phoneLocation.get())
    }

    @Test
    fun recentReturnsAFreshLastKnownFixWithoutWaitingOnACurrentOne() {
        val source = FakeLocationSource(
            currentThrows = AssertionError("waited on a current fix"),
            lastKnownLocation = location(47.6, -122.3, 15f, ageSeconds = 120),
        )
        val phoneLocation = PhoneLocation(ApplicationProvider.getApplicationContext(), source)
        grantLocation()

        val response = requireNotNull(phoneLocation.recent())

        assertEquals(47.6, response.getDouble("lat"), 0.00001)
        assertEquals(120L, response.getLong("age_s"))
        assertFalse(source.currentCalled)
    }

    @Test
    fun recentRejectsStaleFixesAndMissingPermission() {
        val stale = FakeLocationSource(lastKnownLocation = location(47.6, -122.3, 15f, ageSeconds = 900))
        grantLocation()
        assertEquals(null, PhoneLocation(ApplicationProvider.getApplicationContext(), stale).recent())

        val denied = FakeLocationSource(lastKnownLocation = location(47.6, -122.3, 15f, ageSeconds = 1))
        assertEquals(null, PhoneLocation(denied, permissionGranted = { false }).recent())
        assertFalse(denied.lastKnownCalled)
    }

    @Test
    fun missingPermissionRejectsWithoutTouchingSource() {
        val source = FakeLocationSource(currentThrows = AssertionError("source touched"))
        val phoneLocation = PhoneLocation(ApplicationProvider.getApplicationContext(), source)

        assertEquals(null, phoneLocation.get())
        assertFalse(source.currentCalled)
        assertFalse(source.lastKnownCalled)
    }

    private fun grantLocation() {
        val application = ApplicationProvider.getApplicationContext<android.app.Application>()
        org.robolectric.Shadows.shadowOf(application).grantPermissions(
            android.Manifest.permission.ACCESS_COARSE_LOCATION,
        )
    }

    private fun location(lat: Double, lng: Double, accuracy: Float, ageSeconds: Long): Location =
        Location("test").apply {
            latitude = lat
            longitude = lng
            this.accuracy = accuracy
            elapsedRealtimeNanos = SystemClock.elapsedRealtimeNanos() - ageSeconds * 1_000_000_000L
        }

    private class FakeLocationSource(
        private val currentLocation: Location? = null,
        private val lastKnownLocation: Location? = null,
        private val currentThrows: Throwable? = null,
    ) : LocationSource {
        var currentTimeoutMs: Long? = null
        var currentCalled = false
        var lastKnownCalled = false

        override fun current(timeoutMs: Long): Location? {
            currentCalled = true
            currentTimeoutMs = timeoutMs
            currentThrows?.let { throw it }
            return currentLocation
        }

        override fun lastKnown(): Location? {
            lastKnownCalled = true
            return lastKnownLocation
        }
    }
}

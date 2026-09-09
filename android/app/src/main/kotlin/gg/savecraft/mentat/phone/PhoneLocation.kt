package gg.savecraft.mentat.phone

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.location.Location
import android.location.LocationManager
import android.os.CancellationSignal
import android.os.SystemClock
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executor
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference
import org.json.JSONObject

interface LocationSource {
    fun current(timeoutMs: Long): Location?

    fun lastKnown(): Location?
}

class PhoneLocation private constructor(
    private val source: LocationSource,
    private val permissionChecker: () -> Boolean,
    constructorMarker: Unit,
) {
    constructor(context: Context) : this(
        source = AndroidLocationSource(context),
        permissionChecker = {
            val permissionContext = context.applicationContext
            permissionContext.checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) ==
                PackageManager.PERMISSION_GRANTED ||
                permissionContext.checkSelfPermission(Manifest.permission.ACCESS_COARSE_LOCATION) ==
                PackageManager.PERMISSION_GRANTED
        },
        constructorMarker = Unit,
    )

    constructor(context: Context, source: LocationSource) : this(
        source = source,
        permissionChecker = {
            val permissionContext = context.applicationContext
            permissionContext.checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) ==
                PackageManager.PERMISSION_GRANTED ||
                permissionContext.checkSelfPermission(Manifest.permission.ACCESS_COARSE_LOCATION) ==
                PackageManager.PERMISSION_GRANTED
        },
        constructorMarker = Unit,
    )

    constructor(source: LocationSource, permissionGranted: () -> Boolean) : this(
        source = source,
        permissionChecker = permissionGranted,
        constructorMarker = Unit,
    )

    fun get(): String {
        if (!permissionChecker()) {
            throw PhoneRpcException(PhoneRpcCodes.LOCATION_UNAVAILABLE, "location unavailable")
        }
        val current = runCatching { source.current(CURRENT_TIMEOUT_MS) }.getOrNull()
        if (current != null) {
            return encode(current)
        }
        val lastKnown = runCatching { source.lastKnown() }.getOrNull()
        if (lastKnown != null) {
            val ageNanos = ageNanos(lastKnown)
            if (ageNanos <= MAX_LAST_KNOWN_AGE_NANOS) {
                return encode(lastKnown, ageNanos / NANOS_PER_SECOND)
            }
        }
        throw PhoneRpcException(PhoneRpcCodes.LOCATION_UNAVAILABLE, "location unavailable")
    }

    private fun encode(location: Location, ageSeconds: Long = ageSeconds(location)): String =
        JSONObject()
            .put("lat", location.latitude)
            .put("lng", location.longitude)
            .put("accuracy_m", location.accuracy.toDouble())
            .put("age_s", ageSeconds)
            .toString()

    private fun ageSeconds(location: Location): Long = ageNanos(location) / NANOS_PER_SECOND

    private fun ageNanos(location: Location): Long {
        val elapsedNanos = SystemClock.elapsedRealtimeNanos() - location.elapsedRealtimeNanos
        return if (elapsedNanos <= 0L) 0L else elapsedNanos
    }

    private companion object {
        const val CURRENT_TIMEOUT_MS = 5_000L
        const val MAX_LAST_KNOWN_AGE_NANOS = 600L * 1_000_000_000L
        const val NANOS_PER_SECOND = 1_000_000_000L
    }
}

private class AndroidLocationSource(context: Context) : LocationSource {
    private val locationManager = context.applicationContext.getSystemService(LocationManager::class.java)

    override fun current(timeoutMs: Long): Location? {
        val result = AtomicReference<Location?>()
        val completed = CountDownLatch(1)
        val cancellation = CancellationSignal()
        try {
            locationManager.getCurrentLocation(
                LocationManager.FUSED_PROVIDER,
                cancellation,
                Executor { it.run() },
            ) {
                result.set(it)
                completed.countDown()
            }
            completed.await(timeoutMs, TimeUnit.MILLISECONDS)
        } catch (_: SecurityException) {
            return null
        } catch (_: IllegalArgumentException) {
            return null
        } finally {
            cancellation.cancel()
        }
        return result.get()
    }

    override fun lastKnown(): Location? = try {
        locationManager.getLastKnownLocation(LocationManager.FUSED_PROVIDER)
    } catch (_: SecurityException) {
        null
    } catch (_: IllegalArgumentException) {
        null
    }
}

package gg.savecraft.mentat

import android.Manifest
import android.app.Application
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.PowerManager
import android.provider.Settings
import androidx.test.core.app.ApplicationProvider
import gg.savecraft.mentat.core.SessionState
import gg.savecraft.mentat.session.BootReceiver
import gg.savecraft.mentat.session.PhoneCommandService
import gg.savecraft.mentat.ui.detailRes
import gg.savecraft.mentat.ui.titleRes
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows
import org.robolectric.annotation.Config
import org.robolectric.shadows.ShadowSettings

@Config(sdk = [35])
@RunWith(RobolectricTestRunner::class)
class AssistActivityTest {
    private lateinit var application: Application

    @Before
    fun resetPermission() {
        application = ApplicationProvider.getApplicationContext()
        // Every permission the activity may request starts denied.
        Shadows.shadowOf(application).denyPermissions(
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.SEND_SMS,
            Manifest.permission.READ_CONTACTS,
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION,
        )
    }

    /**
     * With every permission missing, the microphone is asked first and the location
     * request follows its grant. The location-to-phone step is covered by
     * [phonePermissionsAreRequestedWhenMissing]: a request chained behind one answered
     * through onRequestPermissionsResult is cancelled by the platform's one-at-a-time
     * rule, so it cannot be answered in the same test.
     */
    @Test
    fun allDeniedPermissionsRequestAudioThenLocation() {
        val activity = Robolectric.buildActivity(AssistActivity::class.java).create().start().get()

        val audioRequest = requireNotNull(Shadows.shadowOf(activity).lastRequestedPermission)
        assertArrayEquals(arrayOf(Manifest.permission.RECORD_AUDIO), audioRequest.requestedPermissions)

        activity.onRequestPermissionsResult(
            audioRequest.requestCode,
            audioRequest.requestedPermissions,
            intArrayOf(PackageManager.PERMISSION_GRANTED),
        )

        val startedServices = startedServiceClassNames()
        assertTrue(startedServices.contains(VOICE_SERVICE))
        assertTrue(startedServices.contains(PHONE_SERVICE))
        assertTrue(Shadows.shadowOf(application).boundServiceConnections.isNotEmpty())
        val locationRequest = requireNotNull(Shadows.shadowOf(activity).lastRequestedPermission)
        assertArrayEquals(
            arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION),
            locationRequest.requestedPermissions,
        )
    }

    @Test
    fun deniedRecordAudioPermissionRequestsPhonePermissionsAfterAudioResult() {
        val activity = Robolectric.buildActivity(AssistActivity::class.java).create().start().get()

        val audioRequest = requireNotNull(Shadows.shadowOf(activity).lastRequestedPermission)
        assertArrayEquals(arrayOf(Manifest.permission.RECORD_AUDIO), audioRequest.requestedPermissions)

        activity.onRequestPermissionsResult(
            audioRequest.requestCode,
            audioRequest.requestedPermissions,
            intArrayOf(PackageManager.PERMISSION_DENIED),
        )

        assertEquals(SessionState.Failed("Permission denied"), activity.uiState.value)
        val phoneRequest = requireNotNull(Shadows.shadowOf(activity).lastRequestedPermission)
        assertArrayEquals(
            arrayOf(Manifest.permission.SEND_SMS, Manifest.permission.READ_CONTACTS),
            phoneRequest.requestedPermissions,
        )
    }

    @Test
    fun deniedRecordAudioPermissionDoesNotStartOrBindVoiceService() {
        grantPhonePermissions()

        // The activity result callback only fires once the activity is started, and the
        // deprecated permission hook is what ComponentActivity routes into the result
        // registry — Robolectric has no other way to answer a permission request.
        val activity = Robolectric.buildActivity(AssistActivity::class.java).create().start().get()

        val request = Shadows.shadowOf(activity).lastRequestedPermission
        activity.onRequestPermissionsResult(
            request.requestCode,
            request.requestedPermissions,
            intArrayOf(PackageManager.PERMISSION_DENIED),
        )

        assertEquals(PHONE_SERVICE, startedServiceClassName())
        assertTrue(Shadows.shadowOf(application).boundServiceConnections.isEmpty())
        assertEquals(SessionState.Failed("Permission denied"), activity.uiState.value)
    }

    @Test
    fun phonePermissionsAreRequestedWhenMissing() {
        grantRecordAudio()

        // Result callbacks bind on start, so the chained request needs a started activity.
        // A result delivered through onRequestPermissionsResult leaves the platform's
        // one-request-at-a-time flag set, so the follow-up request is recorded by the
        // shadow and then cancelled: lastRequestedPermission shows the attempt.
        val activity = Robolectric.buildActivity(AssistActivity::class.java).create().start().get()
        answerLocationRequest(activity)

        val request = Shadows.shadowOf(activity).lastRequestedPermission
        assertArrayEquals(
            arrayOf(Manifest.permission.SEND_SMS, Manifest.permission.READ_CONTACTS),
            request.requestedPermissions,
        )
    }

    @Test
    fun phonePermissionsAreNotRequestedWhenGranted() {
        grantRecordAudio()
        Shadows.shadowOf(application).grantPermissions(
            Manifest.permission.SEND_SMS,
            Manifest.permission.READ_CONTACTS,
        )

        val activity = Robolectric.buildActivity(AssistActivity::class.java).create().start().get()
        answerLocationRequest(activity)

        // The location request stays the last one: nothing followed it.
        val request = requireNotNull(Shadows.shadowOf(activity).lastRequestedPermission)
        assertArrayEquals(
            arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION),
            request.requestedPermissions,
        )
    }

    @Test
    fun overlayAndBatterySettingsAreOpenedOnceWhenMissing() {
        grantRecordAudio()
        ShadowSettings.setCanDrawOverlays(false)
        val powerManager = application.getSystemService(Context.POWER_SERVICE) as PowerManager
        Shadows.shadowOf(powerManager).setIgnoringBatteryOptimizations(application.packageName, false)

        val activity = Robolectric.buildActivity(AssistActivity::class.java).create().start().get()
        val intents = drainStartedActivities(activity)

        val overlayIntents = intents.filter { intent ->
            intent.action == Settings.ACTION_MANAGE_OVERLAY_PERMISSION
        }
        val batteryIntents = intents.filter { intent ->
            intent.action == Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS
        }
        assertEquals(1, overlayIntents.size)
        assertEquals(1, batteryIntents.size)
        assertEquals("package:${application.packageName}", overlayIntents.single().dataString)
        assertEquals("package:${application.packageName}", batteryIntents.single().dataString)
    }

    @Test
    fun settingsAreNotOpenedWhenGranted() {
        grantRecordAudio()
        ShadowSettings.setCanDrawOverlays(true)
        val powerManager = application.getSystemService(Context.POWER_SERVICE) as PowerManager
        Shadows.shadowOf(powerManager).setIgnoringBatteryOptimizations(application.packageName, true)

        val activity = Robolectric.buildActivity(AssistActivity::class.java).create().get()
        val intents = drainStartedActivities(activity)

        assertTrue(intents.none { intent ->
            intent.action == Settings.ACTION_MANAGE_OVERLAY_PERMISSION ||
                intent.action == Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS
        })
    }

    @Test
    fun phoneCommandServiceIsStartedOnLaunch() {
        grantRecordAudio()

        Robolectric.buildActivity(AssistActivity::class.java).create()

        assertTrue(startedServiceClassNames().contains(PHONE_SERVICE))
    }

    @Test
    fun manifestDeclaresThePhonePermissions() {
        val packageInfo = application.packageManager.getPackageInfo(
            application.packageName,
            PackageManager.GET_PERMISSIONS,
        )
        val permissions = packageInfo.requestedPermissions.orEmpty().toSet()
        assertTrue(permissions.contains(Manifest.permission.SEND_SMS))
        assertTrue(permissions.contains(Manifest.permission.READ_CONTACTS))
        assertTrue(permissions.contains(Manifest.permission.SYSTEM_ALERT_WINDOW))
        assertTrue(permissions.contains(Manifest.permission.REQUEST_IGNORE_BATTERY_OPTIMIZATIONS))
        assertTrue(permissions.contains(Manifest.permission.FOREGROUND_SERVICE_SPECIAL_USE))
        assertTrue(permissions.contains(Manifest.permission.RECEIVE_BOOT_COMPLETED))

        val receiverInfo = application.packageManager.getReceiverInfo(
            ComponentName(application, BootReceiver::class.java),
            0,
        )
        assertEquals(application.packageName, receiverInfo.packageName)

        val serviceInfo = application.packageManager.getServiceInfo(
            ComponentName(application, PhoneCommandService::class.java),
            PackageManager.GET_META_DATA,
        )
        assertEquals(
            android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE,
            serviceInfo.foregroundServiceType,
        )
    }

    @Test
    fun microphoneGrantStartsSessionBeforeRequestingLocationPermissions() {
        val activity = Robolectric.buildActivity(AssistActivity::class.java).create().start().get()
        val microphoneRequest = Shadows.shadowOf(activity).lastRequestedPermission

        activity.onRequestPermissionsResult(
            microphoneRequest.requestCode,
            microphoneRequest.requestedPermissions,
            intArrayOf(PackageManager.PERMISSION_GRANTED),
        )

        val locationRequest = Shadows.shadowOf(activity).lastRequestedPermission
        assertTrue(startedServiceClassNames().contains(VOICE_SERVICE))
        assertEquals(
            listOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION),
            locationRequest.requestedPermissions.toList(),
        )
    }

    @Test
    fun preciseLocationPermissionDecisionStartsVoiceService() {
        assertLocationPermissionDecisionStartsSession(
            intArrayOf(PackageManager.PERMISSION_GRANTED, PackageManager.PERMISSION_GRANTED),
        )
    }

    @Test
    fun approximateLocationPermissionDecisionStartsVoiceService() {
        assertLocationPermissionDecisionStartsSession(
            intArrayOf(PackageManager.PERMISSION_DENIED, PackageManager.PERMISSION_GRANTED),
        )
    }

    @Test
    fun deniedLocationPermissionDecisionStartsVoiceService() {
        assertLocationPermissionDecisionStartsSession(
            intArrayOf(PackageManager.PERMISSION_DENIED, PackageManager.PERMISSION_DENIED),
        )
    }

    private fun assertLocationPermissionDecisionStartsSession(results: IntArray) {
        grantRecordAudio()
        val activity = Robolectric.buildActivity(AssistActivity::class.java).create().start().get()
        val request = Shadows.shadowOf(activity).lastRequestedPermission

        assertEquals(
            listOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION),
            request.requestedPermissions.toList(),
        )
        activity.onRequestPermissionsResult(
            request.requestCode,
            request.requestedPermissions,
            results,
        )
        assertEquals(VOICE_SERVICE, startedServiceClassName())
        assertNull(Shadows.shadowOf(application).nextStoppedService)
        assertTrue(Shadows.shadowOf(application).boundServiceConnections.isNotEmpty())
    }

    @Test
    fun grantedRecordAudioPermissionStartsVoiceService() {
        grantRecordAudio()

        Robolectric.buildActivity(AssistActivity::class.java).create()

        assertEquals(VOICE_SERVICE, startedServiceClassName())
        assertTrue(startedServiceClassNames().contains(PHONE_SERVICE))
    }

    /**
     * The activity is exported for assist dispatch, so the platform caller check is the
     * only thing standing between a hostile app and a microphone session: a signature
     * permission no third-party app can hold, which the system uid bypasses by rule.
     */
    @Test
    fun theExportedAssistActivityIsGuardedBySignaturePermission() {
        val info = application.packageManager.getActivityInfo(
            ComponentName(application, AssistActivity::class.java),
            0,
        )

        assertTrue(info.exported)
        assertEquals("android.permission.BIND_VOICE_INTERACTION", info.permission)
    }

    @Test
    fun destroyingTheActivityStopsTheVoiceSession() {
        grantRecordAudio()
        val controller = Robolectric.buildActivity(AssistActivity::class.java).create()
        assertEquals(VOICE_SERVICE, startedServiceClassName())
        assertTrue(startedServiceClassNames().contains(PHONE_SERVICE))

        controller.destroy()

        assertEquals(VOICE_SERVICE, stoppedServiceClassName())
    }

    @Test
    fun destroyingTheActivityStopsASessionThatFailedAfterStarting() {
        grantRecordAudio()
        val controller = Robolectric.buildActivity(AssistActivity::class.java).create()
        val activity = controller.get()
        assertEquals(VOICE_SERVICE, startedServiceClassName())
        assertTrue(startedServiceClassNames().contains(PHONE_SERVICE))
        // A token or connect failure leaves the started service running, so the activity
        // still owes it a stop when it goes away.
        activity.failSession("Token request failed")

        controller.destroy()

        assertEquals(SessionState.Failed("Token request failed"), activity.uiState.value)
        assertEquals(VOICE_SERVICE, stoppedServiceClassName())
    }

    @Test
    fun destroyingAnEndedActivityDoesNotStopTheServiceTwice() {
        grantRecordAudio()
        val controller = Robolectric.buildActivity(AssistActivity::class.java).create()
        controller.get().endVoiceSession()
        assertEquals(VOICE_SERVICE, stoppedServiceClassName())

        controller.destroy()

        assertNull(Shadows.shadowOf(application).nextStoppedService)
    }

    @Test
    fun endBeforeTheServiceBindsStopsTheStartedVoiceService() {
        grantRecordAudio()
        val activity = Robolectric.buildActivity(AssistActivity::class.java).create().get()

        activity.endVoiceSession()

        assertEquals(VOICE_SERVICE, stoppedServiceClassName())
    }

    @Test
    fun foregroundServiceStartFailureFailsTheSession() {
        grantRecordAudio()

        val activity = Robolectric.buildActivity(UnstartableAssistActivity::class.java).create().get()

        assertEquals(SessionState.Failed("start not allowed"), activity.uiState.value)
        assertTrue(Shadows.shadowOf(application).boundServiceConnections.isEmpty())
    }

    @Test
    fun bindFailureStopsTheStartedServiceAndFailsTheSession() {
        grantRecordAudio()

        val activity = Robolectric.buildActivity(UnbindableAssistActivity::class.java).create().get()

        assertEquals(SessionState.Failed("Unable to bind voice session"), activity.uiState.value)
        assertEquals(VOICE_SERVICE, stoppedServiceClassName())
    }

    @Test
    fun statusTitlesComeFromStringResources() {
        assertEquals("Connecting", application.getString(SessionState.Idle.titleRes()))
        assertEquals("Connecting", application.getString(SessionState.FetchingToken.titleRes()))
        assertEquals("Connecting", application.getString(SessionState.Connecting.titleRes()))
        assertEquals("Live", application.getString(SessionState.Live.titleRes()))
        assertEquals("Reconnecting", application.getString(SessionState.Reconnecting.titleRes()))
        assertEquals("Ended", application.getString(SessionState.Ended.titleRes()))
        assertEquals("Failed", application.getString(SessionState.Failed("boom").titleRes()))
    }

    @Test
    fun statusDetailsComeFromStringResourcesExceptTheFailureReason() {
        assertEquals("Starting voice session", application.getString(detailRes(SessionState.Idle)))
        assertEquals("Starting voice session", application.getString(detailRes(SessionState.FetchingToken)))
        assertEquals("Starting voice session", application.getString(detailRes(SessionState.Connecting)))
        assertEquals("Listening", application.getString(detailRes(SessionState.Live)))
        assertEquals("Restoring connection", application.getString(detailRes(SessionState.Reconnecting)))
        assertEquals("Session ended", application.getString(detailRes(SessionState.Ended)))
        // The failure reason is dynamic, so it has no resource of its own.
        assertNull(SessionState.Failed("boom").detailRes())
    }

    @Test
    fun talkScreenControlLabelsComeFromStringResources() {
        assertEquals("Settings", application.getString(R.string.talk_settings))
        assertEquals("Token endpoint", application.getString(R.string.talk_token_endpoint))
        assertEquals("Save", application.getString(R.string.talk_save))
        assertEquals("Mute", application.getString(R.string.talk_mute))
        assertEquals("Unmute", application.getString(R.string.talk_unmute))
        assertEquals("End", application.getString(R.string.talk_end))
    }

    private fun detailRes(state: SessionState): Int =
        requireNotNull(state.detailRes()) { "$state has no static detail string" }

    private fun grantRecordAudio() {
        Shadows.shadowOf(application).grantPermissions(Manifest.permission.RECORD_AUDIO)
    }

    /** Delivers a denied location decision so the phone-permission request that follows it can run. */
    private fun answerLocationRequest(activity: AssistActivity) {
        val request = requireNotNull(Shadows.shadowOf(activity).lastRequestedPermission)
        activity.onRequestPermissionsResult(
            request.requestCode,
            request.requestedPermissions,
            IntArray(request.requestedPermissions.size) { PackageManager.PERMISSION_DENIED },
        )
    }

    private fun grantPhonePermissions() {
        Shadows.shadowOf(application).grantPermissions(
            Manifest.permission.SEND_SMS,
            Manifest.permission.READ_CONTACTS,
        )
    }

    private fun startedServiceClassName(): String? =
        Shadows.shadowOf(application).peekNextStartedService()?.component?.className

    private fun startedServiceClassNames(): List<String> = buildList {
        val shadow = Shadows.shadowOf(application)
        while (true) {
            val intent = shadow.getNextStartedService() ?: break
            intent.component?.className?.let(::add)
        }
    }

    private fun drainStartedActivities(activity: AssistActivity): List<Intent> = buildList {
        val shadow = Shadows.shadowOf(activity)
        while (true) {
            val intent = shadow.nextStartedActivity ?: break
            add(intent)
        }
    }

    private fun stoppedServiceClassName(): String? =
        Shadows.shadowOf(application).nextStoppedService?.component?.className

    class UnstartableAssistActivity : AssistActivity() {
        override fun startVoiceService(intent: Intent) {
            throw SecurityException("start not allowed")
        }
    }

    class UnbindableAssistActivity : AssistActivity() {
        override fun bindVoiceService(intent: Intent): Boolean = false
    }

    private companion object {
        const val VOICE_SERVICE = "gg.savecraft.mentat.session.VoiceSessionService"
        const val PHONE_SERVICE = "gg.savecraft.mentat.session.PhoneCommandService"
    }
}

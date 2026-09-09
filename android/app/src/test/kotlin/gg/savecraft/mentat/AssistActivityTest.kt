package gg.savecraft.mentat

import android.Manifest
import android.app.Application
import android.content.ComponentName
import android.content.Intent
import android.content.pm.PackageManager
import androidx.test.core.app.ApplicationProvider
import gg.savecraft.mentat.core.SessionState
import gg.savecraft.mentat.ui.detailRes
import gg.savecraft.mentat.ui.titleRes
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

@Config(sdk = [35])
@RunWith(RobolectricTestRunner::class)
class AssistActivityTest {
    private lateinit var application: Application

    @Before
    fun resetPermission() {
        application = ApplicationProvider.getApplicationContext()
        Shadows.shadowOf(application).denyPermissions(Manifest.permission.RECORD_AUDIO)
    }

    @Test
    fun deniedRecordAudioPermissionDoesNotStartOrBindVoiceService() {
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

        assertNull(Shadows.shadowOf(application).peekNextStartedService())
        assertTrue(Shadows.shadowOf(application).boundServiceConnections.isEmpty())
        assertEquals(SessionState.Failed("Permission denied"), activity.uiState.value)
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
        assertEquals(VOICE_SERVICE, startedServiceClassName())
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
        val activity = Robolectric.buildActivity(AssistActivity::class.java).create().get()
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
    }

    @Test
    fun grantedRecordAudioPermissionStartsVoiceService() {
        grantRecordAudio()

        Robolectric.buildActivity(AssistActivity::class.java).create()

        assertEquals(VOICE_SERVICE, startedServiceClassName())
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

        controller.destroy()

        assertEquals(VOICE_SERVICE, stoppedServiceClassName())
    }

    @Test
    fun destroyingTheActivityStopsASessionThatFailedAfterStarting() {
        grantRecordAudio()
        val controller = Robolectric.buildActivity(AssistActivity::class.java).create()
        val activity = controller.get()
        assertEquals(VOICE_SERVICE, startedServiceClassName())
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

    private fun startedServiceClassName(): String? =
        Shadows.shadowOf(application).peekNextStartedService()?.component?.className

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
    }
}

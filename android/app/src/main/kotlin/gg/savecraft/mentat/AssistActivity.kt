package gg.savecraft.mentat

import android.Manifest
import android.app.ForegroundServiceStartNotAllowedException
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.IBinder
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import gg.savecraft.mentat.core.SessionEvent
import gg.savecraft.mentat.core.SessionState
import gg.savecraft.mentat.core.SessionStateMachine
import gg.savecraft.mentat.core.TranscriptSegment
import gg.savecraft.mentat.session.AppSettings
import gg.savecraft.mentat.session.VoiceSessionService
import gg.savecraft.mentat.ui.TalkScreen
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

open class AssistActivity : ComponentActivity() {
    private val activityScope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val mutableUiState = MutableStateFlow<SessionState>(SessionState.Idle)
    private val uiTranscript = MutableStateFlow<List<TranscriptSegment>>(emptyList())
    private val uiMicEnabled = MutableStateFlow(false)
    private lateinit var settings: AppSettings
    private var service: VoiceSessionService? = null
    private var bound = false
    private var sessionStarted = false
    private var endRequested = false
    private var stateJob: Job? = null
    private var transcriptJob: Job? = null
    private var micJob: Job? = null
    private var activityStarted = false

    internal val uiState: StateFlow<SessionState> = mutableUiState.asStateFlow()

    private val permissionRequest = registerForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted ->
        if (granted) {
            startAndBindVoiceService()
            requestLocationPermissions()
        } else {
            mutableUiState.value = SessionStateMachine().transition(SessionEvent.PermissionDenied)
        }
    }

    private val locationPermissionRequest = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions(),
    ) { }

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName, binder: IBinder) {
            service = (binder as VoiceSessionService.LocalBinder).service()
            service!!.setAssistVisible(activityStarted)
            stateJob = activityScope.launch {
                service!!.state.collect { mutableUiState.value = it }
            }
            transcriptJob = activityScope.launch {
                service!!.transcript.collect { uiTranscript.value = it }
            }
            micJob = activityScope.launch {
                service!!.micEnabled.collect { uiMicEnabled.value = it }
            }
        }

        override fun onServiceDisconnected(name: ComponentName) {
            stateJob?.cancel()
            transcriptJob?.cancel()
            micJob?.cancel()
            service = null
            bound = false
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        settings = AppSettings(this)
        logAssistIntent()
        setContent {
            val state by uiState.collectAsState()
            val transcript by uiTranscript.collectAsState()
            val micEnabled by uiMicEnabled.collectAsState()
            MaterialTheme {
                TalkScreen(
                    state = state,
                    transcript = transcript,
                    muted = !micEnabled,
                    settings = settings,
                    onMuteChanged = { muted -> service?.mute(muted) },
                    onEnd = {
                        endVoiceSession()
                        finish()
                    },
                )
            }
        }
        beginVoiceSession()
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        logAssistIntent()
        beginVoiceSession()
    }

    override fun onStart() {
        super.onStart()
        activityStarted = true
        service?.setAssistVisible(true)
    }

    override fun onStop() {
        activityStarted = false
        service?.setAssistVisible(false)
        super.onStop()
    }

    override fun onDestroy() {
        // A failed session can still have a running service behind it — a token or connect
        // failure does not stop one — so only an already-ended session skips the stop.
        if (uiState.value != SessionState.Ended) {
            endVoiceSession()
        }
        if (bound) {
            unbindService(connection)
            bound = false
        }
        activityScope.cancel()
        super.onDestroy()
    }

    /**
     * Stops the voice session for good. The service is started, not merely bound, so it
     * outlives this activity unless it is told to stop: end through the binder when the
     * connection is up, and straight through the service intent when it is not, so that
     * ending never depends on how far the binding has progressed.
     */
    internal fun endVoiceSession() {
        if (endRequested || !sessionStarted) {
            return
        }
        endRequested = true
        val session = service
        if (session != null) {
            session.end()
        } else {
            stopService(voiceServiceIntent())
        }
    }

    protected open fun startVoiceService(intent: Intent) {
        startForegroundService(intent)
    }

    protected open fun bindVoiceService(intent: Intent): Boolean =
        bindService(intent, connection, Context.BIND_AUTO_CREATE)

    /**
     * Who may reach this activity at all is settled before it runs: it is exported for
     * assist dispatch and guarded in the manifest by a signature permission, so only the
     * system and our own app can launch it. Everything from here on is a trusted start.
     */
    private fun beginVoiceSession() {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            startAndBindVoiceService()
            requestLocationPermissions()
        } else {
            permissionRequest.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    private fun requestLocationPermissions() {
        locationPermissionRequest.launch(
            arrayOf(
                Manifest.permission.ACCESS_FINE_LOCATION,
                Manifest.permission.ACCESS_COARSE_LOCATION,
            ),
        )
    }

    private fun startAndBindVoiceService() {
        if (bound) {
            return
        }
        val intent = voiceServiceIntent()
        try {
            startVoiceService(intent)
        } catch (exception: ForegroundServiceStartNotAllowedException) {
            failSession(exception.message ?: SERVICE_START_FAILED)
            return
        } catch (exception: SecurityException) {
            failSession(exception.message ?: SERVICE_START_FAILED)
            return
        }
        sessionStarted = true
        bound = bindVoiceService(intent)
        if (!bound) {
            stopService(intent)
            endRequested = true
            failSession(SERVICE_BIND_FAILED)
        }
    }

    internal fun failSession(reason: String) {
        mutableUiState.value = SessionStateMachine().transition(SessionEvent.ServiceStartFailed(reason))
    }

    private fun voiceServiceIntent() = Intent(this, VoiceSessionService::class.java)

    private fun logAssistIntent() {
        Log.i("MentatAssist", "MENTAT_ASSIST_RECEIVED action=" + intent.action)
    }

    private companion object {
        const val SERVICE_START_FAILED = "Unable to start voice service"
        const val SERVICE_BIND_FAILED = "Unable to bind voice session"
    }
}

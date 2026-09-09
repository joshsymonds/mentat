package gg.savecraft.mentat.session

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.Binder
import android.os.IBinder
import android.util.Log
import gg.savecraft.mentat.R
import gg.savecraft.mentat.core.HttpTokenEndpoint
import gg.savecraft.mentat.core.SessionEvent
import gg.savecraft.mentat.core.SessionState
import gg.savecraft.mentat.core.SessionStateMachine
import gg.savecraft.mentat.core.TokenEndpoint
import gg.savecraft.mentat.core.Transcript
import gg.savecraft.mentat.core.TranscriptSegment
import gg.savecraft.mentat.phone.PhoneBridge
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

open class VoiceSessionService : Service() {
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val binder = LocalBinder()
    private lateinit var controller: VoiceSessionController
    private lateinit var bridge: PhoneBridge
    private var started = false

    val state: StateFlow<SessionState>
        get() = controller.state
    val transcript: StateFlow<List<TranscriptSegment>>
        get() = controller.transcript
    val micEnabled: StateFlow<Boolean>
        get() = controller.micEnabled

    override fun onCreate() {
        super.onCreate()
        bridge = phoneBridge()
        controller = VoiceSessionController(
            tokenEndpoint = tokenEndpoint(),
            liveKitSession = liveKitSession(),
            phoneBridge = bridge,
            stopService = ::stopVoiceService,
            scope = serviceScope,
        )
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (!started) {
            started = true
            try {
                startForegroundNotification()
            } catch (exception: Exception) {
                controller.serviceStartFailed(
                    exception.message ?: "Unable to start voice service",
                )
                stopVoiceService()
                return START_NOT_STICKY
            }
            serviceScope.launch {
                controller.start()
            }
        }
        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onDestroy() {
        try {
            controller.close()
        } finally {
            try {
                serviceScope.cancel()
            } finally {
                super.onDestroy()
            }
        }
    }

    fun mute(muted: Boolean) {
        serviceScope.launch {
            controller.mute(muted)
        }
    }

    fun end() {
        serviceScope.launch {
            try {
                controller.end()
            } finally {
                stopForeground(STOP_FOREGROUND_REMOVE)
            }
        }
    }

    protected open fun tokenEndpoint(): TokenEndpoint = HttpTokenEndpoint(AppSettings(this).tokenEndpointUrl)

    protected open fun liveKitSession(): LiveKitSession = AndroidLiveKitSession(this)

    protected open fun phoneBridge(): PhoneBridge = PhoneBridge(this)

    fun setAssistVisible(visible: Boolean) {
        bridge.assistVisible = visible
    }

    protected open fun stopVoiceService() {
        stopSelf()
    }

    protected open fun startForegroundNotification() {
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(
            NotificationChannel(
                NOTIFICATION_CHANNEL_ID,
                getString(R.string.voice_notification_channel),
                NotificationManager.IMPORTANCE_LOW,
            ),
        )
        val notification: Notification = Notification.Builder(this, NOTIFICATION_CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentTitle(getString(R.string.voice_notification_title))
            .setContentText(getString(R.string.voice_notification_text))
            .setOngoing(true)
            .build()
        startForeground(NOTIFICATION_ID, notification)
    }

    inner class LocalBinder : Binder() {
        fun service(): VoiceSessionService = this@VoiceSessionService
    }

    private companion object {
        const val NOTIFICATION_CHANNEL_ID = "voice-session"
        const val NOTIFICATION_ID = 1
    }
}

internal class VoiceSessionController(
    private val tokenEndpoint: TokenEndpoint,
    private val liveKitSession: LiveKitSession,
    private val phoneBridge: PhoneBridge,
    private val stopService: () -> Unit,
    private val scope: CoroutineScope,
) {
    private val machine = SessionStateMachine()
    private val transcriptStore = Transcript()
    private val mutableState = MutableStateFlow(machine.state)
    private val mutableTranscript = MutableStateFlow(transcriptStore.segments)
    private val mutableMicEnabled = MutableStateFlow(false)
    private var ending = false
    private var eventJob: Job? = null
    private var transcriptJob: Job? = null

    val state: StateFlow<SessionState> = mutableState.asStateFlow()
    val transcript: StateFlow<List<TranscriptSegment>> = mutableTranscript.asStateFlow()
    val micEnabled: StateFlow<Boolean> = mutableMicEnabled.asStateFlow()

    suspend fun start() {
        eventJob = scope.launch(start = CoroutineStart.UNDISPATCHED) {
            liveKitSession.events.collect(::onLiveKitEvent)
        }
        transcriptJob = scope.launch(start = CoroutineStart.UNDISPATCHED) {
            liveKitSession.transcripts.collect { segment ->
                transcriptStore.update(segment)
                mutableTranscript.value = transcriptStore.segments
            }
        }
        transition(SessionEvent.AssistInvoked)
        val grant = try {
            withContext(Dispatchers.IO) { tokenEndpoint.fetch() }
        } catch (exception: Exception) {
            transition(SessionEvent.TokenFailed(exception.message ?: "Unable to fetch voice token"))
            return
        }
        transition(SessionEvent.TokenReceived(grant))
        try {
            liveKitSession.connect(grant.url, grant.token)
            liveKitSession.registerPhoneBridge(phoneBridge)
            liveKitSession.setMicEnabled(true)
            mutableMicEnabled.value = true
        } catch (exception: Exception) {
            transition(SessionEvent.ConnectFailed(exception.message ?: "Unable to connect to voice session"))
        }
    }

    fun serviceStartFailed(reason: String) {
        transition(SessionEvent.ServiceStartFailed(reason))
    }

    suspend fun mute(muted: Boolean) {
        val enabled = !muted
        try {
            liveKitSession.setMicEnabled(enabled)
            mutableMicEnabled.value = enabled
        } catch (exception: Exception) {
            Log.w("MentatAssist", "Unable to change microphone state", exception)
        }
    }

    suspend fun end() {
        ending = true
        transition(SessionEvent.EndRequested)
        try {
            liveKitSession.disconnect()
        } finally {
            try {
                close()
            } finally {
                stopService()
            }
        }
    }

    fun close() {
        try {
            eventJob?.cancel()
        } finally {
            try {
                transcriptJob?.cancel()
            } finally {
                liveKitSession.close()
            }
        }
    }

    private fun onLiveKitEvent(event: LiveKitEvent) {
        if (ending) {
            return
        }
        if (event is LiveKitEvent.Disconnected && state.value == SessionState.Live) {
            transition(SessionEvent.ConnectionLost)
        }
        LiveKitSession.eventFor(event)?.let(::transition)
    }

    private fun transition(event: SessionEvent) {
        mutableState.value = machine.transition(event)
    }
}

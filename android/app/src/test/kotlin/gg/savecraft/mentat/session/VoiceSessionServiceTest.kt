package gg.savecraft.mentat.session

import android.content.Intent
import android.os.Looper
import gg.savecraft.mentat.core.CallContext
import gg.savecraft.mentat.core.SessionState
import gg.savecraft.mentat.core.TokenEndpoint
import gg.savecraft.mentat.core.TokenFetchException
import gg.savecraft.mentat.core.TokenGrant
import gg.savecraft.mentat.core.TranscriptSegment
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import java.util.Collections
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
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
class VoiceSessionServiceTest {
    private val scope = CoroutineScope(Dispatchers.Unconfined)

    @Test
    fun happyPathFetchesTokenConnectsAndPublishesMicrophone() = runBlocking {
        val liveKit = FakeLiveKitSession()
        val service = controller(liveKit)

        service.start()
        liveKit.events.emit(LiveKitEvent.Connected)

        assertEquals(SessionState.Live, service.state.value)
        assertEquals("wss://voice.example.com" to "token", liveKit.connection)
        assertTrue(liveKit.callOrder.indexOf("chime") < liveKit.callOrder.indexOf("buffer-start"))
        assertTrue(liveKit.callOrder.indexOf("buffer-start") < liveKit.callOrder.indexOf("connect"))
        assertTrue(liveKit.callOrder.indexOf("connect") < liveKit.callOrder.indexOf("mic"))
        assertTrue(liveKit.callOrder.indexOf("mic") < liveKit.callOrder.indexOf("buffer-end"))
        assertTrue(liveKit.microphoneEnabled.value)
        assertTrue(service.micEnabled.value)
    }

    @Test
    fun captureWaitsForChimeWhileTokenFetchStartsDuringPlayback() = runBlocking {
        val callOrder = Collections.synchronizedList(mutableListOf<String>())
        val chimeCompletion = CompletableDeferred<Unit>()
        val fetchDuringPlayback = CompletableDeferred<Boolean>()
        val liveKit = FakeLiveKitSession(
            callOrder = callOrder,
            chimeCompletion = chimeCompletion,
        )
        val endpoint = RecordingTokenEndpoint(callOrder) {
            fetchDuringPlayback.complete(liveKit.chimePlaying)
        }
        val service = controller(liveKit, endpoint, tokenDispatcher = ImmediateDispatcher())
        val start = async { service.start() }

        assertTrue(withTimeout(2_000) { fetchDuringPlayback.await() })
        assertEquals(listOf("chime", "fetch"), callOrder.toList())

        chimeCompletion.complete(Unit)
        start.await()

        assertTrue(callOrder.indexOf("chime-complete") < callOrder.indexOf("buffer-start"))
        assertTrue(callOrder.indexOf("fetch") < callOrder.indexOf("buffer-start"))
        assertEquals("wss://voice.example.com" to "token", liveKit.connection)
    }

    @Test
    fun chimeFailureStartsCaptureImmediatelyAndStillConnects() = runBlocking {
        val callOrder = Collections.synchronizedList(mutableListOf<String>())
        val fetchStarted = kotlinx.coroutines.CompletableDeferred<Unit>()
        val liveKit = FakeLiveKitSession(
            callOrder = callOrder,
            chimeFailure = IllegalStateException("chime unavailable"),
        )
        val endpoint = RecordingTokenEndpoint(callOrder) { fetchStarted.complete(Unit) }
        val service = controller(liveKit, endpoint)

        service.start()
        withTimeout(2_000) { fetchStarted.await() }

        assertTrue(callOrder.indexOf("chime") < callOrder.indexOf("buffer-start"))
        assertTrue(callOrder.indexOf("buffer-start") < callOrder.indexOf("connect"))
        assertEquals("wss://voice.example.com" to "token", liveKit.connection)
        assertFalse(service.state.value is SessionState.Failed)
    }

    @Test
    fun questionSpokenDuringPreconnectCaptureReachesTheConnectedSession() = runBlocking {
        val captureStarted = CompletableDeferred<Unit>()
        val allowConnect = CompletableDeferred<Unit>()
        val liveKit = FakeLiveKitSession(
            captureStarted = captureStarted,
            connectGate = allowConnect,
        )
        val service = controller(liveKit)
        val start = async { service.start() }

        withTimeout(2_000) { captureStarted.await() }
        liveKit.speakQuestion("Hey Mentat, what's next?")
        allowConnect.complete(Unit)
        start.await()

        assertEquals("wss://voice.example.com" to "token", liveKit.connection)
        assertEquals("Hey Mentat, what's next?", liveKit.questionDeliveredAtConnect)
    }

    @Test
    fun tokenFailureFailsTheSession() = runBlocking {
        val service = controller(FakeLiveKitSession(), FailingTokenEndpoint)

        service.start()

        assertEquals(SessionState.Failed("Token request failed"), service.state.value)
    }

    @Test
    fun connectionFailureFailsTheSession() = runBlocking {
        val liveKit = FakeLiveKitSession(connectFailure = IllegalStateException("connection refused"))
        val service = controller(liveKit)

        service.start()

        assertEquals(SessionState.Failed("connection refused"), service.state.value)
        assertTrue(liveKit.callOrder.indexOf("chime") < liveKit.callOrder.indexOf("buffer-start"))
        assertTrue(liveKit.callOrder.indexOf("buffer-start") < liveKit.callOrder.indexOf("connect"))
        assertTrue(liveKit.callOrder.indexOf("connect") < liveKit.callOrder.indexOf("buffer-end"))
    }

    @Test
    fun microphonePublishFailureAfterConnectionFailsTheSession() = runBlocking {
        val liveKit = FakeLiveKitSession(
            connectEvent = LiveKitEvent.Connected,
            setMicFailure = IllegalStateException("microphone rejected"),
        )
        val service = controller(liveKit)

        service.start()

        assertEquals(SessionState.Failed("microphone rejected"), service.state.value)
    }

    @Test
    fun startForegroundFailureFailsTheSessionAndStopsTheService() {
        FailingForegroundService.stopped = false
        val service = Robolectric.buildService(FailingForegroundService::class.java).create().get()

        service.onStartCommand(Intent(), 0, 1)

        assertEquals(SessionState.Failed("notification rejected"), service.state.value)
        assertTrue(FailingForegroundService.stopped)
    }

    @Test
    fun endDisconnectsClosesAndStopsTheService() = runBlocking {
        val liveKit = FakeLiveKitSession()
        var stopped = false
        val service = controller(liveKit, stopService = { stopped = true })

        service.start()
        liveKit.events.emit(LiveKitEvent.Connected)
        service.end()

        assertTrue(liveKit.closed)
        assertTrue(stopped)
        assertEquals(SessionState.Ended, service.state.value)
    }

    @Test
    fun destroyingTheServiceClosesTheLiveKitSession() {
        FakeLiveKitVoiceSessionService.liveKit = FakeLiveKitSession()
        val service = Robolectric.buildService(FakeLiveKitVoiceSessionService::class.java).create().get()

        service.onDestroy()

        assertTrue(FakeLiveKitVoiceSessionService.liveKit.closed)
    }

    @Test
    fun endStopsTheServiceEvenWhenDisconnectFails() = runBlocking {
        val liveKit = FakeLiveKitSession()
        liveKit.disconnectFailure = IllegalStateException("disconnect failed")
        var stopped = false
        val service = controller(liveKit, stopService = { stopped = true })

        service.start()
        liveKit.events.emit(LiveKitEvent.Connected)
        val thrown = runCatching { service.end() }.exceptionOrNull()

        assertEquals("disconnect failed", thrown?.message)
        assertTrue(liveKit.closed)
        assertTrue(stopped)
    }

    @Test
    fun endStopsTheServiceEvenWhenClosingTheLiveKitSessionFails() = runBlocking {
        val liveKit = FakeLiveKitSession()
        liveKit.closeFailure = IllegalStateException("close failed")
        var stopped = false
        val service = controller(liveKit, stopService = { stopped = true })

        service.start()
        liveKit.events.emit(LiveKitEvent.Connected)
        val thrown = runCatching { service.end() }.exceptionOrNull()

        assertEquals("close failed", thrown?.message)
        assertTrue(liveKit.disconnected)
        assertTrue(stopped)
    }

    @Test
    fun endStopsTheForegroundEvenWhenTheSessionFailsToClose() {
        val liveKit = FakeLiveKitSession()
        liveKit.closeFailure = IllegalStateException("close failed")
        FakeLiveKitVoiceSessionService.liveKit = liveKit
        val service = Robolectric.buildService(FakeLiveKitVoiceSessionService::class.java).create().get()

        service.end()

        assertTrue(liveKit.disconnected)
        assertTrue(Shadows.shadowOf(service).isForegroundStopped)
    }

    @Test
    fun destroyingTheServiceCancelsItsScopeEvenWhenClosingFails() {
        val liveKit = FakeLiveKitSession()
        liveKit.closeFailure = IllegalStateException("close failed")
        FakeLiveKitVoiceSessionService.liveKit = liveKit
        val service = Robolectric.buildService(FakeLiveKitVoiceSessionService::class.java).create().get()

        val thrown = runCatching { service.onDestroy() }.exceptionOrNull()

        assertEquals("close failed", thrown?.message)
        assertTrue(liveKit.closed)
        // A cancelled scope no longer dispatches work, so the microphone request is dropped.
        service.mute(false)
        assertFalse(liveKit.microphoneEnabled.value)
    }

    @Test
    fun gracefulRemoteDisconnectEndsTheSessionWithoutReconnecting() = runBlocking {
        val liveKit = FakeLiveKitSession()
        var stopped = false
        val states = mutableListOf<SessionState>()
        val service = controller(liveKit, stopService = { stopped = true })
        val stateJob = launch(start = CoroutineStart.UNDISPATCHED) {
            service.state.collect(states::add)
        }

        service.start()
        liveKit.events.emit(LiveKitEvent.Connected)
        liveKit.events.emit(LiveKitEvent.Disconnected("ROOM_DELETED", graceful = true))
        stateJob.cancel()

        assertEquals(SessionState.Ended, service.state.value)
        assertFalse(states.contains(SessionState.Reconnecting))
        assertTrue(stopped)
        assertFalse(liveKit.disconnected)
        assertTrue(liveKit.closed)
    }

    @Test
    fun terminalDisconnectFromLiveFailsTheSession() = runBlocking {
        val liveKit = FakeLiveKitSession()
        val service = controller(liveKit)

        service.start()
        liveKit.events.emit(LiveKitEvent.Connected)
        liveKit.events.emit(LiveKitEvent.Disconnected("server closed", graceful = false))

        assertEquals(SessionState.Failed("server closed"), service.state.value)
    }

    @Test
    fun muteAndTranscriptionUpdateTheExposedFlows() = runBlocking {
        val liveKit = FakeLiveKitSession()
        val service = controller(liveKit)

        service.start()
        service.mute(true)
        liveKit.transcripts.emit(
            TranscriptSegment("one", "agent", "Hel", final = false),
        )
        liveKit.transcripts.emit(
            TranscriptSegment("one", "agent", "Hello", final = true),
        )

        assertFalse(liveKit.microphoneEnabled.value)
        assertFalse(service.micEnabled.value)
        assertEquals(
            listOf(TranscriptSegment("one", "agent", "Hello", final = true)),
            service.transcript.value,
        )
    }

    @Test
    fun muteFailureKeepsTheAuthoritativeMicrophoneState() = runBlocking {
        val liveKit = FakeLiveKitSession()
        val service = controller(liveKit)
        service.start()
        liveKit.setMicFailure = IllegalStateException("microphone rejected")

        service.mute(true)

        assertTrue(service.micEnabled.value)
        assertTrue(liveKit.microphoneEnabled.value)
    }

    @Test
    fun callContextRidesWithTheTokenRequest() = runBlocking {
        val context = CallContext(timeZone = "America/Los_Angeles", location = null, driving = true)
        val endpoint = RecordingTokenEndpoint()

        controller(FakeLiveKitSession(), endpoint, callContext = { context }).start()

        assertEquals(listOf<CallContext?>(context), endpoint.contexts)
    }

    @Test
    fun unreadableCallContextStillFetchesATokenWithoutOne() = runBlocking {
        val endpoint = RecordingTokenEndpoint()
        val liveKit = FakeLiveKitSession()

        controller(liveKit, endpoint, callContext = { throw SecurityException("no location") }).start()

        assertEquals(listOf<CallContext?>(null), endpoint.contexts)
        assertEquals("wss://voice.example.com" to "token", liveKit.connection)
    }

    private fun controller(
        liveKitSession: FakeLiveKitSession,
        tokenEndpoint: TokenEndpoint = FakeTokenEndpoint,
        stopService: () -> Unit = {},
        callContext: () -> CallContext? = { null },
        tokenDispatcher: CoroutineDispatcher = Dispatchers.IO,
    ) = VoiceSessionController(
        tokenEndpoint = tokenEndpoint,
        liveKitSession = liveKitSession,
        stopService = stopService,
        scope = scope,
        callContext = callContext,
        tokenDispatcher = tokenDispatcher,
    )

    private class ImmediateDispatcher : CoroutineDispatcher() {
        override fun isDispatchNeeded(context: kotlin.coroutines.CoroutineContext): Boolean = false

        override fun dispatch(context: kotlin.coroutines.CoroutineContext, block: Runnable) {
            block.run()
        }
    }

    private class RecordingTokenEndpoint(
        private val callOrder: MutableList<String> = mutableListOf(),
        private val onFetch: () -> Unit = {},
    ) : TokenEndpoint {
        val contexts = mutableListOf<CallContext?>()

        override fun fetch(context: CallContext?): TokenGrant {
            callOrder += "fetch"
            onFetch()
            contexts += context
            return FakeTokenEndpoint.fetch(context)
        }
    }

    private object FakeTokenEndpoint : TokenEndpoint {
        override fun fetch(context: CallContext?) = TokenGrant(
            token = "token",
            room = "room",
            url = "wss://voice.example.com",
            expiresAt = "2026-08-19T16:00:00Z",
        )
    }

    private object FailingTokenEndpoint : TokenEndpoint {
        override fun fetch(context: CallContext?): TokenGrant = throw TokenFetchException("Token request failed")
    }

    class FakeLiveKitSession(
        private val connectEvent: LiveKitEvent? = null,
        private val connectFailure: Exception? = null,
        var setMicFailure: Exception? = null,
        var disconnectFailure: Exception? = null,
        var closeFailure: Exception? = null,
        val callOrder: MutableList<String> = mutableListOf(),
        private val chimeFailure: Exception? = null,
        private val chimeCompletion: CompletableDeferred<Unit>? = null,
        private val captureStarted: CompletableDeferred<Unit>? = null,
        private val connectGate: CompletableDeferred<Unit>? = null,
    ) : LiveKitSession {
        override val events = MutableSharedFlow<LiveKitEvent>()
        override val transcripts = MutableSharedFlow<TranscriptSegment>()
        val microphoneEnabled = MutableStateFlow(false)
        var connection: Pair<String, String>? = null
        var questionDeliveredAtConnect: String? = null
        var chimePlaying = false
            private set
        private var capturingPreconnectAudio = false
        private val capturedAudio = mutableListOf<String>()
        var disconnected = false
        var closed = false

        override suspend fun withPreconnectAudio(operation: suspend () -> Unit) {
            callOrder += "buffer-start"
            capturingPreconnectAudio = true
            captureStarted?.complete(Unit)
            try {
                operation()
            } finally {
                capturingPreconnectAudio = false
                callOrder += "buffer-end"
            }
        }

        override suspend fun playListeningChime() {
            callOrder += "chime"
            chimeFailure?.let { throw it }
            chimePlaying = true
            try {
                chimeCompletion?.await()
                callOrder += "chime-complete"
            } finally {
                chimePlaying = false
            }
        }

        fun speakQuestion(text: String) {
            check(capturingPreconnectAudio) { "Question was spoken outside preconnect capture" }
            capturedAudio += text
        }

        override suspend fun connect(url: String, token: String) {
            callOrder += "connect"
            connectFailure?.let { throw it }
            connectGate?.await()
            connection = url to token
            questionDeliveredAtConnect = capturedAudio.joinToString("")
            connectEvent?.let { events.emit(it) }
        }


        override suspend fun setMicEnabled(enabled: Boolean) {
            callOrder += "mic"
            setMicFailure?.let { throw it }
            microphoneEnabled.value = enabled
        }

        override suspend fun disconnect() {
            disconnected = true
            disconnectFailure?.let { throw it }
        }

        override fun close() {
            closed = true
            closeFailure?.let { throw it }
        }
    }

    class FailingForegroundService : VoiceSessionService() {
        override fun liveKitSession(): LiveKitSession = FakeLiveKitSession()

        override fun startForegroundNotification() {
            throw IllegalStateException("notification rejected")
        }

        override fun stopVoiceService() {
            stopped = true
        }

        companion object {
            var stopped = false
        }
    }

    class FakeLiveKitVoiceSessionService : VoiceSessionService() {
        override fun tokenEndpoint(): TokenEndpoint = FakeTokenEndpoint

        override fun liveKitSession(): LiveKitSession = liveKit

        override fun startForegroundNotification() {}

        companion object {
            lateinit var liveKit: FakeLiveKitSession
        }
    }
}

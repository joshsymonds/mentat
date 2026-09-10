package gg.savecraft.mentat.session

import android.content.Context
import gg.savecraft.mentat.core.SessionEvent
import gg.savecraft.mentat.core.TranscriptSegment
import gg.savecraft.mentat.phone.PhoneBridge
import io.livekit.android.LiveKit
import io.livekit.android.events.DisconnectReason
import io.livekit.android.events.RoomEvent
import io.livekit.android.events.collect
import io.livekit.android.room.Room
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.launch

sealed interface LiveKitEvent {
    data object Connected : LiveKitEvent
    data object Reconnecting : LiveKitEvent
    data object Reconnected : LiveKitEvent
    data class Disconnected(val reason: String, val graceful: Boolean) : LiveKitEvent
}

interface LiveKitSession {
    val events: Flow<LiveKitEvent>
    val transcripts: Flow<TranscriptSegment>

    suspend fun connect(url: String, token: String)

    fun registerPhoneBridge(bridge: PhoneBridge)

    suspend fun setMicEnabled(enabled: Boolean)

    suspend fun disconnect()

    fun close()

    companion object {
        fun gracefulDisconnect(reason: DisconnectReason): Boolean =
            reason == DisconnectReason.ROOM_DELETED ||
                reason == DisconnectReason.PARTICIPANT_REMOVED

        fun eventFor(event: LiveKitEvent): SessionEvent? = when (event) {
            LiveKitEvent.Connected -> SessionEvent.RoomConnected
            LiveKitEvent.Reconnecting -> SessionEvent.ConnectionLost
            LiveKitEvent.Reconnected -> SessionEvent.Reconnected
            is LiveKitEvent.Disconnected -> if (event.graceful) {
                SessionEvent.EndRequested
            } else {
                SessionEvent.ReconnectFailed(event.reason)
            }
        }

        fun transcriptSegmentFor(
            streamId: String,
            participantIdentity: String,
            text: String,
            attributes: Map<String, String>,
        ) = TranscriptSegment(
            id = attributes[SEGMENT_ID_ATTRIBUTE] ?: streamId,
            participantIdentity = participantIdentity,
            text = text,
            final = attributes[TRANSCRIPTION_FINAL_ATTRIBUTE] == "true",
        )
    }
}

class AndroidLiveKitSession(context: Context) : LiveKitSession {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val room: Room = LiveKit.create(context.applicationContext)
    private val mutableEvents = MutableSharedFlow<LiveKitEvent>(extraBufferCapacity = 8)
    private val mutableTranscripts = MutableSharedFlow<TranscriptSegment>(extraBufferCapacity = 8)
    private var disconnected = false
    private var closed = false

    override val events: Flow<LiveKitEvent> = mutableEvents.asSharedFlow()
    override val transcripts: Flow<TranscriptSegment> = mutableTranscripts.asSharedFlow()

    init {
        scope.launch {
            room.events.collect { event ->
                event.toLiveKitEvent()?.let(mutableEvents::tryEmit)
            }
        }
        room.registerTextStreamHandler(TRANSCRIPTION_TOPIC) { reader, participantIdentity ->
            scope.launch {
                val text = reader.readAll().joinToString("")
                mutableTranscripts.emit(
                    LiveKitSession.transcriptSegmentFor(
                        streamId = reader.info.id,
                        participantIdentity = participantIdentity.value,
                        text = text,
                        attributes = reader.info.attributes,
                    ),
                )
            }
        }
    }

    override fun registerPhoneBridge(bridge: PhoneBridge) {
        bridge.register(room.localParticipant)
    }

    override suspend fun connect(url: String, token: String) {
        room.connect(url, token)
    }

    override suspend fun setMicEnabled(enabled: Boolean) {
        room.localParticipant.setMicrophoneEnabled(enabled)
    }

    override suspend fun disconnect() {
        if (!disconnected) {
            room.disconnect()
            disconnected = true
        }
    }

    override fun close() {
        if (closed) {
            return
        }
        closed = true
        try {
            if (!disconnected) {
                room.disconnect()
                disconnected = true
            }
        } finally {
            try {
                room.release()
            } finally {
                scope.cancel()
            }
        }
    }

    private fun RoomEvent.toLiveKitEvent(): LiveKitEvent? = when (this) {
        is RoomEvent.Connected -> LiveKitEvent.Connected
        is RoomEvent.Reconnecting -> LiveKitEvent.Reconnecting
        is RoomEvent.Reconnected -> LiveKitEvent.Reconnected
        is RoomEvent.Disconnected -> {
            disconnected = true
            LiveKitEvent.Disconnected(
                reason = error?.message ?: reason.name,
                graceful = LiveKitSession.gracefulDisconnect(reason),
            )
        }
        else -> null
    }
}

private const val TRANSCRIPTION_TOPIC = "lk.transcription"
private const val TRANSCRIPTION_FINAL_ATTRIBUTE = "lk.transcription_final"
private const val SEGMENT_ID_ATTRIBUTE = "lk.segment_id"

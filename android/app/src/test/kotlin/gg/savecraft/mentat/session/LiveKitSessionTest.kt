package gg.savecraft.mentat.session

import android.media.AudioAttributes
import android.media.MediaPlayer
import sun.misc.Unsafe
import gg.savecraft.mentat.core.SessionEvent
import gg.savecraft.mentat.core.TranscriptSegment
import io.livekit.android.events.DisconnectReason
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.RuntimeEnvironment
import org.robolectric.Shadows.shadowOf
import org.robolectric.shadows.ShadowMediaPlayer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlinx.coroutines.CoroutineStart
import kotlinx.coroutines.async
import kotlinx.coroutines.runBlocking

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [35])
class LiveKitSessionTest {
    @Test
    fun listeningChimeUsesAssistantAudioUsage() = runBlocking {
        var player: MediaPlayer? = null
        ShadowMediaPlayer.setMediaInfoProvider { ShadowMediaPlayer.MediaInfo(1_000, 0) }
        ShadowMediaPlayer.setCreateListener { created, _ -> player = created }
        val unsafeField = Unsafe::class.java.getDeclaredField("theUnsafe").apply {
            isAccessible = true
        }
        val session = (unsafeField.get(null) as Unsafe)
            .allocateInstance(AndroidLiveKitSession::class.java) as AndroidLiveKitSession
        AndroidLiveKitSession::class.java.getDeclaredField("appContext").apply {
            isAccessible = true
        }.set(session, RuntimeEnvironment.getApplication())
        try {
            val playback = async(start = CoroutineStart.UNDISPATCHED) {
                session.playListeningChime()
            }
            shadowOf(checkNotNull(player)).invokeCompletionListener()
            playback.await()

            assertEquals(
                AudioAttributes.USAGE_ASSISTANT,
                shadowOf(checkNotNull(player)).audioAttributes?.usage,
            )
        } finally {
            player?.release()
            ShadowMediaPlayer.setCreateListener(null)
        }
    }

    @Test
    fun disconnectReasonsClassifyGracefulEnds() {
        assertTrue(LiveKitSession.gracefulDisconnect(DisconnectReason.ROOM_DELETED))
        assertTrue(LiveKitSession.gracefulDisconnect(DisconnectReason.PARTICIPANT_REMOVED))
        assertFalse(LiveKitSession.gracefulDisconnect(DisconnectReason.CLIENT_INITIATED))
        assertFalse(LiveKitSession.gracefulDisconnect(DisconnectReason.CONNECTION_TIMEOUT))
        assertFalse(LiveKitSession.gracefulDisconnect(DisconnectReason.SERVER_SHUTDOWN))
    }

    @Test
    fun roomEventsMapToSessionEvents() {
        assertEquals(SessionEvent.ConnectionLost, LiveKitSession.eventFor(LiveKitEvent.Reconnecting))
        assertEquals(SessionEvent.Reconnected, LiveKitSession.eventFor(LiveKitEvent.Reconnected))
        assertEquals(
            SessionEvent.EndRequested,
            LiveKitSession.eventFor(
                LiveKitEvent.Disconnected("ROOM_DELETED", graceful = true),
            ),
        )
        assertEquals(
            SessionEvent.EndRequested,
            LiveKitSession.eventFor(
                LiveKitEvent.Disconnected("PARTICIPANT_REMOVED", graceful = true),
            ),
        )
        assertEquals(
            SessionEvent.ReconnectFailed("CONNECTION_TIMEOUT"),
            LiveKitSession.eventFor(
                LiveKitEvent.Disconnected("CONNECTION_TIMEOUT", graceful = false),
            ),
        )
    }

    @Test
    fun transcriptionUsesSegmentAttributeWhenPresent() {
        assertEquals(
            TranscriptSegment(
                id = "segment-1",
                participantIdentity = "agent",
                text = "Hello there",
                final = true,
            ),
            LiveKitSession.transcriptSegmentFor(
                streamId = "stream-1",
                participantIdentity = "agent",
                text = "Hello there",
                attributes = mapOf(
                    "lk.segment_id" to "segment-1",
                    "lk.transcription_final" to "true",
                ),
            ),
        )
    }

    @Test
    fun transcriptionFallsBackToStreamIdAndTreatsOtherValuesAsPartial() {
        assertEquals(
            TranscriptSegment(
                id = "stream-1",
                participantIdentity = "caller",
                text = "Hel",
                final = false,
            ),
            LiveKitSession.transcriptSegmentFor(
                streamId = "stream-1",
                participantIdentity = "caller",
                text = "Hel",
                attributes = mapOf("lk.transcription_final" to "TRUE"),
            ),
        )
    }
}

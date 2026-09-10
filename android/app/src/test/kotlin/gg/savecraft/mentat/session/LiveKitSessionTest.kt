package gg.savecraft.mentat.session

import gg.savecraft.mentat.core.SessionEvent
import gg.savecraft.mentat.core.TranscriptSegment
import io.livekit.android.events.DisconnectReason
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class LiveKitSessionTest {
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

package gg.savecraft.mentat.session

import gg.savecraft.mentat.core.SessionEvent
import gg.savecraft.mentat.core.TranscriptSegment
import org.junit.Assert.assertEquals
import org.junit.Test

class LiveKitSessionTest {
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

package gg.savecraft.mentat.core

import android.content.Intent
import android.net.Uri
import android.provider.AlarmClock
import java.time.Instant
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@Config(sdk = [35])
@RunWith(RobolectricTestRunner::class)
class PhoneIntentsTest {
    private val expiry = "2026-09-12T00:00:00Z"

    @Test
    fun parsesBridgeCommandsWithIdAndExpiry() {
        assertEquals(
            PhoneCommand.Navigate("n", "Trader Joe's", "123 Main St", "ChIJabc", 40.1, -73.2, Instant.parse(expiry)),
            PhoneCommand.parse("""{"kind":"navigate","id":"n","name":"Trader Joe's","address":"123 Main St","place_id":"ChIJabc","lat":40.1,"lng":-73.2,"expires_at":"$expiry"}"""),
        )
        assertEquals(
            PhoneCommand.Dial("d", "+15551212", Instant.parse(expiry)),
            PhoneCommand.parse("""{"kind":"dial","id":"d","number":"+15551212","expires_at":"$expiry"}"""),
        )
        assertEquals(
            PhoneCommand.Alarm("a", 7, 30, "Wake up", Instant.parse(expiry)),
            PhoneCommand.parse("""{"kind":"alarm","id":"a","hour":7,"minute":30,"label":"Wake up","expires_at":"$expiry"}"""),
        )
        assertEquals(
            PhoneCommand.Timer("t", 90, null, Instant.parse(expiry)),
            PhoneCommand.parse("""{"kind":"timer","id":"t","seconds":90,"expires_at":"$expiry"}"""),
        )
        assertEquals(
            PhoneCommand.Location("l", Instant.parse(expiry)),
            PhoneCommand.parse("""{"kind":"location","id":"l","expires_at":"$expiry"}"""),
        )
    }

    @Test
    fun intentBuildersMatchPhoneActions() {
        val navigate = PhoneIntents.intentFor(
            PhoneCommand.Navigate("n", "Trader Joe's", "123 Main St", "ChIJabc", 40.1, -73.2, Instant.parse(expiry)),
        )
        assertEquals(Intent.ACTION_VIEW, navigate.action)
        assertEquals(
            Uri.parse("https://www.google.com/maps/dir/?api=1&destination=Trader%20Joe%27s%2C%20123%20Main%20St&destination_place_id=ChIJabc&travelmode=driving&dir_action=navigate"),
            navigate.data,
        )
        assertEquals("com.google.android.apps.maps", navigate.`package`)
        assertTrue(navigate.flags and Intent.FLAG_ACTIVITY_NEW_TASK != 0)

        val dial = PhoneIntents.intentFor(PhoneCommand.Dial("d", "+15551212", Instant.parse(expiry)))
        assertEquals(Intent.ACTION_DIAL, dial.action)
        assertEquals(Uri.parse("tel:+15551212"), dial.data)

        val alarm = PhoneIntents.intentFor(PhoneCommand.Alarm("a", 7, 30, "Wake up", Instant.parse(expiry)))
        assertEquals(AlarmClock.ACTION_SET_ALARM, alarm.action)
        assertEquals(7, alarm.getIntExtra(AlarmClock.EXTRA_HOUR, -1))
        assertEquals(30, alarm.getIntExtra(AlarmClock.EXTRA_MINUTES, -1))
        assertEquals("Wake up", alarm.getStringExtra(AlarmClock.EXTRA_MESSAGE))
        assertTrue(alarm.getBooleanExtra(AlarmClock.EXTRA_SKIP_UI, false))

        val timer = PhoneIntents.intentFor(PhoneCommand.Timer("t", 90, null, Instant.parse(expiry)))
        assertEquals(AlarmClock.ACTION_SET_TIMER, timer.action)
        assertEquals(90, timer.getIntExtra(AlarmClock.EXTRA_LENGTH, -1))
        assertFalse(timer.hasExtra(AlarmClock.EXTRA_MESSAGE))
    }

    @Test
    fun malformedBridgeCommandsAreRejected() {
        val payloads = listOf(
            "{}",
            """{"kind":"navigate","id":"n","name":"x","address":"y","place_id":"p","lat":91,"lng":0,"expires_at":"$expiry"}""",
            """{"kind":"dial","id":"d","number":"+1"}""",
            """{"kind":"alarm","id":"a","hour":24,"minute":1,"expires_at":"$expiry"}""",
            """{"kind":"timer","id":"t","seconds":-1,"expires_at":"$expiry"}""",
            """{"kind":"location","id":"l","expires_at":"tomorrow"}""",
            """{"kind":"unknown","id":"x","expires_at":"$expiry"}""",
        )
        payloads.forEach { payload ->
            assertTrue("$payload did not fail", runCatching { PhoneCommand.parse(payload) }.isFailure)
        }
    }
}

package gg.savecraft.mentat.phone

import android.provider.AlarmClock
import android.content.ActivityNotFoundException
import android.content.Intent
import android.net.Uri
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.shadows.ShadowLog
import org.robolectric.annotation.Config

@Config(sdk = [35])
@RunWith(RobolectricTestRunner::class)
class PhoneCommandsTest {
    @Test
    fun navigateBuildsMapsDirectionsIntent() {
        val command = PhoneCommands.parse(
            """{"kind":"navigate","name":"Trader Joe's","address":"123 Main St","place_id":"ChIJabc","lat":40.1,"lng":-73.2}""",
        )
        val intent = PhoneCommands.intentFor(command)

        assertEquals(Intent.ACTION_VIEW, intent.action)
        assertEquals(
            Uri.parse(
                "https://www.google.com/maps/dir/?api=1&destination=Trader%20Joe%27s%2C%20123%20Main%20St&destination_place_id=ChIJabc&travelmode=driving&dir_action=navigate",
            ),
            intent.data,
        )
        assertEquals("com.google.android.apps.maps", intent.`package`)
        assertTrue((intent.flags and Intent.FLAG_ACTIVITY_NEW_TASK) != 0)
    }

    @Test
    fun dialBuildsDialerIntent() {
        val intent = PhoneCommands.intentFor(PhoneCommands.parse("""{"kind":"dial","number":"+15551212"}"""))

        assertEquals(Intent.ACTION_DIAL, intent.action)
        assertEquals(Uri.parse("tel:+15551212"), intent.data)
        assertTrue((intent.flags and Intent.FLAG_ACTIVITY_NEW_TASK) != 0)
    }

    @Test
    fun smsBuildsPrefilledMessagesIntent() {
        val intent = PhoneCommands.intentFor(
            PhoneCommands.parse("""{"kind":"sms","number":"+15551212","body":"Running late"}"""),
        )

        assertEquals(Intent.ACTION_SENDTO, intent.action)
        assertEquals(Uri.parse("smsto:+15551212"), intent.data)
        assertEquals("Running late", intent.getStringExtra("sms_body"))
        assertTrue((intent.flags and Intent.FLAG_ACTIVITY_NEW_TASK) != 0)
    }

    @Test
    fun alarmBuildsClockIntent() {
        val intent = PhoneCommands.intentFor(
            PhoneCommands.parse("""{"kind":"alarm","hour":7,"minute":30,"label":"Wake up"}"""),
        )

        assertEquals(AlarmClock.ACTION_SET_ALARM, intent.action)
        assertEquals(7, intent.getIntExtra(AlarmClock.EXTRA_HOUR, -1))
        assertEquals(30, intent.getIntExtra(AlarmClock.EXTRA_MINUTES, -1))
        assertEquals("Wake up", intent.getStringExtra(AlarmClock.EXTRA_MESSAGE))
        assertTrue(intent.getBooleanExtra(AlarmClock.EXTRA_SKIP_UI, false))
        assertTrue((intent.flags and Intent.FLAG_ACTIVITY_NEW_TASK) != 0)
    }

    @Test
    fun alarmOmitsOptionalMessage() {
        val intent = PhoneCommands.intentFor(
            PhoneCommands.parse("""{"kind":"alarm","hour":7,"minute":30}"""),
        )

        assertFalse(intent.hasExtra(AlarmClock.EXTRA_MESSAGE))
    }

    @Test
    fun timerBuildsClockIntent() {
        val intent = PhoneCommands.intentFor(
            PhoneCommands.parse("""{"kind":"timer","seconds":90,"label":"Tea"}"""),
        )

        assertEquals(AlarmClock.ACTION_SET_TIMER, intent.action)
        assertEquals(90, intent.getIntExtra(AlarmClock.EXTRA_LENGTH, -1))
        assertEquals("Tea", intent.getStringExtra(AlarmClock.EXTRA_MESSAGE))
        assertTrue(intent.getBooleanExtra(AlarmClock.EXTRA_SKIP_UI, false))
        assertTrue((intent.flags and Intent.FLAG_ACTIVITY_NEW_TASK) != 0)
    }

    @Test
    fun openBuildsWebIntent() {
        val intent = PhoneCommands.intentFor(
            PhoneCommands.parse("""{"kind":"open","url":"https://example.com/a"}"""),
        )

        assertEquals(Intent.ACTION_VIEW, intent.action)
        assertEquals(Uri.parse("https://example.com/a"), intent.data)
        assertTrue((intent.flags and Intent.FLAG_ACTIVITY_NEW_TASK) != 0)
    }

    @Test
    fun invalidKindsFieldsNumbersRangesAndSchemesUseCode1600() {
        val payloads = listOf(
            "{}",
            """{"kind":"unknown"}""",
            """{"kind":"dial"}""",
            """{"kind":"alarm","hour":"morning","minute":1}""",
            """{"kind":"alarm","hour":24,"minute":1}""",
            """{"kind":"alarm","hour":1,"minute":60}""",
            """{"kind":"timer","seconds":"soon"}""",
            """{"kind":"timer","seconds":-1}""",
            """{"kind":"open","url":"ftp://example.com"}""",
            """{"kind":"open","url":"intent://example"}""",
            """{"kind":"open","url":"spotify:track:abc"}""",
            """{"kind":"open","url":"javascript:alert(1)"}""",
            """{"kind":"open","url":"relative/path"}""",
            """{"kind":"open","url":""}""",
        )

        payloads.forEach { payload ->
            val error = runCatching { PhoneCommands.parse(payload) }.exceptionOrNull()
            assertTrue("$payload did not fail", error is PhoneRpcException)
            assertEquals(1600, (error as PhoneRpcException).code)
        }
    }

    @Test
    fun bridgeRegistersCommandAndLocationOnlyOnce() {
        val methods = mutableListOf<String>()
        val bridge = PhoneBridge(location = null, launcher = {})
        val registrar = PhoneRpcRegistrar { method, _ -> methods += method }

        bridge.register(registrar)
        bridge.register(registrar)

        assertEquals(listOf("mentat.command", "mentat.location"), methods)
    }

    @Test
    fun bridgeRefusesCommandsWhenAssistIsNotVisible() {
        var launches = 0
        val bridge = PhoneBridge(
            location = null,
            launcher = { launches++ },
        )

        val error = runCatching {
            bridge.handleCommand("""{"kind":"dial","number":"+15551212"}""")
        }.exceptionOrNull()

        assertEquals(1603, (error as PhoneRpcException).code)
        assertEquals(0, launches)
    }

    @Test
    fun bridgeMapsMissingActivityTo1601() {
        val bridge = PhoneBridge(
            location = null,
            launcher = { throw ActivityNotFoundException("no dialer") },
        )
        bridge.assistVisible = true

        val error = runCatching {
            bridge.handleCommand("""{"kind":"dial","number":"+15551212"}""")
        }.exceptionOrNull()

        assertEquals(1601, (error as PhoneRpcException).code)
    }

    @Test
    fun bridgeSuccessLogDoesNotExposeIntentData() {
        ShadowLog.clear()
        val bridge = PhoneBridge(location = null, launcher = {})
        bridge.assistVisible = true

        bridge.handleCommand(
            """{"kind":"open","url":"https://example.invalid/p?marker=PHONE_LOG_SECRET"}""",
        )

        assertTrue(
            ShadowLog.getLogsForTag("MentatAssist").none {
                it.msg.contains("PHONE_LOG_SECRET")
            },
        )
    }

    @Test
    fun bridgeReturnsSuccessAfterLaunch() {
        var launched: Intent? = null
        val bridge = PhoneBridge(
            location = null,
            launcher = { launched = it },
        )
        bridge.assistVisible = true

        assertEquals(
            """{"ok":true}""",
            bridge.handleCommand("""{"kind":"dial","number":"+15551212"}"""),
        )
        assertEquals(Intent.ACTION_DIAL, launched?.action)
    }
}

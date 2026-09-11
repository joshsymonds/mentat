package gg.savecraft.mentat.core

import android.content.Intent
import android.net.Uri
import android.provider.AlarmClock

object PhoneIntents {
    fun intentFor(command: PhoneCommand): Intent = when (command) {
        is PhoneCommand.Navigate -> Intent(
            Intent.ACTION_VIEW,
            Uri.parse(
                "https://www.google.com/maps/dir/?api=1" +
                    "&destination=${encodeQueryComponent(command.name + ", " + command.address)}" +
                    "&destination_place_id=${encodeQueryComponent(command.placeId)}" +
                    "&travelmode=driving&dir_action=navigate",
            ),
        ).setPackage("com.google.android.apps.maps")

        is PhoneCommand.Dial -> Intent(Intent.ACTION_DIAL, Uri.parse("tel:${command.number}"))

        is PhoneCommand.Sms -> Intent(Intent.ACTION_SENDTO, Uri.parse("smsto:${command.to}")).apply {
            putExtra("sms_body", command.body)
        }

        is PhoneCommand.Alarm -> Intent(AlarmClock.ACTION_SET_ALARM).apply {
            putExtra(AlarmClock.EXTRA_HOUR, command.hour)
            putExtra(AlarmClock.EXTRA_MINUTES, command.minute)
            command.label?.let { putExtra(AlarmClock.EXTRA_MESSAGE, it) }
            putExtra(AlarmClock.EXTRA_SKIP_UI, true)
        }

        is PhoneCommand.Timer -> Intent(AlarmClock.ACTION_SET_TIMER).apply {
            putExtra(AlarmClock.EXTRA_LENGTH, command.seconds)
            command.label?.let { putExtra(AlarmClock.EXTRA_MESSAGE, it) }
            putExtra(AlarmClock.EXTRA_SKIP_UI, true)
        }

        is PhoneCommand.Open -> Intent(Intent.ACTION_VIEW, Uri.parse(command.uri))
        is PhoneCommand.Conversations,
        is PhoneCommand.Messages,
        is PhoneCommand.Search,
        is PhoneCommand.Location,
        PhoneCommand.Ping -> throw IllegalArgumentException("Command does not launch an intent")
    }.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)

    private fun encodeQueryComponent(value: String): String = buildString {
        value.toByteArray(Charsets.UTF_8).forEach { byte ->
            val character = byte.toInt() and 0xff
            if (
                character in 'A'.code..'Z'.code ||
                    character in 'a'.code..'z'.code ||
                    character in '0'.code..'9'.code ||
                    character == '-'.code ||
                    character == '.'.code ||
                    character == '_'.code ||
                    character == '~'.code
            ) {
                append(character.toChar())
            } else {
                append('%')
                append("%02X".format(java.util.Locale.ROOT, character))
            }
        }
    }
}

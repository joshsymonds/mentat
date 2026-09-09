package gg.savecraft.mentat.phone

import android.content.Intent
import android.net.Uri
import android.provider.AlarmClock
import kotlin.math.floor

sealed interface PhoneCommand {
    data class Navigate(
        val name: String,
        val address: String,
        val placeId: String,
        val lat: Double,
        val lng: Double,
    ) : PhoneCommand

    data class Dial(val number: String) : PhoneCommand

    data class Sms(val number: String, val body: String) : PhoneCommand

    data class Alarm(val hour: Int, val minute: Int, val label: String?) : PhoneCommand

    data class Timer(val seconds: Int, val label: String?) : PhoneCommand

    data class Open(val url: String) : PhoneCommand
}

object PhoneCommands {
    fun parse(json: String): PhoneCommand {
        val payload = try {
            org.json.JSONObject(json)
        } catch (exception: Exception) {
            throw invalid("invalid JSON", exception)
        }
        return try {
            when (requiredString(payload, "kind")) {
                "navigate" -> PhoneCommand.Navigate(
                    name = requiredString(payload, "name"),
                    address = requiredString(payload, "address"),
                    placeId = requiredString(payload, "place_id"),
                    lat = requiredDouble(payload, "lat", -90.0, 90.0),
                    lng = requiredDouble(payload, "lng", -180.0, 180.0),
                )

                "dial" -> PhoneCommand.Dial(requiredString(payload, "number"))
                "sms" -> PhoneCommand.Sms(
                    number = requiredString(payload, "number"),
                    body = requiredString(payload, "body"),
                )

                "alarm" -> PhoneCommand.Alarm(
                    hour = requiredInt(payload, "hour", 0, 23),
                    minute = requiredInt(payload, "minute", 0, 59),
                    label = optionalString(payload, "label"),
                )

                "timer" -> PhoneCommand.Timer(
                    seconds = requiredInt(payload, "seconds", 0, Int.MAX_VALUE),
                    label = optionalString(payload, "label"),
                )

                "open" -> PhoneCommand.Open(requiredWebUrl(payload, "url"))
                else -> throw invalid("unknown command kind")
            }
        } catch (exception: PhoneRpcException) {
            throw exception
        } catch (exception: Exception) {
            throw invalid("invalid command", exception)
        }
    }

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
        is PhoneCommand.Sms -> Intent(Intent.ACTION_SENDTO, Uri.parse("smsto:${command.number}")).apply {
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

        is PhoneCommand.Open -> Intent(Intent.ACTION_VIEW, Uri.parse(validateWebUrl(command.url)))
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

    private fun requiredString(payload: org.json.JSONObject, name: String): String {
        val value = payload.opt(name)
        if (value !is String || value.isBlank()) {
            throw invalid("missing or invalid $name")
        }
        return value
    }

    private fun optionalString(payload: org.json.JSONObject, name: String): String? {
        if (!payload.has(name) || payload.isNull(name)) {
            return null
        }
        val value = payload.opt(name)
        if (value !is String) {
            throw invalid("invalid $name")
        }
        return value
    }

    private fun requiredDouble(
        payload: org.json.JSONObject,
        name: String,
        minimum: Double,
        maximum: Double,
    ): Double {
        val value = payload.opt(name)
        val number = value as? Number
        val result = number?.toDouble()
        if (result == null || !result.isFinite() || result < minimum || result > maximum) {
            throw invalid("missing or invalid $name")
        }
        return result
    }

    private fun requiredInt(
        payload: org.json.JSONObject,
        name: String,
        minimum: Int,
        maximum: Int,
    ): Int {
        val value = payload.opt(name)
        val number = value as? Number
        val result = number?.toDouble()
        if (
            result == null ||
            !result.isFinite() ||
            floor(result) != result ||
            result < minimum ||
            result > maximum
        ) {
            throw invalid("missing or invalid $name")
        }
        return result.toInt()
    }

    private fun requiredWebUrl(payload: org.json.JSONObject, name: String): String =
        validateWebUrl(requiredString(payload, name))

    private fun validateWebUrl(value: String): String {
        val uri = Uri.parse(value)
        val scheme = uri.scheme?.lowercase()
        if (scheme != "http" && scheme != "https" || uri.host.isNullOrBlank()) {
            throw invalid("URL scheme is not allowed")
        }
        return value
    }

    private fun invalid(message: String, cause: Throwable? = null): PhoneRpcException =
        PhoneRpcException(PhoneRpcCodes.INVALID_COMMAND, message, cause)
}

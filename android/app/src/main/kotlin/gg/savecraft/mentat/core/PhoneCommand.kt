package gg.savecraft.mentat.core

import java.time.Instant
import kotlin.math.floor
import org.json.JSONObject

sealed class PhoneCommand {
    abstract val id: String?
    abstract val expiresAt: Instant?

    data class Navigate(
        override val id: String,
        val name: String,
        val address: String,
        val placeId: String,
        val lat: Double,
        val lng: Double,
        override val expiresAt: Instant,
    ) : PhoneCommand()

    data class Dial(
        override val id: String,
        val number: String,
        override val expiresAt: Instant,
    ) : PhoneCommand()

    data class Sms(
        override val id: String,
        val to: String,
        val body: String,
        override val expiresAt: Instant,
    ) : PhoneCommand()

    data class Alarm(
        override val id: String,
        val hour: Int,
        val minute: Int,
        val label: String?,
        override val expiresAt: Instant,
    ) : PhoneCommand()

    data class Timer(
        override val id: String,
        val seconds: Int,
        val label: String?,
        override val expiresAt: Instant,
    ) : PhoneCommand()

    data class Open(
        override val id: String,
        val uri: String,
        override val expiresAt: Instant,
    ) : PhoneCommand()

    data class Location(
        override val id: String,
        override val expiresAt: Instant,
    ) : PhoneCommand()

    data class Conversations(
        override val id: String,
        val limit: Int,
        override val expiresAt: Instant,
    ) : PhoneCommand()

    data class Messages(
        override val id: String,
        val conversation: String,
        val limit: Int,
        val before: String?,
        override val expiresAt: Instant,
    ) : PhoneCommand()

    data class Search(
        override val id: String,
        val query: String,
        val limit: Int,
        val before: String?,
        override val expiresAt: Instant,
    ) : PhoneCommand()

    data object Ping : PhoneCommand() {
        override val id: String? = null
        override val expiresAt: Instant? = null
    }

    companion object {
        fun parse(line: String): PhoneCommand = parse(JSONObject(line))

        fun parse(json: JSONObject): PhoneCommand = when (requiredString(json, "kind")) {
            "navigate" -> Navigate(
                id = requiredString(json, "id"),
                name = requiredString(json, "name"),
                address = requiredString(json, "address"),
                placeId = requiredString(json, "place_id"),
                lat = requiredDouble(json, "lat", -90.0, 90.0),
                lng = requiredDouble(json, "lng", -180.0, 180.0),
                expiresAt = requiredInstant(json),
            )

            "dial" -> Dial(
                id = requiredString(json, "id"),
                number = requiredString(json, "number"),
                expiresAt = requiredInstant(json),
            )

            "sms" -> Sms(
                id = requiredString(json, "id"),
                to = requiredString(json, "to"),
                body = requiredString(json, "body"),
                expiresAt = requiredInstant(json),
            )

            "alarm" -> Alarm(
                id = requiredString(json, "id"),
                hour = requiredInt(json, "hour", 0, 23),
                minute = requiredInt(json, "minute", 0, 59),
                label = optionalString(json, "label"),
                expiresAt = requiredInstant(json),
            )

            "timer" -> Timer(
                id = requiredString(json, "id"),
                seconds = requiredInt(json, "seconds", 0, Int.MAX_VALUE),
                label = optionalString(json, "label"),
                expiresAt = requiredInstant(json),
            )

            "open" -> Open(
                id = requiredString(json, "id"),
                uri = requiredString(json, "uri"),
                expiresAt = requiredInstant(json),
            )

            "location" -> Location(
                id = requiredString(json, "id"),
                expiresAt = requiredInstant(json),
            )

            "conversations" -> Conversations(
                id = requiredString(json, "id"),
                limit = requiredInt(json, "limit", 1, 100),
                expiresAt = requiredInstant(json),
            )

            "messages" -> Messages(
                id = requiredString(json, "id"),
                conversation = requiredString(json, "conversation"),
                limit = requiredInt(json, "limit", 1, 100),
                before = optionalString(json, "before"),
                expiresAt = requiredInstant(json),
            )

            "search" -> Search(
                id = requiredString(json, "id"),
                query = requiredString(json, "query"),
                limit = requiredInt(json, "limit", 1, 100),
                before = optionalString(json, "before"),
                expiresAt = requiredInstant(json),
            )

            "ping" -> Ping
            else -> throw IllegalArgumentException("Unknown phone command kind")
        }

        private fun requiredString(json: JSONObject, name: String): String {
            val value = json.opt(name)
            if (value !is String || value.isBlank()) {
                throw IllegalArgumentException("Missing or invalid $name")
            }
            return value
        }

        private fun optionalString(json: JSONObject, name: String): String? {
            if (!json.has(name) || json.isNull(name)) return null
            val value = json.opt(name)
            if (value !is String) throw IllegalArgumentException("Invalid $name")
            return value
        }

        private fun requiredInstant(json: JSONObject): Instant =
            Instant.parse(requiredString(json, "expires_at"))

        private fun requiredDouble(
            json: JSONObject,
            name: String,
            minimum: Double,
            maximum: Double,
        ): Double {
            val value = json.opt(name) as? Number
            val result = value?.toDouble()
            if (result == null || !result.isFinite() || result < minimum || result > maximum) {
                throw IllegalArgumentException("Missing or invalid $name")
            }
            return result
        }

        private fun requiredInt(
            json: JSONObject,
            name: String,
            minimum: Int,
            maximum: Int,
        ): Int {
            val result = (json.opt(name) as? Number)?.toDouble()
            if (
                result == null ||
                    !result.isFinite() ||
                    floor(result) != result ||
                    result < minimum ||
                    result > maximum
            ) {
                throw IllegalArgumentException("Missing or invalid $name")
            }
            return result.toInt()
        }
    }
}

data class PhoneResult(
    val id: String,
    val status: String,
    val detail: String,
    val payload: JSONObject? = null,
) {
    fun toJson(): JSONObject = JSONObject()
        .put("id", id)
        .put("status", status)
        .put("detail", detail)
        .apply { payload?.let { put("payload", it) } }
}

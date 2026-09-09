package gg.savecraft.mentat.core

import java.time.Instant
import org.json.JSONObject

sealed class PhoneCommand {
    abstract val id: String?
    abstract val expiresAt: Instant?

    data class Sms(
        override val id: String,
        val to: String,
        val body: String,
        override val expiresAt: Instant,
    ) : PhoneCommand()

    data class Open(
        override val id: String,
        val uri: String,
        override val expiresAt: Instant,
    ) : PhoneCommand()

    data object Ping : PhoneCommand() {
        override val id: String? = null
        override val expiresAt: Instant? = null
    }

    companion object {
        fun parse(json: JSONObject): PhoneCommand {
            return when (json.getString("kind")) {
                "sms" -> Sms(
                    id = json.getString("id"),
                    to = json.getString("to"),
                    body = json.getString("body"),
                    expiresAt = Instant.parse(json.getString("expires_at")),
                )
                "open" -> Open(
                    id = json.getString("id"),
                    uri = json.getString("uri"),
                    expiresAt = Instant.parse(json.getString("expires_at")),
                )
                "ping" -> Ping
                else -> throw IllegalArgumentException("Unknown phone command kind")
            }
        }

        fun parse(line: String): PhoneCommand = parse(JSONObject(line))
    }
}

data class PhoneResult(
    val id: String,
    val status: String,
    val detail: String,
) {
    fun toJson(): JSONObject = JSONObject()
        .put("id", id)
        .put("status", status)
        .put("detail", detail)
}

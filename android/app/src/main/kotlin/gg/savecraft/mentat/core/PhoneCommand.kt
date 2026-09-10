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
                "conversations" -> Conversations(
                    id = json.getString("id"),
                    limit = json.getInt("limit"),
                    expiresAt = Instant.parse(json.getString("expires_at")),
                )
                "messages" -> Messages(
                    id = json.getString("id"),
                    conversation = json.getString("conversation"),
                    limit = json.getInt("limit"),
                    before = json.stringOrNull("before"),
                    expiresAt = Instant.parse(json.getString("expires_at")),
                )
                "search" -> Search(
                    id = json.getString("id"),
                    query = json.getString("query"),
                    limit = json.getInt("limit"),
                    before = json.stringOrNull("before"),
                    expiresAt = Instant.parse(json.getString("expires_at")),
                )
                "ping" -> Ping
                else -> throw IllegalArgumentException("Unknown phone command kind")
            }
        }

        private fun JSONObject.stringOrNull(name: String): String? =
            if (isNull(name)) null else optString(name, null)

        fun parse(line: String): PhoneCommand = parse(JSONObject(line))
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

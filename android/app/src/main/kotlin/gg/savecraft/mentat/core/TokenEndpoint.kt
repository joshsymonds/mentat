package gg.savecraft.mentat.core

import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URI
import java.net.URL
import java.time.Instant
import org.json.JSONObject

data class TokenGrant(
    val token: String,
    val room: String,
    val url: String,
    val expiresAt: String,
)

interface TokenEndpoint {
    fun fetch(context: CallContext? = null): TokenGrant
}

class TokenFetchException(
    message: String,
    cause: Throwable? = null,
) : Exception(message, cause)

class HttpTokenEndpoint(
    baseUrl: String,
    val connectTimeoutMillis: Int = DEFAULT_TIMEOUT_MILLIS,
    val readTimeoutMillis: Int = DEFAULT_TIMEOUT_MILLIS,
    val maxResponseBytes: Int = DEFAULT_MAX_RESPONSE_BYTES,
) : TokenEndpoint {
    private val endpointUrl = "${baseUrl.trimEnd('/')}/v1/voice/token"

    override fun fetch(context: CallContext?): TokenGrant {
        var connection: HttpURLConnection? = null
        val body = context?.toRequestBody()?.toByteArray(Charsets.UTF_8) ?: ByteArray(0)
        try {
            connection = URL(endpointUrl).openConnection() as HttpURLConnection
            connection.connectTimeout = connectTimeoutMillis
            connection.readTimeout = readTimeoutMillis
            connection.requestMethod = "POST"
            connection.doOutput = true
            if (body.isNotEmpty()) {
                connection.setRequestProperty("Content-Type", "application/json")
            }
            connection.setFixedLengthStreamingMode(body.size)
            connection.outputStream.use { it.write(body) }

            val status = connection.responseCode
            if (status != HttpURLConnection.HTTP_OK) {
                connection.errorStream?.close()
                throw TokenFetchException("Token endpoint returned HTTP $status")
            }

            val body = connection.inputStream.use(::readBounded)
            return parseGrant(body)
        } catch (exception: TokenFetchException) {
            throw exception
        } catch (exception: Exception) {
            throw TokenFetchException("Token request failed", exception)
        } finally {
            connection?.disconnect()
        }
    }

    private fun readBounded(stream: InputStream): String {
        val body = ByteArrayOutputStream()
        val chunk = ByteArray(CHUNK_BYTES)
        while (true) {
            val read = stream.read(chunk)
            if (read == -1) {
                return body.toString(Charsets.UTF_8.name())
            }
            if (body.size() + read > maxResponseBytes) {
                throw TokenFetchException("Token response exceeded $maxResponseBytes bytes")
            }
            body.write(chunk, 0, read)
        }
    }

    private fun parseGrant(body: String): TokenGrant {
        try {
            val json = JSONObject(body)
            val url = json.getString("url")
            val expiresAt = json.getString("expires_at")
            val scheme = URI.create(url).scheme
            if (scheme != "ws" && scheme != "wss") {
                throw IllegalArgumentException("Token URL must use ws or wss")
            }
            Instant.parse(expiresAt)

            return TokenGrant(
                token = json.getString("token"),
                room = json.getString("room"),
                url = url,
                expiresAt = expiresAt,
            )
        } catch (exception: Exception) {
            throw TokenFetchException("Invalid token response", exception)
        }
    }

    private companion object {
        const val DEFAULT_TIMEOUT_MILLIS = 10_000
        const val DEFAULT_MAX_RESPONSE_BYTES = 64 * 1024
        const val CHUNK_BYTES = 8 * 1024
    }
}

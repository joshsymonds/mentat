package gg.savecraft.mentat.core

import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext

interface CommandStream {
    suspend fun run(handler: suspend (PhoneCommand) -> PhoneResult)
    fun close()
}

class HttpCommandStream(
    baseUrl: String,
    private val connectTimeoutMillis: Int = DEFAULT_CONNECT_TIMEOUT_MILLIS,
    private val readTimeoutMillis: Int = DEFAULT_READ_TIMEOUT_MILLIS,
    private val backoffMillis: (attempt: Int) -> Long = ::defaultBackoffMillis,
) : CommandStream {
    private val commandsUrl = "${baseUrl.trimEnd('/')}/v1/phone/commands"
    private val resultsUrl = "${baseUrl.trimEnd('/')}/v1/phone/results"

    @Volatile
    private var closed = false
    @Volatile
    private var activeConnection: HttpURLConnection? = null

    override suspend fun run(handler: suspend (PhoneCommand) -> PhoneResult) = withContext(Dispatchers.IO) {
        var attempt = 0
        while (isActive && !closed) {
            try {
                readConnection(handler)
                attempt = 0
                throw IOException("Command stream closed")
            } catch (exception: CancellationException) {
                throw exception
            } catch (_: Exception) {
                if (!isActive || closed) {
                    break
                }
                delay(backoffMillis(attempt))
                attempt++
            }
        }
    }

    private suspend fun readConnection(handler: suspend (PhoneCommand) -> PhoneResult) {
        val connection = URL(commandsUrl).openConnection() as HttpURLConnection
        activeConnection = connection
        try {
            connection.connectTimeout = connectTimeoutMillis
            connection.readTimeout = readTimeoutMillis
            connection.requestMethod = "GET"
            connection.setRequestProperty("Accept", "application/x-ndjson")
            if (connection.responseCode != HttpURLConnection.HTTP_OK) {
                throw IOException("Command endpoint returned HTTP ${connection.responseCode}")
            }
            connection.inputStream.bufferedReader().useLines { lines ->
                lines.forEach { line ->
                    currentCoroutineContext().ensureActive()
                    val command = try {
                        PhoneCommand.parse(line)
                    } catch (_: Exception) {
                        return@forEach
                    }
                    if (command == PhoneCommand.Ping) {
                        return@forEach
                    }
                    val result = handler(command)
                    postResult(result)
                }
            }
        } finally {
            if (activeConnection === connection) {
                activeConnection = null
            }
            connection.disconnect()
        }
    }

    private fun postResult(result: PhoneResult) {
        val connection = URL(resultsUrl).openConnection() as HttpURLConnection
        try {
            connection.connectTimeout = connectTimeoutMillis
            connection.readTimeout = connectTimeoutMillis
            connection.requestMethod = "POST"
            connection.doOutput = true
            connection.setRequestProperty("Content-Type", "application/json")
            val bytes = result.toJson().toString().toByteArray(Charsets.UTF_8)
            connection.setFixedLengthStreamingMode(bytes.size)
            connection.outputStream.use { it.write(bytes) }
            val responseCode = connection.responseCode
            if (responseCode !in 200..299) {
                throw IOException("Result endpoint returned HTTP $responseCode")
            }
        } finally {
            connection.disconnect()
        }
    }

    override fun close() {
        closed = true
        activeConnection?.disconnect()
    }

    private companion object {
        const val DEFAULT_CONNECT_TIMEOUT_MILLIS = 10_000
        const val DEFAULT_READ_TIMEOUT_MILLIS = 45_000

        fun defaultBackoffMillis(attempt: Int): Long =
            (1_000L shl attempt.coerceAtMost(6)).coerceAtMost(60_000L)
    }
}

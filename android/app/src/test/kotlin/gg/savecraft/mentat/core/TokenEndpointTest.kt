package gg.savecraft.mentat.core

import com.sun.net.httpserver.HttpServer
import java.io.IOException
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.ServerSocket
import java.net.Socket
import java.util.Collections
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.json.JSONObject
import org.junit.Test

class TokenEndpointTest {
    @Test
    fun fetchPostsEmptyBodyAndParsesGrant() {
        val method = AtomicReference<String>()
        val bodySize = AtomicInteger(-1)
        val response = """
            {
              "token": "jwt-token",
              "room": "android-room",
              "url": "wss://voice.example.com",
              "expires_at": "2026-08-19T16:00:00.000Z"
            }
        """.trimIndent()

        withServer(
            response = response,
            onRequest = { requestMethod, requestBody ->
                method.set(requestMethod)
                bodySize.set(requestBody.size)
            },
        ) { baseUrl ->
            assertEquals(
                TokenGrant(
                    token = "jwt-token",
                    room = "android-room",
                    url = "wss://voice.example.com",
                    expiresAt = "2026-08-19T16:00:00.000Z",
                ),
                HttpTokenEndpoint(baseUrl).fetch(),
            )
        }

        assertEquals("POST", method.get())
        assertEquals(0, bodySize.get())
    }

    @Test
    fun fetchPostsTheCallContextAsJson() {
        val body = AtomicReference<String>()
        val response = """
            {"token": "t", "room": "r", "url": "wss://voice.example.com", "expires_at": "2026-08-19T16:00:00.000Z"}
        """.trimIndent()
        val location = JSONObject().put("lat", 47.6).put("lng", -122.3).put("accuracy_m", 12.0).put("age_s", 30)

        withServer(response = response, onRequest = { _, requestBody -> body.set(String(requestBody)) }) { baseUrl ->
            HttpTokenEndpoint(baseUrl).fetch(CallContext("America/Los_Angeles", location, driving = false))
        }

        val context = JSONObject(body.get()).getJSONObject("context")
        assertEquals("America/Los_Angeles", context.getString("time_zone"))
        assertEquals(false, context.getBoolean("driving"))
        assertEquals(47.6, context.getJSONObject("location").getDouble("lat"), 0.00001)
    }

    @Test
    fun nonSuccessResponseThrowsTokenFetchException() {
        withServer(status = 404, response = "not found") { baseUrl ->
            assertThrows(TokenFetchException::class.java) {
                HttpTokenEndpoint(baseUrl).fetch()
            }
        }
    }

    @Test
    fun malformedJsonThrowsTokenFetchException() {
        withServer(response = "{not json") { baseUrl ->
            assertThrows(TokenFetchException::class.java) {
                HttpTokenEndpoint(baseUrl).fetch()
            }
        }
    }

    @Test
    fun nonWebSocketUrlThrowsTokenFetchException() {
        val response = """
            {
              "token": "jwt-token",
              "room": "android-room",
              "url": "https://voice.example.com",
              "expires_at": "2026-08-19T16:00:00.000Z"
            }
        """.trimIndent()

        withServer(response = response) { baseUrl ->
            assertThrows(TokenFetchException::class.java) {
                HttpTokenEndpoint(baseUrl).fetch()
            }
        }
    }

    @Test
    fun nonIsoExpiryThrowsTokenFetchException() {
        val response = """
            {
              "token": "jwt-token",
              "room": "android-room",
              "url": "wss://voice.example.com",
              "expires_at": "tomorrow"
            }
        """.trimIndent()

        withServer(response = response) { baseUrl ->
            assertThrows(TokenFetchException::class.java) {
                HttpTokenEndpoint(baseUrl).fetch()
            }
        }
    }

    @Test
    fun refusedConnectionThrowsTokenFetchException() {
        val port = ServerSocket(0, 1, InetAddress.getLoopbackAddress()).use { it.localPort }

        assertThrows(TokenFetchException::class.java) {
            HttpTokenEndpoint("http://127.0.0.1:$port").fetch()
        }
    }

    @Test
    fun stalledServerThrowsTokenFetchException() {
        val server = ServerSocket(0, 1, InetAddress.getLoopbackAddress())
        val accepted = Collections.synchronizedList(mutableListOf<Socket>())
        val acceptor = Thread {
            try {
                while (true) {
                    accepted.add(server.accept())
                }
            } catch (_: IOException) {
                // The socket is closed once the assertion below finishes.
            }
        }
        acceptor.isDaemon = true
        acceptor.start()

        try {
            assertThrows(TokenFetchException::class.java) {
                HttpTokenEndpoint(
                    "http://127.0.0.1:${server.localPort}",
                    connectTimeoutMillis = 250,
                    readTimeoutMillis = 250,
                ).fetch()
            }
        } finally {
            server.close()
            accepted.forEach { it.close() }
        }
    }

    @Test
    fun oversizedResponseThrowsTokenFetchException() {
        val response = """
            {
              "token": "${"j".repeat(4096)}",
              "room": "android-room",
              "url": "wss://voice.example.com",
              "expires_at": "2026-08-19T16:00:00.000Z"
            }
        """.trimIndent()

        withServer(response = response) { baseUrl ->
            val thrown = assertThrows(TokenFetchException::class.java) {
                HttpTokenEndpoint(baseUrl, maxResponseBytes = 1024).fetch()
            }
            assertEquals("Token response exceeded 1024 bytes", thrown.message)
        }
    }

    @Test
    fun responseWithinTheCapIsParsed() {
        val response = """
            {
              "token": "jwt-token",
              "room": "android-room",
              "url": "wss://voice.example.com",
              "expires_at": "2026-08-19T16:00:00.000Z"
            }
        """.trimIndent()

        withServer(response = response) { baseUrl ->
            assertEquals(
                "jwt-token",
                HttpTokenEndpoint(baseUrl, maxResponseBytes = response.length).fetch().token,
            )
        }
    }

    @Test
    fun productionDefaultsAreFiniteAndBounded() {
        val endpoint = HttpTokenEndpoint("http://127.0.0.1:1")

        assertEquals(10_000, endpoint.connectTimeoutMillis)
        assertEquals(10_000, endpoint.readTimeoutMillis)
        assertEquals(64 * 1024, endpoint.maxResponseBytes)
    }

    private fun withServer(
        status: Int = 200,
        response: String,
        onRequest: (String, ByteArray) -> Unit = { _, _ -> },
        block: (String) -> Unit,
    ) {
        val server = HttpServer.create(InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0)
        server.createContext("/v1/voice/token") { exchange ->
            val requestBody = exchange.requestBody.use { it.readBytes() }
            onRequest(exchange.requestMethod, requestBody)
            val bytes = response.toByteArray()
            exchange.sendResponseHeaders(status, bytes.size.toLong())
            exchange.responseBody.use { it.write(bytes) }
        }
        server.start()
        try {
            block("http://127.0.0.1:${server.address.port}")
        } finally {
            server.stop(0)
        }
    }
}

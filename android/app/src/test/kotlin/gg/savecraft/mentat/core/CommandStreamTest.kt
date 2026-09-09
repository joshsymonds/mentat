package gg.savecraft.mentat.core

import com.sun.net.httpserver.HttpServer
import java.net.InetAddress
import java.net.InetSocketAddress
import java.time.Instant
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class CommandStreamTest {
    @Test
    fun readsCommandsPostsResultsAndReconnectsAfterDrop() = runBlocking {
        val commandRequests = AtomicInteger()
        val results = ConcurrentLinkedQueue<String>()
        val bothCommands = CountDownLatch(2)
        val server = HttpServer.create(InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0)
        server.createContext("/v1/phone/commands") { exchange ->
            val request = commandRequests.incrementAndGet()
            exchange.responseHeaders.add("Content-Type", "application/x-ndjson")
            exchange.sendResponseHeaders(200, 0)
            exchange.responseBody.bufferedWriter().use { writer ->
                if (request == 1) {
                    writer.appendLine(
                        "{\"id\":\"one\",\"kind\":\"sms\",\"to\":\"+15551212\",\"body\":\"hello\",\"expires_at\":\"2026-09-10T00:00:00Z\"}",
                    )
                    writer.appendLine(
                        "{\"id\":\"two\",\"kind\":\"open\",\"uri\":\"https://example.com\",\"expires_at\":\"2026-09-10T00:00:00Z\"}",
                    )
                    writer.flush()
                } else {
                    writer.appendLine("{\"kind\":\"ping\"}")
                    writer.flush()
                }
            }
        }
        server.createContext("/v1/phone/results") { exchange ->
            results.add(exchange.requestBody.bufferedReader().use { it.readText() })
            exchange.sendResponseHeaders(204, -1)
            exchange.close()
            bothCommands.countDown()
        }
        server.start()

        val stream = HttpCommandStream(
            "http://127.0.0.1:${server.address.port}",
            backoffMillis = { 0L },
        )
        val executor = launch {
            stream.run { command ->
                val id = requireNotNull(command.id)
                bothCommands.countDown()
                PhoneResult(id, "ok", "done")
            }
        }

        try {
            withTimeout(5_000) { while (commandRequests.get() < 2) delay(10) }
            assertTrue(bothCommands.await(5, TimeUnit.SECONDS))
            executor.cancel()
            executor.join()
            assertEquals(listOf("one", "two"), results.map { JSONObject(it).getString("id") }.toList())
            assertEquals(2, commandRequests.get())
        } finally {
            executor.cancel()
            server.stop(0)
        }
    }

    @Test
    fun parsesAndEncodesPhoneMessages() {
        val command = PhoneCommand.parse(
            JSONObject(
                "{\"id\":\"abc\",\"kind\":\"sms\",\"to\":\"Mum\",\"body\":\"Hi\",\"expires_at\":\"2026-09-10T00:00:00Z\"}",
            ),
        )
        assertEquals(
            PhoneCommand.Sms("abc", "Mum", "Hi", Instant.parse("2026-09-10T00:00:00Z")),
            command,
        )
        val result = PhoneResult("abc", "ok", "sent").toJson()
        assertEquals("abc", result.getString("id"))
        assertEquals("ok", result.getString("status"))
        assertEquals("sent", result.getString("detail"))
        assertEquals(PhoneCommand.Ping, PhoneCommand.parse(JSONObject("{\"kind\":\"ping\"}")))
    }
}

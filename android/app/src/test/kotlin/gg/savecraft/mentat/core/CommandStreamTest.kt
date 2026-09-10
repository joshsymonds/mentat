package gg.savecraft.mentat.core

import com.sun.net.httpserver.HttpServer
import java.net.InetAddress
import java.net.InetSocketAddress
import java.time.Instant
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicReference
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
        val commandPhoneHeader = AtomicReference<String?>()
        val results = ConcurrentLinkedQueue<String>()
        val bothCommands = CountDownLatch(2)
        val server = HttpServer.create(InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0)
        server.createContext("/v1/phone/commands") { exchange ->
            commandPhoneHeader.set(exchange.requestHeaders.getFirst("X-Mentat-Phone"))
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
            assertEquals("1", commandPhoneHeader.get())
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
    fun streamsReadCommandsAndSerializesPayloads() = runBlocking {
        val commandRequests = AtomicInteger()
        val beforeSeen = AtomicReference<String?>()
        val results = ConcurrentLinkedQueue<String>()
        val threeResults = CountDownLatch(3)
        val server = HttpServer.create(InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0)
        server.createContext("/v1/phone/commands") { exchange ->
            val request = commandRequests.incrementAndGet()
            exchange.responseHeaders.add("Content-Type", "application/x-ndjson")
            exchange.sendResponseHeaders(200, 0)
            exchange.responseBody.bufferedWriter().use { writer ->
                if (request == 1) {
                    writer.appendLine("{\"id\":\"c\",\"kind\":\"conversations\",\"limit\":20,\"expires_at\":\"2026-09-10T00:00:00Z\"}")
                    writer.appendLine("{\"id\":\"m\",\"kind\":\"messages\",\"conversation\":\"sms:42\",\"limit\":50,\"before\":\"123:sms:s1\",\"expires_at\":\"2026-09-10T00:00:00Z\"}")
                    writer.appendLine("{\"id\":\"s\",\"kind\":\"search\",\"query\":\"hello\",\"limit\":30,\"expires_at\":\"2026-09-10T00:00:00Z\"}")
                } else {
                    writer.appendLine("{\"kind\":\"ping\"}")
                }
                writer.flush()
            }
        }
        server.createContext("/v1/phone/results") { exchange ->
            results.add(exchange.requestBody.bufferedReader().use { it.readText() })
            exchange.sendResponseHeaders(204, -1)
            exchange.close()
            threeResults.countDown()
        }
        server.start()

        val stream = HttpCommandStream(
            "http://127.0.0.1:${server.address.port}",
            backoffMillis = { 0L },
        )
        val runner = launch {
            stream.run { command ->
                when (command) {
                    is PhoneCommand.Conversations -> PhoneResult(
                        command.id,
                        "ok",
                        "conversations",
                        JSONObject().put("conversations", org.json.JSONArray()),
                    )
                    is PhoneCommand.Messages -> {
                        beforeSeen.set(command.before)
                        PhoneResult(
                            command.id,
                            "ok",
                            "messages",
                            JSONObject().put("messages", org.json.JSONArray().put(JSONObject().put("id", "sms:s1"))),
                        )
                    }
                    is PhoneCommand.Search -> PhoneResult(
                        command.id,
                        "ok",
                        "search",
                        JSONObject().put("messages", org.json.JSONArray()),
                    )
                    else -> error("unexpected command $command")
                }
            }
        }

        try {
            withTimeout(5_000) { while (commandRequests.get() < 1) delay(10) }
            assertTrue("requests=${commandRequests.get()}, results=${results.size}", threeResults.await(5, TimeUnit.SECONDS))
            assertEquals("123:sms:s1", beforeSeen.get())
            val byId = results.associateBy { JSONObject(it).getString("id") }
            assertEquals(
                JSONObject().put("conversations", org.json.JSONArray()).toString(),
                JSONObject(requireNotNull(byId["c"])).getJSONObject("payload").toString(),
            )
            assertEquals(
                JSONObject().put("messages", org.json.JSONArray().put(JSONObject().put("id", "sms:s1"))).toString(),
                JSONObject(requireNotNull(byId["m"])).getJSONObject("payload").toString(),
            )
            assertEquals(
                JSONObject().put("messages", org.json.JSONArray()).toString(),
                JSONObject(requireNotNull(byId["s"])).getJSONObject("payload").toString(),
            )
        } finally {
            stream.close()
            runner.cancel()
            runner.join()
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

    @Test
    fun parsesReadCommandsWithOptionalBefore() {
        val expires = Instant.parse("2026-09-10T00:00:00Z")
        assertEquals(
            PhoneCommand.Conversations("c", 20, expires),
            PhoneCommand.parse(JSONObject("{\"kind\":\"conversations\",\"id\":\"c\",\"limit\":20,\"expires_at\":\"2026-09-10T00:00:00Z\"}")),
        )
        assertEquals(
            PhoneCommand.Messages("m", "sms:42", 50, "123:sms:s1", expires),
            PhoneCommand.parse(JSONObject("{\"kind\":\"messages\",\"id\":\"m\",\"conversation\":\"sms:42\",\"limit\":50,\"before\":\"123:sms:s1\",\"expires_at\":\"2026-09-10T00:00:00Z\"}")),
        )
        assertEquals(
            PhoneCommand.Search("s", "hello", 30, null, expires),
            PhoneCommand.parse(JSONObject("{\"kind\":\"search\",\"id\":\"s\",\"query\":\"hello\",\"limit\":30,\"expires_at\":\"2026-09-10T00:00:00Z\"}")),
        )
    }

    @Test
    fun resultPayloadIsAnObjectAndIsOmittedWhenAbsent() {
        val payload = JSONObject().put("messages", org.json.JSONArray().put(JSONObject().put("body", "a")))
        assertEquals(payload.toString(), PhoneResult("x", "ok", "done", payload).toJson().getJSONObject("payload").toString())
        assertTrue(!PhoneResult("x", "ok", "done").toJson().has("payload"))
    }

    @Test
    fun defaultBackoffUsesCappedExponentialSchedule() {
        assertEquals(
            listOf(1_000L, 2_000L, 4_000L, 8_000L, 16_000L, 32_000L, 60_000L, 60_000L),
            (0..7).map(::defaultBackoffMillis),
        )
    }
}

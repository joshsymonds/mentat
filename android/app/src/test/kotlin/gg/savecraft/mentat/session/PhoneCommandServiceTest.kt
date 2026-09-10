package gg.savecraft.mentat.session

import android.app.Application
import android.content.ContentProvider
import android.content.ContentValues
import android.database.Cursor
import android.database.MatrixCursor
import android.net.Uri
import android.app.NotificationManager
import android.content.ComponentName
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Looper
import androidx.test.core.app.ApplicationProvider
import gg.savecraft.mentat.core.Attachment
import gg.savecraft.mentat.core.Boundary
import gg.savecraft.mentat.core.Clock
import gg.savecraft.mentat.core.ConversationRow
import gg.savecraft.mentat.core.MessageRow
import gg.savecraft.mentat.core.MessageStore
import gg.savecraft.mentat.core.CommandExecutor
import gg.savecraft.mentat.core.CommandStream
import gg.savecraft.mentat.core.ContactMatch
import gg.savecraft.mentat.core.ContactResolver
import gg.savecraft.mentat.core.IntentLauncher
import gg.savecraft.mentat.core.PhoneCommand
import gg.savecraft.mentat.core.PhoneResult
import gg.savecraft.mentat.core.SmsSendOutcome
import gg.savecraft.mentat.core.SmsSender
import java.io.File
import kotlinx.coroutines.runBlocking
import java.time.Instant
import javax.xml.parsers.DocumentBuilderFactory
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows
import org.robolectric.shadows.ShadowContentResolver
import org.robolectric.annotation.Config

@Config(sdk = [35])
@RunWith(RobolectricTestRunner::class)
class PhoneCommandServiceTest {
    @Test
    fun startsForegroundOnPhoneBridgeChannelAndStopsStreamOnDestroy() {
        val stream = FakeCommandStream()
        FakePhoneCommandService.stream = stream
        FakePhoneCommandService.executor = testExecutor(RecordingIntentLauncher())
        val service = Robolectric.buildService(FakePhoneCommandService::class.java).create().get()

        service.onStartCommand(Intent(), 0, 1)

        val manager = service.getSystemService(NotificationManager::class.java)
        assertTrue(manager.notificationChannels.any { it.id == "phone-bridge" })
        val shadow = Shadows.shadowOf(service)
        assertTrue(shadow.isLastForegroundNotificationAttached)
        assertEquals(2, shadow.lastForegroundNotificationId)
        assertEquals("phone-bridge", shadow.lastForegroundNotification.channelId)
        service.onDestroy()
        assertTrue(stream.closed)
    }

    @Test
    fun serviceExecutesOpenCommandAndPostsResult() {
        val stream = FakeCommandStream()
        val launcher = RecordingIntentLauncher()
        FakePhoneCommandService.stream = stream
        FakePhoneCommandService.executor = CommandExecutor(
            smsSender = AllowingSmsSender(),
            contactResolver = EmptyContactResolver(),
            intentLauncher = launcher,
            messageStore = EmptyMessageStore(),
            clock = FixedClock(Instant.parse("2026-09-09T00:00:00Z")),
        )
        val service = Robolectric.buildService(FakePhoneCommandService::class.java).create().get()

        service.onStartCommand(Intent(), 0, 1)
        Shadows.shadowOf(Looper.getMainLooper()).idle()

        assertTrue(stream.handlerInvoked)
        assertEquals(listOf("https://example.test"), launcher.launchedUris)
        assertEquals(PhoneResult("open-id", "ok", "launched"), stream.result)
    }

    @Test
    fun productionExecutorUsesAndroidMessageStore() = runBlocking {
        val application = ApplicationProvider.getApplicationContext<Application>()
        Shadows.shadowOf(application).grantPermissions(android.Manifest.permission.READ_SMS)
        val provider = RecordingMmsSmsProvider()
        ShadowContentResolver.registerProviderInternal("mms-sms", provider)
        val service = Robolectric.buildService(ProductionPhoneCommandService::class.java).create().get()

        service.productionExecutor().execute(
            PhoneCommand.Conversations(
                id = "c",
                limit = 5,
                expiresAt = Instant.parse("2999-01-01T00:00:00Z"),
            ),
        )

        assertEquals(listOf("content://mms-sms/conversations?simple=true"), provider.queries)
    }

    @Test
    fun manifestDeclaresSpecialUseForegroundService() {
        val application = ApplicationProvider.getApplicationContext<Application>()
        val info = application.packageManager.getServiceInfo(
            android.content.ComponentName(application, PhoneCommandService::class.java),
            0,
        )
        assertEquals(ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE, info.foregroundServiceType)
    }

    @Test
    fun manifestDeclaresSpecialUseForegroundServiceSubtype() {
        val application = ApplicationProvider.getApplicationContext<Application>()
        val propertyValue = try {
            application.packageManager.getProperty(
                SPECIAL_USE_PROPERTY,
                ComponentName(application, PhoneCommandService::class.java),
            ).string
        } catch (_: Exception) {
            manifestSpecialUsePropertyValue()
        }
        assertEquals(SPECIAL_USE_PROPERTY_VALUE, propertyValue)
    }

    class FakeCommandStream : CommandStream {
        var closed = false
        var handlerInvoked = false
        var result: PhoneResult? = null

        override suspend fun run(handler: suspend (PhoneCommand) -> PhoneResult) {
            handlerInvoked = true
            result = handler(
                PhoneCommand.Open(
                    id = "open-id",
                    uri = "https://example.test",
                    expiresAt = Instant.parse("2026-09-09T00:01:00Z"),
                ),
            )
        }

        override fun close() {
            closed = true
        }
    }

    class FakePhoneCommandService : PhoneCommandService() {
        override fun commandStream(): CommandStream = stream
        override fun commandExecutor(): CommandExecutor = executor

        companion object {
            lateinit var stream: FakeCommandStream
            lateinit var executor: CommandExecutor
        }
    }

    class ProductionPhoneCommandService : PhoneCommandService() {
        override fun commandStream(): CommandStream = FakeCommandStream()

        fun productionExecutor(): CommandExecutor = super.commandExecutor()
    }

    private class RecordingMmsSmsProvider : ContentProvider() {
        val queries = mutableListOf<String>()

        override fun onCreate(): Boolean = true

        override fun query(
            uri: Uri,
            projection: Array<out String>?,
            selection: String?,
            selectionArgs: Array<out String>?,
            sortOrder: String?,
        ): Cursor {
            queries += uri.toString()
            return MatrixCursor(arrayOf("_id", "recipient_ids", "date"))
        }

        override fun getType(uri: Uri): String? = null
        override fun insert(uri: Uri, values: ContentValues?): Uri? = null
        override fun delete(uri: Uri, selection: String?, selectionArgs: Array<out String>?): Int = 0
        override fun update(uri: Uri, values: ContentValues?, selection: String?, selectionArgs: Array<out String>?): Int = 0
    }

    private class EmptyMessageStore : MessageStore {
        override val hasPermission: Boolean = true
        override fun conversations(limit: Int): List<ConversationRow> = emptyList()
        override fun messages(threadId: Long, limit: Int, before: Boundary?): List<MessageRow> = emptyList()
        override fun search(query: String, limit: Int, before: Boundary?): List<MessageRow> = emptyList()
        override fun directThreadsFor(numbers: List<String>): List<Pair<Long, gg.savecraft.mentat.core.Participant>> = emptyList()
    }

    private class AllowingSmsSender : SmsSender {
        override val hasPermission: Boolean = true
        override suspend fun send(number: String, body: String, deadline: Instant): SmsSendOutcome =
            SmsSendOutcome.Sent
    }

    private class EmptyContactResolver : ContactResolver {
        override val hasPermission: Boolean = true
        override fun resolve(name: String): List<ContactMatch> = emptyList()
    }

    private class RecordingIntentLauncher : IntentLauncher {
        val launchedUris = mutableListOf<String>()
        override fun canDrawOverlays(): Boolean = true
        override fun launch(uri: String) {
            launchedUris += uri
        }
    }

    private class FixedClock(private val instant: Instant) : Clock {
        override fun now(): Instant = instant
    }

    private fun testExecutor(launcher: IntentLauncher): CommandExecutor = CommandExecutor(
        smsSender = AllowingSmsSender(),
        contactResolver = EmptyContactResolver(),
        intentLauncher = launcher,
        messageStore = EmptyMessageStore(),
        clock = FixedClock(Instant.parse("2026-09-09T00:00:00Z")),
    )

    private fun manifestSpecialUsePropertyValue(): String {
        val manifest = File("src/main/AndroidManifest.xml")
        assertTrue(manifest.exists())
        val document = DocumentBuilderFactory.newInstance().newDocumentBuilder().parse(manifest)
        val services = document.getElementsByTagName("service")
        val service = (0 until services.length)
            .map { services.item(it) }
            .first { node ->
                node.attributes.getNamedItem("android:name")?.nodeValue ==
                    ".session.PhoneCommandService"
            }
        val properties = service.childNodes
        val property = (0 until properties.length)
            .map { properties.item(it) }
            .filter { it.nodeName == "property" }
            .first { node ->
                node.attributes.getNamedItem("android:name")?.nodeValue == SPECIAL_USE_PROPERTY
            }
        return requireNotNull(property.attributes.getNamedItem("android:value")?.nodeValue)
    }

    private companion object {
        const val SPECIAL_USE_PROPERTY = "android.app.PROPERTY_SPECIAL_USE_FGS_SUBTYPE"
        const val SPECIAL_USE_PROPERTY_VALUE =
            "persistent command channel to the owner's mentat daemon on the tailnet"
    }
}

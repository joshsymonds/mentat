package gg.savecraft.mentat.session

import android.Manifest
import java.io.ByteArrayInputStream
import android.app.Application
import android.content.ContentProvider
import android.content.ContentValues
import android.database.Cursor
import android.database.MatrixCursor
import android.net.Uri
import android.provider.ContactsContract
import androidx.test.core.app.ApplicationProvider
import gg.savecraft.mentat.core.Boundary
import gg.savecraft.mentat.core.Participant
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows
import org.robolectric.annotation.Config
import org.robolectric.shadows.ShadowContentResolver

@Config(sdk = [35])
@RunWith(RobolectricTestRunner::class)
class AndroidMessageStoreTest {
    private lateinit var application: Application
    private lateinit var provider: FakeTelephonyProvider

    @Before
    fun setUp() {
        application = ApplicationProvider.getApplicationContext()
        Shadows.shadowOf(application).grantPermissions(Manifest.permission.READ_SMS, Manifest.permission.READ_CONTACTS)
        provider = FakeTelephonyProvider()
        ShadowContentResolver.registerProviderInternal("sms", provider)
        ShadowContentResolver.registerProviderInternal("mms", provider)
        ShadowContentResolver.registerProviderInternal("mms-sms", provider)
        ShadowContentResolver.registerProviderInternal(ContactsContract.AUTHORITY, provider)
    }

    @Test
    fun conversationsCountReadableRowsUnreadRowsAndSkipUnreadableThreads() {
        val rows = AndroidMessageStore(application).conversations(20)
        val thread = rows.single { it.threadId == 42L }
        assertEquals(4, thread.messageCount)
        assertEquals(2, thread.unreadCount)
        assertEquals("sms:42", "sms:${thread.threadId}")
        assertEquals("Mum", thread.participants.single().name)
        assertTrue(rows.none { it.threadId == 99L })
    }

    @Test
    fun mmsTextPartsAttachmentsAndGroupSenderAreMapped() {
        val rows = AndroidMessageStore(application).messages(42L, 20, null)
        val mms = rows.first { it.id == "m10" }
        assertEquals("hello from mms", mms.body)
        assertEquals("in", mms.direction)
        assertEquals(1, mms.attachments.size)
        assertEquals("image/jpeg", mms.attachments.single().contentType)
        assertEquals("photo.jpg", mms.attachments.single().name)
        assertEquals("+15551212121", mms.from?.number)
    }

    @Test
    fun mmsSearchReadsTextColumnsAndStreamBackedTextParts() {
        provider.includeStreamBackedMms = true
        Shadows.shadowOf(application.contentResolver).registerInputStreamSupplier(
            Uri.parse("content://mms/part/104"),
            { ByteArrayInputStream("stream says Needle here".toByteArray()) },
        )
        val store = AndroidMessageStore(application)

        val streamMatch = store.search("needle", 10, null)
        assertEquals(listOf("m11"), streamMatch.map { it.id })
        assertEquals("stream says Needle here", streamMatch.single().body)

        val textColumnMatch = store.search("hello", 10, null)
        assertEquals(listOf("m10"), textColumnMatch.map { it.id })
    }

    @Test
    fun phoneLookupIsCachedPerCallAndDirectNumberMatchesCanonicalAddress() {
        provider.phoneLookupQueries = 0
        AndroidMessageStore(application).conversations(20)
        assertEquals(1, provider.phoneLookupQueries)

        provider.inserts = 0
        provider.threadIdQueries = 0
        val threads = AndroidMessageStore(application).directThreadsFor(listOf("(555) 121-2121"))
        assertEquals(listOf(42L to Participant("Mum", "+15551212121")), threads)
        assertEquals(0, provider.inserts)
        assertEquals(0, provider.threadIdQueries)

        provider.inserts = 0
        provider.threadIdQueries = 0
        assertTrue(AndroidMessageStore(application).directThreadsFor(listOf("+15550009999")).isEmpty())
        assertEquals(0, provider.inserts)
        assertEquals(0, provider.threadIdQueries)
    }

    @Test
    fun cursorBoundaryExcludesEqualTimestampRowsByMessageId() {
        provider.useCursorRows = true
        val store = AndroidMessageStore(application)

        assertEquals(
            listOf("s1", "s4"),
            store.messages(42L, 10, Boundary(1_000, "s2")).map { it.id },
        )
        assertEquals(
            listOf("s4"),
            store.messages(42L, 10, Boundary(1_000, null)).map { it.id },
        )
    }

    private class FakeTelephonyProvider : ContentProvider() {
        var phoneLookupQueries = 0
        var inserts = 0
        var threadIdQueries = 0
        var useCursorRows = false
        var includeStreamBackedMms = false

        override fun onCreate(): Boolean = true

        override fun query(
            uri: Uri,
            projection: Array<out String>?,
            selection: String?,
            selectionArgs: Array<out String>?,
            sortOrder: String?,
        ): Cursor {
            if (uri.toString().contains("threadID")) threadIdQueries += 1
            if (uri.authority == ContactsContract.AUTHORITY) {
                phoneLookupQueries += 1
                return MatrixCursor(arrayOf(ContactsContract.PhoneLookup.DISPLAY_NAME)).apply { addRow(arrayOf("Mum")) }
            }
            return when (uri.authority) {
                "mms-sms" -> when {
                    uri.path?.contains("canonical-addresses") == true -> MatrixCursor(arrayOf("_id", "address")).apply {
                        if (selectionArgs?.singleOrNull() == "2") addRow(arrayOf(2L, "+15550000000"))
                        else addRow(arrayOf(1L, "+15551212121"))
                    }
                    else -> MatrixCursor(arrayOf("_id", "recipient_ids", "date")).apply {
                        addRow(arrayOf(42L, "1", 4_000L))
                        addRow(arrayOf(99L, "2", 5_000L))
                    }
                }
                "sms" -> if (useCursorRows) {
                    MatrixCursor(arrayOf("_id", "thread_id", "address", "body", "date", "type", "read")).apply {
                        addRow(arrayOf(1L, 42L, "+15551212121", "s1", 1_000L, 1, 1))
                        addRow(arrayOf(2L, 42L, "+15551212121", "s2", 1_000L, 1, 1))
                        addRow(arrayOf(3L, 42L, "+15551212121", "s3", 1_000L, 1, 1))
                        addRow(arrayOf(4L, 42L, "+15551212121", "s4", 999L, 1, 1))
                    }
                } else MatrixCursor(arrayOf("_id", "thread_id", "address", "body", "date", "type", "read")).apply {
                    addRow(arrayOf(1L, 42L, "+15551212121", "sms in", 1_000L, 1, 0))
                    addRow(arrayOf(2L, 42L, "+15551212121", "sms out", 4_000L, 2, 1))
                    addRow(arrayOf(3L, 42L, "+15551212121", "draft", 6_000L, 3, 0))
                    addRow(arrayOf(4L, 99L, "+15551212121", "other", 7_000L, 3, 0))
                }
                "mms" -> if (useCursorRows) {
                    MatrixCursor(arrayOf("_id", "thread_id", "date", "msg_box", "read"))
                } else if (uri.path?.startsWith("/part") == true) {
                    MatrixCursor(arrayOf("_id", "mid", "ct", "text", "name", "cl")).apply {
                        val parts = mutableListOf(
                            arrayOf<Any?>(101L, 10L, "text/plain", "hello from mms", null, null),
                            arrayOf<Any?>(102L, 10L, "application/smil", "<smil>", null, null),
                            arrayOf<Any?>(103L, 10L, "image/jpeg", null, "photo.jpg", "fallback.jpg"),
                        )
                        if (includeStreamBackedMms) {
                            parts += arrayOf(104L, 11L, "text/plain", null, null, null)
                        }
                        parts.filter { row ->
                            if (selection?.contains("text LIKE") != true) return@filter true
                            val contentType = row[2] as String
                            val text = row[3] as String?
                            val wantedType = selectionArgs?.getOrNull(0)
                            val pattern = selectionArgs?.getOrNull(1).orEmpty()
                                .removePrefix("%").removeSuffix("%")
                            contentType == wantedType && (text?.contains(pattern, ignoreCase = true) == true ||
                                (text == null && selection.contains("text IS NULL")))
                        }.forEach(::addRow)
                    }
                } else if (uri.path?.matches(Regex("/\\d+/addr")) == true) {
                    MatrixCursor(arrayOf("address", "type")).apply { addRow(arrayOf("+15551212121", 137)) }
                } else {
                    MatrixCursor(arrayOf("_id", "thread_id", "date", "msg_box", "read")).apply {
                        addRow(arrayOf(10L, 42L, 2L, 1, 0))
                        addRow(arrayOf(11L, 42L, 3L, 2, 1))
                        addRow(arrayOf(12L, 99L, 5L, 3, 0))
                    }
                }
                else -> MatrixCursor(arrayOf("_id"))
            }
        }

        override fun getType(uri: Uri): String? = null
        override fun insert(uri: Uri, values: ContentValues?): Uri? { inserts += 1; return null }
        override fun delete(uri: Uri, selection: String?, selectionArgs: Array<out String>?): Int = 0
        override fun update(uri: Uri, values: ContentValues?, selection: String?, selectionArgs: Array<out String>?): Int = 0
    }
}

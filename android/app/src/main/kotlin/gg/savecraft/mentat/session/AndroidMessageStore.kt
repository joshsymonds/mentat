package gg.savecraft.mentat.session

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.database.Cursor
import android.net.Uri
import android.provider.ContactsContract
import android.provider.Telephony
import gg.savecraft.mentat.core.Attachment
import gg.savecraft.mentat.core.Boundary
import gg.savecraft.mentat.core.ConversationRow
import gg.savecraft.mentat.core.MessageRow
import gg.savecraft.mentat.core.MessageStore
import gg.savecraft.mentat.core.Participant

class AndroidMessageStore(
    private val context: Context,
) : MessageStore {
    private val resolver = context.contentResolver

    override val hasPermission: Boolean
        get() = context.checkSelfPermission(Manifest.permission.READ_SMS) == PackageManager.PERMISSION_GRANTED

    override fun conversations(limit: Int): List<ConversationRow> {
        if (!hasPermission || limit <= 0) return emptyList()
        val names = NameCache(context)
        val result = mutableListOf<ConversationRow>()
        query(
            CONVERSATIONS_URI,
            arrayOf(COLUMN_ID, COLUMN_RECIPIENT_IDS, COLUMN_DATE),
            null,
            null,
            "$COLUMN_DATE DESC",
        )?.use { cursor ->
            while (cursor.moveToNext() && result.size < limit) {
                val threadId = cursor.long(COLUMN_ID) ?: continue
                val recipientIds = parseRecipientIds(cursor.string(COLUMN_RECIPIENT_IDS))
                val sms = readSms(threadId)
                val mms = readMms(threadId)
                val readableCount = sms.size + mms.size
                if (readableCount == 0) continue
                val allMessages = sms.map { smsMessage(it, names) } + mms.map { mmsMessage(it, names) }
                val last = allMessages.maxWithOrNull(messageOrder) ?: continue
                result += ConversationRow(
                    threadId = threadId,
                    participants = participants(recipientIds, names),
                    lastDirection = last.direction,
                    lastBody = last.body,
                    lastAtMs = last.atMs,
                    messageCount = readableCount,
                    unreadCount = sms.count { it.type == SMS_INBOX && it.read == 0 } +
                        mms.count { it.read == 0 && it.msgBox == MMS_INBOX },
                )
            }
        }
        return result
    }

    override fun messages(threadId: Long, limit: Int, before: Boundary?): List<MessageRow> {
        if (!hasPermission || limit <= 0) return emptyList()
        val names = NameCache(context)
        val rows = readSms(threadId).map { smsMessage(it, names) } +
            readMms(threadId).map { mmsMessage(it, names) }
        return rows
            .filter { olderThan(it, before) }
            .sortedWith(messageOrder)
            .take(limit)
    }

    override fun search(query: String, limit: Int, before: Boundary?): List<MessageRow> {
        if (!hasPermission || limit <= 0) return emptyList()
        val names = NameCache(context)
        val escaped = "%${escapeLike(query)}%"
        val sms = readSmsForSearch(escaped)
            .filter { it.body?.contains(query, ignoreCase = true) == true }
            .map { smsMessage(it, names) }
        val matchingMmsIds = readMatchingMmsIds(escaped, query)
        val mms = readMms(null)
            .filter { it.id in matchingMmsIds }
            .map { mmsMessage(it, names) }
        return (sms + mms)
            .filter { olderThan(it, before) }
            .sortedWith(messageOrder)
            .take(limit)
    }

    override fun directThreadsFor(numbers: List<String>): List<Pair<Long, Participant>> {
        if (!hasPermission || numbers.isEmpty()) return emptyList()
        val wanted = numbers.map(::normalizeNumber).toSet()
        val names = NameCache(context)
        val result = mutableListOf<Pair<Long, Participant>>()
        query(CONVERSATIONS_URI, arrayOf(COLUMN_ID, COLUMN_RECIPIENT_IDS), null, null, null)?.use { cursor ->
            while (cursor.moveToNext()) {
                val threadId = cursor.long(COLUMN_ID) ?: continue
                val recipientIds = parseRecipientIds(cursor.string(COLUMN_RECIPIENT_IDS))
                if (recipientIds.size != 1) continue
                val participant = participantFor(recipientIds.single(), names) ?: continue
                if (normalizeNumber(participant.number) in wanted) {
                    result += threadId to participant
                }
            }
        }
        return result.distinctBy { it.first }
    }

    private fun readSms(threadId: Long?): List<SmsRecord> {
        val rows = mutableListOf<SmsRecord>()
        query(
            Telephony.Sms.CONTENT_URI,
            arrayOf(COLUMN_ID, COLUMN_THREAD_ID, COLUMN_ADDRESS, COLUMN_BODY, COLUMN_DATE, COLUMN_TYPE, COLUMN_READ),
            if (threadId == null) "$COLUMN_TYPE IN (1,2)" else "$COLUMN_THREAD_ID=? AND $COLUMN_TYPE IN (1,2)",
            threadId?.let { arrayOf(it.toString()) },
            "$COLUMN_DATE DESC",
        )?.use { cursor ->
            while (cursor.moveToNext()) {
                val row = SmsRecord(
                    id = cursor.long(COLUMN_ID) ?: continue,
                    threadId = cursor.long(COLUMN_THREAD_ID) ?: continue,
                    address = cursor.string(COLUMN_ADDRESS),
                    body = cursor.string(COLUMN_BODY),
                    atMs = cursor.long(COLUMN_DATE) ?: continue,
                    type = cursor.int(COLUMN_TYPE) ?: continue,
                    read = cursor.int(COLUMN_READ) ?: 1,
                )
                if (row.type in SMS_READABLE && (threadId == null || row.threadId == threadId)) rows += row
            }
        }
        return rows
    }

    private fun readSmsForSearch(pattern: String): List<SmsRecord> {
        val rows = mutableListOf<SmsRecord>()
        query(
            Telephony.Sms.CONTENT_URI,
            arrayOf(COLUMN_ID, COLUMN_THREAD_ID, COLUMN_ADDRESS, COLUMN_BODY, COLUMN_DATE, COLUMN_TYPE, COLUMN_READ),
            "$COLUMN_BODY LIKE ? ESCAPE '\\' AND $COLUMN_TYPE IN (1,2)",
            arrayOf(pattern),
            "$COLUMN_DATE DESC",
        )?.use { cursor ->
            while (cursor.moveToNext()) {
                val row = SmsRecord(
                    id = cursor.long(COLUMN_ID) ?: continue,
                    threadId = cursor.long(COLUMN_THREAD_ID) ?: continue,
                    address = cursor.string(COLUMN_ADDRESS),
                    body = cursor.string(COLUMN_BODY),
                    atMs = cursor.long(COLUMN_DATE) ?: continue,
                    type = cursor.int(COLUMN_TYPE) ?: continue,
                    read = cursor.int(COLUMN_READ) ?: 1,
                )
                if (row.type in SMS_READABLE) rows += row
            }
        }
        return rows
    }

    private fun readMms(threadId: Long?): List<MmsRecord> {
        val rows = mutableListOf<MmsRecord>()
        query(
            Telephony.Mms.CONTENT_URI,
            arrayOf(COLUMN_ID, COLUMN_THREAD_ID, COLUMN_DATE, COLUMN_MSG_BOX, COLUMN_READ),
            if (threadId == null) "$COLUMN_MSG_BOX IN (1,2)" else "$COLUMN_THREAD_ID=? AND $COLUMN_MSG_BOX IN (1,2)",
            threadId?.let { arrayOf(it.toString()) },
            "$COLUMN_DATE DESC",
        )?.use { cursor ->
            while (cursor.moveToNext()) {
                val row = MmsRecord(
                    id = cursor.long(COLUMN_ID) ?: continue,
                    threadId = cursor.long(COLUMN_THREAD_ID) ?: continue,
                    atMs = (cursor.long(COLUMN_DATE) ?: continue) * 1_000L,
                    msgBox = cursor.int(COLUMN_MSG_BOX) ?: continue,
                    read = cursor.int(COLUMN_READ) ?: 1,
                )
                if (row.msgBox in MMS_READABLE && (threadId == null || row.threadId == threadId)) rows += row
            }
        }
        return rows
    }

    private fun readMatchingMmsIds(pattern: String, queryText: String): Set<Long> {
        val ids = mutableSetOf<Long>()
        query(
            MMS_PART_URI,
            arrayOf(COLUMN_ID, COLUMN_MID, COLUMN_CONTENT_TYPE, COLUMN_TEXT, COLUMN_NAME, COLUMN_CONTENT_LOCATION),
            "$COLUMN_CONTENT_TYPE=? AND ($COLUMN_TEXT LIKE ? ESCAPE '\\' OR $COLUMN_TEXT IS NULL)",
            arrayOf(TEXT_PLAIN, pattern),
            null,
        )?.use { cursor ->
            while (cursor.moveToNext()) {
                val contentType = cursor.string(COLUMN_CONTENT_TYPE)
                val mid = cursor.long(COLUMN_MID)
                if (contentType != TEXT_PLAIN || mid == null) continue
                val text = partText(cursor, contentType)
                if (text?.contains(queryText, ignoreCase = true) == true) ids += mid
            }
        }
        return ids
    }

    private fun smsMessage(row: SmsRecord, names: NameCache): MessageRow = MessageRow(
        id = "s${row.id}",
        threadId = row.threadId,
        direction = if (row.type == SMS_INBOX) "in" else "out",
        from = if (row.type == SMS_INBOX && row.address != null) participant(row.address, names) else null,
        body = row.body.orEmpty(),
        atMs = row.atMs,
        attachments = emptyList(),
    )

    private fun mmsMessage(row: MmsRecord, names: NameCache): MessageRow {
        val parts = readParts(row.id)
        val sender = if (row.msgBox == MMS_INBOX) readMmsSender(row.id, names) else null
        return MessageRow(
            id = "m${row.id}",
            threadId = row.threadId,
            direction = if (row.msgBox == MMS_INBOX) "in" else "out",
            from = sender,
            body = parts.filter { it.contentType == TEXT_PLAIN }.joinToString("") { it.text.orEmpty() },
            atMs = row.atMs,
            attachments = parts.filter { it.contentType != TEXT_PLAIN && it.contentType != APPLICATION_SMIL }
                .map { Attachment(it.contentType, it.name ?: it.contentLocation) },
        )
    }

    private fun partText(cursor: Cursor, contentType: String): String? {
        val text = cursor.string(COLUMN_TEXT)
        if (text != null || contentType != TEXT_PLAIN) return text
        val id = cursor.long(COLUMN_ID) ?: return null
        return runCatching {
            resolver.openInputStream(Uri.withAppendedPath(MMS_PART_URI, id.toString()))
                ?.bufferedReader()
                ?.use { it.readText() }
        }.getOrNull()
    }

    private fun readParts(messageId: Long): List<MmsPart> {
        val parts = mutableListOf<MmsPart>()
        query(
            MMS_PART_URI,
            arrayOf(COLUMN_ID, COLUMN_MID, COLUMN_CONTENT_TYPE, COLUMN_TEXT, COLUMN_NAME, COLUMN_CONTENT_LOCATION),
            "$COLUMN_MID=?",
            arrayOf(messageId.toString()),
            "$COLUMN_ID ASC",
        )?.use { cursor ->
            while (cursor.moveToNext()) {
                val id = cursor.long(COLUMN_ID) ?: continue
                val mid = cursor.long(COLUMN_MID) ?: continue
                if (mid != messageId) continue
                val contentType = cursor.string(COLUMN_CONTENT_TYPE) ?: continue
                val text = partText(cursor, contentType)
                parts += MmsPart(
                    contentType = contentType,
                    text = text,
                    name = cursor.string(COLUMN_NAME),
                    contentLocation = cursor.string(COLUMN_CONTENT_LOCATION),
                )
            }
        }
        return parts
    }

    private fun readMmsSender(messageId: Long, names: NameCache): Participant? {
        val uri = Uri.parse("content://mms/$messageId/addr")
        return query(uri, arrayOf(COLUMN_ADDRESS, COLUMN_TYPE), "$COLUMN_TYPE=?", arrayOf(MMS_FROM_TYPE.toString()), null)?.use { cursor ->
            while (cursor.moveToNext()) {
                val type = cursor.int(COLUMN_TYPE)
                val address = cursor.string(COLUMN_ADDRESS)
                if (type == MMS_FROM_TYPE && !address.isNullOrBlank()) return@use participant(address, names)
            }
            null
        }
    }

    private fun parseRecipientIds(value: String?): List<Long> =
        value.orEmpty().split(Regex("[\\s,]+"))
            .mapNotNull { it.toLongOrNull() }

    private fun participants(ids: List<Long>, names: NameCache): List<Participant> =
        ids.mapNotNull { participantFor(it, names) }

    private fun participantFor(id: Long, names: NameCache): Participant? {
        return query(
            CANONICAL_ADDRESSES_URI,
            arrayOf(COLUMN_ID, COLUMN_ADDRESS),
            "$COLUMN_ID=?",
            arrayOf(id.toString()),
            null,
        )?.use { cursor ->
            while (cursor.moveToNext()) {
                val rowId = cursor.long(COLUMN_ID)
                val address = cursor.string(COLUMN_ADDRESS)
                if (rowId == id && !address.isNullOrBlank()) return@use participant(address, names)
            }
            null
        }
    }

    private fun participant(number: String, names: NameCache): Participant =
        Participant(names.lookup(number), number)

    private fun query(
        uri: Uri,
        projection: Array<String>,
        selection: String?,
        selectionArgs: Array<String>?,
        sortOrder: String?,
    ): Cursor? = if (hasPermission) resolver.query(uri, projection, selection, selectionArgs, sortOrder) else null

    private fun olderThan(row: MessageRow, before: Boundary?): Boolean {
        if (before == null) return true
        if (row.atMs < before.timeMs) return true
        if (row.atMs > before.timeMs) return false
        return before.id != null && row.id < before.id
    }

    private fun normalizeNumber(number: String): String {
        val digits = number.filter(Char::isDigit)
        return if (digits.length == 11 && digits.startsWith('1')) digits.drop(1) else digits
    }

    private fun escapeLike(value: String): String = buildString {
        value.forEach { char ->
            if (char == '\\' || char == '%' || char == '_') append('\\')
            append(char)
        }
    }

    private class NameCache(private val context: Context) {
        private val names = mutableMapOf<String, String?>()

        fun lookup(number: String): String? = names.getOrPut(number) {
            val uri = Uri.withAppendedPath(
                ContactsContract.PhoneLookup.CONTENT_FILTER_URI,
                Uri.encode(number),
            )
            runCatching {
                context.contentResolver.query(uri, arrayOf(ContactsContract.PhoneLookup.DISPLAY_NAME), null, null, null)
                    ?.use { cursor ->
                        if (cursor.moveToFirst()) cursor.string(ContactsContract.PhoneLookup.DISPLAY_NAME) else null
                    }
            }.getOrNull()
        }
    }

    private data class SmsRecord(
        val id: Long,
        val threadId: Long,
        val address: String?,
        val body: String?,
        val atMs: Long,
        val type: Int,
        val read: Int,
    )

    private data class MmsRecord(
        val id: Long,
        val threadId: Long,
        val atMs: Long,
        val msgBox: Int,
        val read: Int,
    )

    private data class MmsPart(
        val contentType: String,
        val text: String?,
        val name: String?,
        val contentLocation: String?,
    )

    private val messageOrder = compareByDescending<MessageRow> { it.atMs }.thenByDescending { it.id }

    private companion object {
        const val COLUMN_ID = "_id"
        const val COLUMN_THREAD_ID = "thread_id"
        const val COLUMN_RECIPIENT_IDS = "recipient_ids"
        const val COLUMN_ADDRESS = "address"
        const val COLUMN_BODY = "body"
        const val COLUMN_DATE = "date"
        const val COLUMN_TYPE = "type"
        const val COLUMN_READ = "read"
        const val COLUMN_MSG_BOX = "msg_box"
        const val COLUMN_MID = "mid"
        const val COLUMN_CONTENT_TYPE = "ct"
        const val COLUMN_TEXT = "text"
        const val COLUMN_NAME = "name"
        const val COLUMN_CONTENT_LOCATION = "cl"
        const val SMS_INBOX = 1
        const val MMS_INBOX = 1
        const val MMS_FROM_TYPE = 137
        const val TEXT_PLAIN = "text/plain"
        const val APPLICATION_SMIL = "application/smil"
        val SMS_READABLE = setOf(1, 2)
        val MMS_READABLE = setOf(1, 2)
        val CONVERSATIONS_URI = Uri.parse("content://mms-sms/conversations?simple=true")
        val CANONICAL_ADDRESSES_URI = Uri.parse("content://mms-sms/canonical-addresses")
        val MMS_PART_URI = Uri.parse("content://mms/part")
    }

}

private fun Cursor.string(column: String): String? {
    val index = getColumnIndex(column)
    return if (index >= 0 && !isNull(index)) getString(index) else null
}

private fun Cursor.long(column: String): Long? {
    val index = getColumnIndex(column)
    return if (index >= 0 && !isNull(index)) getLong(index) else null
}

private fun Cursor.int(column: String): Int? {
    val index = getColumnIndex(column)
    return if (index >= 0 && !isNull(index)) getInt(index) else null
}

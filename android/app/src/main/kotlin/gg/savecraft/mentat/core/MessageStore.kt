package gg.savecraft.mentat.core

data class Boundary(val timeMs: Long, val id: String?)

data class Participant(val name: String?, val number: String)

data class Attachment(val contentType: String, val name: String?)

data class ConversationRow(
    val threadId: Long,
    val participants: List<Participant>,
    val lastDirection: String,
    val lastBody: String,
    val lastAtMs: Long,
    val messageCount: Int,
    val unreadCount: Int,
)

data class MessageRow(
    val id: String,
    val threadId: Long,
    val direction: String,
    val from: Participant?,
    val body: String,
    val atMs: Long,
    val attachments: List<Attachment>,
)

interface MessageStore {
    val hasPermission: Boolean
    fun conversations(limit: Int): List<ConversationRow>
    fun messages(threadId: Long, limit: Int, before: Boundary?): List<MessageRow>
    fun search(query: String, limit: Int, before: Boundary?): List<MessageRow>
    fun directThreadsFor(numbers: List<String>): List<Pair<Long, Participant>>
}

package gg.savecraft.mentat.core

import android.content.ActivityNotFoundException
import java.time.Instant
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

sealed class SmsSendOutcome {
    data object Sent : SmsSendOutcome()
    data class Failed(val resultCode: Int) : SmsSendOutcome()
    data object Unknown : SmsSendOutcome()
    data object Expired : SmsSendOutcome()
}

interface SmsSender {
    val hasPermission: Boolean
    suspend fun send(number: String, body: String, deadline: Instant): SmsSendOutcome
}

data class ContactMatch(val displayName: String, val number: String)

interface ContactResolver {
    val hasPermission: Boolean
    fun resolve(name: String): List<ContactMatch>
}

interface IntentLauncher {
    fun canDrawOverlays(): Boolean
    fun launch(uri: String)
}

interface Clock {
    fun now(): Instant
}

object SystemClock : Clock {
    override fun now(): Instant = Instant.now()
}

class CommandExecutor(
    private val smsSender: SmsSender,
    private val contactResolver: ContactResolver,
    private val intentLauncher: IntentLauncher,
    private val clock: Clock = SystemClock,
) {
    suspend fun execute(command: PhoneCommand): PhoneResult {
        return when (command) {
            is PhoneCommand.Sms -> executeSms(command)
            is PhoneCommand.Open -> executeOpen(command)
            PhoneCommand.Ping -> PhoneResult("", "ok", "ping")
        }
    }

    private suspend fun executeSms(command: PhoneCommand.Sms): PhoneResult {
        if (expired(command.expiresAt)) {
            return PhoneResult(command.id, "error", "expired")
        }

        val number = if (isDialableNumber(command.to)) {
            if (!smsSender.hasPermission) {
                return PhoneResult(command.id, "error", "permission denied")
            }
            command.to
        } else {
            if (!contactResolver.hasPermission || !smsSender.hasPermission) {
                return PhoneResult(command.id, "error", "permission denied")
            }
            val matches = contactResolver.resolve(command.to)
                .distinctBy { normalizeNumber(it.number) }
            when (matches.size) {
                0 -> return PhoneResult(command.id, "error", "not found")
                1 -> matches.single().number
                else -> {
                    val details = matches.joinToString(", ") { "${it.displayName} (${it.number})" }
                    return PhoneResult(command.id, "error", "ambiguous: $details")
                }
            }
        }

        return when (val outcome = try {
            withContext(Dispatchers.IO) {
                smsSender.send(number, command.body, command.expiresAt)
            }
        } catch (_: SecurityException) {
            return PhoneResult(command.id, "error", "permission denied")
        }) {
            SmsSendOutcome.Sent -> PhoneResult(command.id, "ok", "sent to $number")
            is SmsSendOutcome.Failed -> PhoneResult(command.id, "error", "send failed: ${outcome.resultCode}")
            SmsSendOutcome.Unknown -> PhoneResult(command.id, "error", "outcome unknown")
            SmsSendOutcome.Expired -> PhoneResult(command.id, "error", "expired")
        }
    }

    private fun executeOpen(command: PhoneCommand.Open): PhoneResult {
        if (expired(command.expiresAt)) {
            return PhoneResult(command.id, "error", "expired")
        }
        if (!intentLauncher.canDrawOverlays()) {
            return PhoneResult(command.id, "error", "overlay permission missing")
        }
        return try {
            intentLauncher.launch(command.uri)
            PhoneResult(command.id, "ok", "launched")
        } catch (_: ActivityNotFoundException) {
            PhoneResult(command.id, "error", "no handler")
        }
    }

    private fun expired(deadline: Instant): Boolean = !clock.now().isBefore(deadline)

    private fun isDialableNumber(value: String): Boolean =
        value.matches(Regex("^\\+?[0-9][0-9 .()\\-]{2,}$"))

    private fun normalizeNumber(value: String): String =
        value.filter { it.isDigit() }.let { digits -> if (value.trim().startsWith('+')) "+$digits" else digits }
}

package gg.savecraft.mentat.session

import android.Manifest
import android.app.Activity
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.PackageManager
import android.net.Uri
import android.os.PowerManager
import android.provider.ContactsContract
import android.provider.Settings
import android.telephony.SmsManager
import gg.savecraft.mentat.core.Clock
import gg.savecraft.mentat.core.ContactMatch
import gg.savecraft.mentat.core.ContactResolver
import gg.savecraft.mentat.core.IntentLauncher
import gg.savecraft.mentat.core.SmsSendOutcome
import gg.savecraft.mentat.core.SmsSender
import gg.savecraft.mentat.core.SystemClock
import java.time.Duration
import java.time.Instant
import java.util.UUID
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull

interface SmsGateway {
    fun divideMessage(body: String): ArrayList<String>
    fun send(number: String, parts: ArrayList<String>, sentIntents: ArrayList<PendingIntent>)
}

private class AndroidSmsGateway : SmsGateway {
    private val manager = SmsManager.getDefault()

    override fun divideMessage(body: String): ArrayList<String> = manager.divideMessage(body)

    override fun send(number: String, parts: ArrayList<String>, sentIntents: ArrayList<PendingIntent>) {
        manager.sendMultipartTextMessage(number, null, parts, sentIntents, null)
    }
}

class AndroidSmsSender(
    private val context: Context,
    private val clock: Clock = SystemClock,
    private val gateway: SmsGateway = AndroidSmsGateway(),
) : SmsSender {
    override val hasPermission: Boolean
        get() = context.checkSelfPermission(Manifest.permission.SEND_SMS) == PackageManager.PERMISSION_GRANTED

    override suspend fun send(number: String, body: String, deadline: Instant): SmsSendOutcome = withContext(Dispatchers.IO) {
        if (!hasPermission) {
            return@withContext SmsSendOutcome.Failed(SmsManager.RESULT_ERROR_GENERIC_FAILURE)
        }
        val parts = gateway.divideMessage(body)
        if (parts.isEmpty()) {
            return@withContext SmsSendOutcome.Sent
        }
        val action = "${context.packageName}.SMS_SENT.${UUID.randomUUID()}"
        val completed = CompletableDeferred<SmsSendOutcome>()
        val outcomes = arrayOfNulls<Boolean>(parts.size)
        val receiver = object : BroadcastReceiver() {
            override fun onReceive(receiverContext: Context, intent: Intent) {
                if (completed.isCompleted) {
                    return
                }
                val index = intent.getIntExtra(PART_INDEX, -1)
                if (index !in outcomes.indices || outcomes[index] != null) {
                    return
                }
                if (resultCode != Activity.RESULT_OK) {
                    completed.complete(SmsSendOutcome.Failed(resultCode))
                    return
                }
                outcomes[index] = true
                if (outcomes.all { it == true }) {
                    completed.complete(SmsSendOutcome.Sent)
                }
            }
        }
        val filter = IntentFilter(action)
        context.registerReceiver(receiver, filter, Context.RECEIVER_NOT_EXPORTED)
        try {
            val sentIntents = ArrayList<PendingIntent>(parts.size).apply {
                parts.forEachIndexed { index, _ ->
                    add(
                        PendingIntent.getBroadcast(
                            context,
                            index,
                            Intent(action).setPackage(context.packageName).putExtra(PART_INDEX, index),
                            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
                        ),
                    )
                }
            }
            if (!clock.now().isBefore(deadline)) {
                return@withContext SmsSendOutcome.Expired
            }
            try {
                gateway.send(number, parts, sentIntents)
            } catch (exception: SecurityException) {
                throw exception
            } catch (_: Exception) {
                return@withContext SmsSendOutcome.Failed(SmsManager.RESULT_ERROR_GENERIC_FAILURE)
            }
            val remainingMillis = Duration.between(clock.now(), deadline).toMillis()
            if (remainingMillis <= 0) {
                SmsSendOutcome.Unknown
            } else {
                withTimeoutOrNull(remainingMillis) { completed.await() } ?: SmsSendOutcome.Unknown
            }
        } finally {
            runCatching { context.unregisterReceiver(receiver) }
        }
    }

    private companion object {
        const val PART_INDEX = "part_index"
    }
}

class AndroidContactResolver(
    private val context: Context,
) : ContactResolver {
    override val hasPermission: Boolean
        get() = context.checkSelfPermission(Manifest.permission.READ_CONTACTS) == PackageManager.PERMISSION_GRANTED

    override fun resolve(name: String): List<ContactMatch> {
        if (!hasPermission) {
            return emptyList()
        }
        val uri = Uri.withAppendedPath(
            ContactsContract.CommonDataKinds.Phone.CONTENT_FILTER_URI,
            Uri.encode(name),
        )
        val projection = arrayOf(
            ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME,
            ContactsContract.CommonDataKinds.Phone.NUMBER,
        )
        return context.contentResolver.query(uri, projection, null, null, null)?.use { cursor ->
            val displayNameIndex = cursor.getColumnIndex(ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME)
            val numberIndex = cursor.getColumnIndex(ContactsContract.CommonDataKinds.Phone.NUMBER)
            buildList {
                while (cursor.moveToNext()) {
                    if (displayNameIndex >= 0 && numberIndex >= 0) {
                        add(ContactMatch(cursor.getString(displayNameIndex), cursor.getString(numberIndex)))
                    }
                }
            }
        } ?: emptyList()
    }
}

class AndroidIntentLauncher(
    private val context: Context,
) : IntentLauncher {
    override fun canDrawOverlays(): Boolean = Settings.canDrawOverlays(context)

    override fun launch(intent: Intent) {
        context.startActivity(intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
    }
}

fun isBatteryOptimizationIgnored(context: Context): Boolean =
    (context.getSystemService(Context.POWER_SERVICE) as? PowerManager)
        ?.isIgnoringBatteryOptimizations(context.packageName)
        ?: false

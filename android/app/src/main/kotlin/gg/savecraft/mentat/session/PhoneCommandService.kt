package gg.savecraft.mentat.session

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.util.Log
import gg.savecraft.mentat.R
import gg.savecraft.mentat.core.CommandExecutor
import gg.savecraft.mentat.core.CommandStream
import gg.savecraft.mentat.core.HttpCommandStream
import gg.savecraft.mentat.core.SystemClock
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

open class PhoneCommandService : Service() {
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private lateinit var stream: CommandStream
    private lateinit var executor: CommandExecutor
    private var started = false

    override fun onCreate() {
        super.onCreate()
        stream = commandStream()
        executor = commandExecutor()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (!started) {
            started = true
            try {
                startForegroundNotification()
            } catch (exception: Exception) {
                Log.e(TAG, "Unable to start phone command service", exception)
                stopSelf()
                return START_NOT_STICKY
            }
            serviceScope.launch {
                runCatching {
                    stream.run { command -> executor.execute(command) }
                }.onFailure { exception ->
                    if (exception !is kotlinx.coroutines.CancellationException) {
                        Log.e(TAG, "Phone command stream stopped", exception)
                    }
                }
            }
        }
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        try {
            stream.close()
        } finally {
            serviceScope.cancel()
            super.onDestroy()
        }
    }

    protected open fun commandStream(): CommandStream =
        HttpCommandStream(AppSettings(this).tokenEndpointUrl)

    protected open fun commandExecutor(): CommandExecutor =
        CommandExecutor(
            smsSender = AndroidSmsSender(this, SystemClock),
            contactResolver = AndroidContactResolver(this),
            intentLauncher = AndroidIntentLauncher(this),
            clock = SystemClock,
        )

    protected open fun startForegroundNotification() {
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(
            NotificationChannel(
                NOTIFICATION_CHANNEL_ID,
                getString(R.string.phone_notification_channel),
                NotificationManager.IMPORTANCE_LOW,
            ),
        )
        val notification: Notification = Notification.Builder(this, NOTIFICATION_CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setContentTitle(getString(R.string.phone_notification_title))
            .setContentText(getString(R.string.phone_notification_text))
            .setOngoing(true)
            .build()
        startForeground(NOTIFICATION_ID, notification)
    }

    private companion object {
        const val TAG = "MentatPhone"
        const val NOTIFICATION_CHANNEL_ID = "phone-bridge"
        const val NOTIFICATION_ID = 2
    }
}

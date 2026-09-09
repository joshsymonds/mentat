package gg.savecraft.mentat.phone

import android.content.ActivityNotFoundException
import android.content.Context
import android.content.Intent
import android.util.Log
import io.livekit.android.room.participant.LocalParticipant
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import io.livekit.android.rpc.RpcError

object PhoneRpcCodes {
    const val INVALID_COMMAND = 1600
    const val ACTIVITY_UNAVAILABLE = 1601
    const val LOCATION_UNAVAILABLE = 1602
    const val NOT_IN_FRONT = 1603
}

fun interface PhoneRpcRegistrar {
    fun registerRpcMethod(method: String, handler: suspend (String) -> String)
}

class PhoneRpcException(
    val code: Int,
    message: String,
    cause: Throwable? = null,
) : Exception(message, cause)

class PhoneBridge(
    private val location: PhoneLocation?,
    private val launcher: (Intent) -> Unit,
) {
    @Volatile
    var assistVisible: Boolean = false

    private var registered = false

    constructor(context: Context) : this(
        location = PhoneLocation(context),
        launcher = { intent -> context.applicationContext.startActivity(intent) },
    )

    fun handleCommand(payload: String): String {
        if (!assistVisible) {
            Log.i(TAG, "command refused: assist screen is not in front")
            throw PhoneRpcException(PhoneRpcCodes.NOT_IN_FRONT, "not in front")
        }
        val intent = PhoneCommands.intentFor(PhoneCommands.parse(payload))
        try {
            launcher(intent)
        } catch (exception: ActivityNotFoundException) {
            Log.w(TAG, "command refused: no activity for ${intent.action}", exception)
            throw PhoneRpcException(
                PhoneRpcCodes.ACTIVITY_UNAVAILABLE,
                "no activity to handle intent",
                exception,
            )
        }
        Log.i(TAG, "command launched action=${intent.action}")
        return "{\"ok\":true}"
    }

    fun handleLocation(): String = location?.get()
        ?: throw PhoneRpcException(PhoneRpcCodes.LOCATION_UNAVAILABLE, "location unavailable")

    fun register(localParticipant: LocalParticipant) {
        register(PhoneRpcRegistrar { method, handler ->
            localParticipant.registerRpcMethod(method) { invocation ->
                try {
                    handler(invocation.payload)
                } catch (exception: PhoneRpcException) {
                    throw exception.toRpcError()
                }
            }
        })
    }

    fun register(registrar: PhoneRpcRegistrar) {
        synchronized(this) {
            if (registered) {
                return
            }
            registered = true
        }
        registrar.registerRpcMethod(COMMAND_METHOD) { payload -> handleCommand(payload) }
        registrar.registerRpcMethod(LOCATION_METHOD) {
            withContext(Dispatchers.IO) { handleLocation() }
        }
    }

    private fun PhoneRpcException.toRpcError(): RpcError =
        RpcError(code, message ?: "phone RPC failed", "", this)

    private companion object {
        const val TAG = "MentatAssist"
        const val COMMAND_METHOD = "mentat.command"
        const val LOCATION_METHOD = "mentat.location"
    }
}

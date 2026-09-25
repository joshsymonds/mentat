package gg.savecraft.mentat.core

import org.json.JSONObject

/** What the phone knows as a call starts; mentatd hands it to the voice for the greeting. */
data class CallContext(
    val timeZone: String,
    val location: JSONObject?,
    val driving: Boolean,
) {
    fun toRequestBody(): String {
        val context = JSONObject()
            .put("time_zone", timeZone)
            .put("driving", driving)
        if (location != null) {
            context.put("location", location)
        }
        return JSONObject().put("context", context).toString()
    }
}

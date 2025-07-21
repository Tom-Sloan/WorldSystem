package com.tomscompany.websockettest

import android.content.Context
import android.content.SharedPreferences

object PreferenceManager {
    private const val PREF_NAME = "WebSocketTestPrefs"
    private const val KEY_PERMISSIONS_REQUESTED = "permissions_requested"

    private fun getSharedPreferences(context: Context): SharedPreferences {
        return context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)
    }

    fun setPermissionsRequested(context: Context, requested: Boolean) {
        getSharedPreferences(context).edit().putBoolean(KEY_PERMISSIONS_REQUESTED, requested).apply()
    }

    fun isPermissionsRequested(context: Context): Boolean {
        return getSharedPreferences(context).getBoolean(KEY_PERMISSIONS_REQUESTED, false)
    }
}

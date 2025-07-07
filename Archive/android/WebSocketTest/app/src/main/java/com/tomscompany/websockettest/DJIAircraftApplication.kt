package com.tomscompany.websockettest
import android.content.Context
import android.util.Log

/**
 * Class Description
 *
 * @author Hoker
 * @date 2022/3/2
 *
 * Copyright (c) 2022, DJI All Rights Reserved.
 */
class DJIAircraftApplication : DJIApplication() {

    override fun attachBaseContext(base: Context?) {
        super.attachBaseContext(base)

        try {
            com.secneo.sdk.Helper.install(this)
        } catch (e: Exception) {
            Log.e("DJIAircraftApplication", "Error installing Helper: ${e.message}", e)
        }
    }
}

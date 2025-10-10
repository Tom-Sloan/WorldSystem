package com.tomscompany.websockettest

import android.app.Application

/**
 * Class Description
 *
 * @author 
 * @date 
 *
 * Copyright 
 */
open class DJIApplication : Application() {

    override fun onCreate() {
        super.onCreate()
        // Keep the SDK initialization here
        MSDKManager.initMobileSDK(this)
    }

}

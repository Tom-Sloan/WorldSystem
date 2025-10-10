package com.tomscompany.websockettest

import android.Manifest
import android.app.AlertDialog
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.provider.Settings
import android.util.Log
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.widget.Button
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.navigation.findNavController
import androidx.navigation.ui.AppBarConfiguration
import androidx.navigation.ui.navigateUp
import com.google.android.material.snackbar.Snackbar
import com.tomscompany.websockettest.databinding.ActivityMainBinding
import dji.sdk.keyvalue.value.common.EmptyMsg
import dji.v5.common.callback.CommonCallbacks
import dji.v5.common.error.IDJIError
import android.os.Environment
import android.os.Build
import androidx.appcompat.widget.Toolbar
import androidx.appcompat.app.ActionBar
import android.view.animation.AnimationUtils
import dji.v5.manager.KeyManager
import dji.sdk.keyvalue.key.CameraKey
import dji.sdk.keyvalue.key.KeyTools
import dji.sdk.keyvalue.value.camera.CameraMode
import dji.sdk.keyvalue.value.camera.VideoFrameRate
import dji.sdk.keyvalue.value.camera.VideoResolution
import dji.sdk.keyvalue.value.camera.VideoResolutionFrameRate
import dji.sdk.keyvalue.value.common.ComponentIndexType


class MainActivity : AppCompatActivity() {

    private val TAG = "MainActivity"
    private val REQUEST_PERMISSION_CODE = 12345
    private val REQUEST_MANAGE_EXTERNAL_STORAGE_PERMISSION = 12346

    private lateinit var appBarConfiguration: AppBarConfiguration
    private lateinit var binding: ActivityMainBinding

    private lateinit var btnTakeOff: Button
    private lateinit var btnLand: Button
    private lateinit var basicAircraftControlVM: BasicAircraftControlVM
    private val virtualStickVM: VirtualStickVM by viewModels()
    private val h264VideoStreamVM: H264VideoStreamVM by viewModels()

    private lateinit var connectButton: Button

    private lateinit var websocketMessageHandler: WebsocketMessageHandler

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Force landscape orientation
        requestedOrientation = android.content.pm.ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE

        // Data Binding Setup
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val navController = findNavController(R.id.nav_host_fragment_content_main)
        appBarConfiguration = AppBarConfiguration(navController.graph)

        // Initialize Connect Button
        connectButton = binding.fab as Button
        
        // Start pulse animation
        val pulseAnimation = AnimationUtils.loadAnimation(this, R.anim.hud_pulse)
        connectButton.startAnimation(pulseAnimation)
        
        connectButton.setOnClickListener {
            initializeWebSocket()
            VideoWebsocketContainer.connect()
        }

        // Initialize WebSocket using the singleton instance
        initializeWebSocket()

        // Add connection status listener
        WebsocketContainer.addConnectionStatusListener(::updateConnectButtonState)
        
        // Initialize video WebSocket connection
        VideoWebsocketContainer.connect()

        Thread {
            val success = SntpClient.requestTime()
            runOnUiThread {
                if (success) {
                    WebsocketContainer.sendStatus("Time synchronized. Offset: ${SntpClient.ntpTimeOffset} ms")
                } else {
                    WebsocketContainer.sendStatus("Time synchronization failed.")
                }
            }
        }.start()

        // Observe SDK Manager
        observeSDKManager()

        checkAndRequestPermissions()

        // Initialize BasicAircraftControlVM
        basicAircraftControlVM = BasicAircraftControlVM()

        // Initialize UI components
        btnTakeOff = findViewById(R.id.btn_takeoff)
        btnLand = findViewById(R.id.btn_land)

        // Initialize WebSocket message handler AFTER all required components are initialized
        websocketMessageHandler = WebsocketMessageHandler(
            this,
            virtualStickVM,
            LiveStreamVM(),  // Create a new instance
            basicAircraftControlVM
        )
        
        // Set the message handler in WebsocketContainer
        WebsocketContainer.setMessageHandler(websocketMessageHandler)

        btnTakeOff.setOnClickListener {
            checkAndEnableVirtualStickThenExecute {
                basicAircraftControlVM.startTakeOff(object :
                    CommonCallbacks.CompletionCallbackWithParam<EmptyMsg> {
                    override fun onSuccess(t: EmptyMsg?) {
                        showToast("start takeOff onSuccess.")
                        WebsocketContainer.sendStatus("start takeOff onSuccess.")
                    }

                    override fun onFailure(error: IDJIError) {
                        showToast("start takeOff onFailure, $error")
                        WebsocketContainer.sendStatus("start takeOff onFailure, $error")
                    }
                })
            }
        }

        btnLand.setOnClickListener {
            checkAndEnableVirtualStickThenExecute {
                basicAircraftControlVM.startLanding(object :
                    CommonCallbacks.CompletionCallbackWithParam<EmptyMsg> {
                    override fun onSuccess(t: EmptyMsg?) {
                        showToast("start landing onSuccess.")
                    }

                    override fun onFailure(error: IDJIError) {
                        showToast("start landing onFailure, $error")
                    }
                })
            }
        }
    }

    private fun initializeWebSocket() {
        WebsocketContainer.connect { message ->
            Log.d(TAG, "Handled message: $message")
            runOnUiThread {
                showToast(message)
            }
        }
    }

    private fun updateConnectButtonState(isConnected: Boolean) {
        runOnUiThread {
            val pulseAnimation = AnimationUtils.loadAnimation(this, R.anim.hud_pulse)
            if (isConnected) {
                connectButton.text = "SERVER CONNECTED"
                connectButton.clearAnimation()
                connectButton.alpha = 1.0f
                connectButton.isEnabled = false
                
                // Fade out after 2 seconds
                connectButton.animate()
                    .alpha(0f)
                    .setStartDelay(2000)
                    .setDuration(1000)
                    .withEndAction {
                        connectButton.visibility = View.GONE
                    }
                    .start()
            } else {
                // Cancel any ongoing animations
                connectButton.animate().cancel()
                connectButton.visibility = View.VISIBLE
                connectButton.alpha = 0f
                connectButton.text = "CONNECT TO SERVER"
                connectButton.isEnabled = true
                
                // Fade in
                connectButton.animate()
                    .alpha(1.0f)
                    .setDuration(500)
                    .withEndAction {
                        connectButton.startAnimation(pulseAnimation)
                    }
                    .start()
            }
        }
    }

    private fun observeSDKManager() {
        // Observe Register State
        MSDKManager.lvRegisterState.observe(this) { resultPair ->
            val message = if (resultPair.first) {
                "Register Success"
            } else {
                "Register Failure: ${resultPair.second}"
            }
            showToast(message)
            WebsocketContainer.sendStatus(message)
        }

        // Observe Product Connection State
        MSDKManager.lvProductConnectionState.observe(this) { resultPair ->
            val message = "Product: ${resultPair.second}, ConnectionState: ${resultPair.first}"
            showToast(message)
            WebsocketContainer.sendStatus(message)
        }

        // Observe Product Changes
        MSDKManager.lvProductChanges.observe(this) { productId ->
            val message = "Product: $productId Changed"
            showToast(message)
            WebsocketContainer.sendStatus(message)
        }

        // Observe Init Process
        MSDKManager.lvInitProcess.observe(this) { processPair ->
            val message = "Init Process event: ${processPair.first.name}"
            showToast(message)
            WebsocketContainer.sendStatus(message)
        }

        // Observe Database Download Progress
        MSDKManager.lvDBDownloadProgress.observe(this) { resultPair ->
            val message = "Database Download Progress current: ${resultPair.first}, total: ${resultPair.second}"
            showToast(message)
            WebsocketContainer.sendStatus(message)
        }
    }

    private fun areAllPermissionsGranted(): Boolean {
        return getMissingPermissions().isEmpty()
    }

    private fun checkAndRequestPermissions() {
        val missingPermissions = getMissingPermissions()
        
        if (missingPermissions.isEmpty()) {
            WebsocketContainer.sendStatus("All required permissions are granted")
        } else {
            WebsocketContainer.sendStatus("Missing permissions: ${missingPermissions.joinToString(", ")}")
            requestPermissions(missingPermissions)
        }
    }

    private fun getMissingPermissions(): List<String> {
        val requiredPermissions = mutableListOf(
            Manifest.permission.READ_PHONE_STATE,
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.ACCESS_COARSE_LOCATION,
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_NETWORK_STATE,
            Manifest.permission.ACCESS_WIFI_STATE,
            Manifest.permission.INTERNET,
            Manifest.permission.KILL_BACKGROUND_PROCESSES,
            Manifest.permission.VIBRATE
        )

        // Add storage permissions based on API level
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            requiredPermissions.addAll(listOf(
                Manifest.permission.READ_MEDIA_IMAGES,
                Manifest.permission.READ_MEDIA_VIDEO,
                Manifest.permission.READ_MEDIA_AUDIO
            ))
        } else {
            requiredPermissions.addAll(listOf(
                Manifest.permission.READ_EXTERNAL_STORAGE,
                Manifest.permission.WRITE_EXTERNAL_STORAGE
            ))
        }

        val missingPermissions = requiredPermissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }.toMutableList()

        // Check for MANAGE_EXTERNAL_STORAGE separately
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R && !Environment.isExternalStorageManager()) {
            missingPermissions.add(Manifest.permission.MANAGE_EXTERNAL_STORAGE)
        }

        return missingPermissions
    }

    private fun requestPermissions(permissions: List<String>) {
        val regularPermissions = permissions.filter { it != Manifest.permission.MANAGE_EXTERNAL_STORAGE }
        if (regularPermissions.isNotEmpty()) {
            WebsocketContainer.sendStatus("Requesting regular permissions: ${regularPermissions.joinToString(", ")}")
            ActivityCompat.requestPermissions(this, regularPermissions.toTypedArray(), REQUEST_PERMISSION_CODE)
        }
        
        if (permissions.contains(Manifest.permission.MANAGE_EXTERNAL_STORAGE)) {
            requestManageExternalStoragePermission()
        }
    }

    private fun showPermissionSettingsDialog() {
        AlertDialog.Builder(this)
            .setTitle("Permissions Required")
            .setMessage("Some permissions are still needed. Please grant them in the app settings.")
            .setPositiveButton("Go to Settings") { _, _ ->
                val intent = Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS)
                val uri = Uri.fromParts("package", packageName, null)
                intent.data = uri
                startActivity(intent)
            }
            .setNegativeButton("Cancel") { dialog, _ ->
                dialog.dismiss()
                WebsocketContainer.sendStatus("User cancelled permission request. App may not function correctly.")
            }
            .setCancelable(false)
            .show()
    }

    private fun requestManageExternalStoragePermission() {
        WebsocketContainer.sendStatus("Requesting MANAGE_EXTERNAL_STORAGE permission")
        val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION).apply {
            data = Uri.parse("package:$packageName")
        }
        startActivityForResult(intent, REQUEST_MANAGE_EXTERNAL_STORAGE_PERMISSION)
    }

    // Handle Permission Request Results
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_PERMISSION_CODE) {
            val grantedPermissions = permissions.filterIndexed { index, _ -> grantResults[index] == PackageManager.PERMISSION_GRANTED }
            val deniedPermissions = permissions.filterIndexed { index, _ -> grantResults[index] != PackageManager.PERMISSION_GRANTED }
            
            WebsocketContainer.sendStatus("Granted permissions: ${grantedPermissions.joinToString(", ")}")
            WebsocketContainer.sendStatus("Denied permissions: ${deniedPermissions.joinToString(", ")}")

            val stillMissingPermissions = getMissingPermissions()
            if (stillMissingPermissions.isEmpty()) {
                WebsocketContainer.sendStatus("All required permissions are now granted")
            } else {
                WebsocketContainer.sendStatus("Still missing permissions: ${stillMissingPermissions.joinToString(", ")}")
                showPermissionSettingsDialog()
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_MANAGE_EXTERNAL_STORAGE_PERMISSION) {
            if (Environment.isExternalStorageManager()) {
                WebsocketContainer.sendStatus("MANAGE_EXTERNAL_STORAGE permission granted")
            } else {
                WebsocketContainer.sendStatus("MANAGE_EXTERNAL_STORAGE permission denied")
            }
            checkAndRequestPermissions() // Re-check all permissions
        }
    }

    // Inflate the Options Menu
    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.menu_main, menu)
        return true
    }

    // Handle Option Item Selections
    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            R.id.action_settings -> true // Handle settings action
            else -> super.onOptionsItemSelected(item)
        }
    }

    // Clean Up Resources on Destroy
    override fun onDestroy() {
        super.onDestroy()
        WebsocketContainer.removeConnectionStatusListener(::updateConnectButtonState)
        WebsocketContainer.close()
        VideoWebsocketContainer.close()
    }

    // Support Navigate Up
    override fun onSupportNavigateUp(): Boolean {
        val navController = findNavController(R.id.nav_host_fragment_content_main)
        return navController.navigateUp(appBarConfiguration)
                || super.onSupportNavigateUp()
    }

    private fun checkAndEnableVirtualStickThenExecute(action: () -> Unit) {
        if (virtualStickVM.isVirtualStickEnabled.value == true) {
            action()
        } else {
            showToast("Virtual Stick not enabled. Enabling...")
            virtualStickVM.enableVirtualStick { success, error ->
                if (success) {
                    showToast("Virtual Stick enabled successfully.")
                    action()
                } else {
                    showToast("Failed to enable Virtual Stick: $error")
                }
            }
        }
    }

    private fun showSnackbar(message: String) {
        Snackbar.make(binding.root, message, Snackbar.LENGTH_LONG).show()
    }

    private fun showToast(content: String) {
        Toast.makeText(this, content, Toast.LENGTH_SHORT).show()
    }

    fun setVideoModeAndResolution() {
        // First, ensure the camera is in VIDEO_NORMAL mode.
        val cameraModeKey = KeyTools.createKey(
            CameraKey.KeyCameraMode,
            ComponentIndexType.LEFT_OR_MAIN
        )
        // Set camera mode to VIDEO_NORMAL. (Ensure that VIDEO_NORMAL is defined in your DJI SDK v5)
        KeyManager.getInstance().setValue(cameraModeKey, CameraMode.VIDEO_NORMAL, object : CommonCallbacks.CompletionCallback {
            override fun onSuccess() {
                // After the camera mode is set, now set the video resolution and frame rate.
                setCameraVideoResolutionAndFrameRate()
            }

            override fun onFailure(error: IDJIError) {
                runOnUiThread {
                    Toast.makeText(
                        this@MainActivity,
                        "Failed to update VideoInformation (mode): ${error.description()}",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }
        })
    }
    fun setCameraVideoResolutionAndFrameRate() {
        val resolutionKey = KeyTools.createKey(
            CameraKey.KeyVideoResolutionFrameRate,
            ComponentIndexType.LEFT_OR_MAIN
        )
        // Here we try to set a desired setting.
        val desiredSetting = VideoResolutionFrameRate(VideoResolution.RESOLUTION_1920x1080, VideoFrameRate.RATE_60FPS)

        KeyManager.getInstance().setValue(resolutionKey, desiredSetting, object : CommonCallbacks.CompletionCallback {
            override fun onSuccess() {
                runOnUiThread {
                    Toast.makeText(
                        this@MainActivity,
                        "Successfully updated VideoInformation",
                        Toast.LENGTH_SHORT
                    ).show()
                    WebsocketContainer.sendStatus("Successfully updated VideoInformation")
                }
            }

            override fun onFailure(error: IDJIError) {
                // Declare errorDetails at the top so it's in scope.
                val errorDetails = error.toString()

                runOnUiThread {
                    Toast.makeText(
                        this@MainActivity,
                        "Error: $errorDetails",
                        Toast.LENGTH_LONG
                    ).show()
                }

                // Now query the supported video resolution/frame rate range.
                val rangeKey = KeyTools.createKey(
                    CameraKey.KeyVideoResolutionFrameRateRange,
                    ComponentIndexType.LEFT_OR_MAIN
                )
                KeyManager.getInstance().getValue(rangeKey, object : CommonCallbacks.CompletionCallbackWithParam<List<VideoResolutionFrameRate>> {
                    override fun onSuccess(result: List<VideoResolutionFrameRate>?) {
                        // Convert each VideoResolutionFrameRate to a friendly JSON string.
                        val friendlyRanges = result?.map { range ->
                            val resolutionStr = when (range.resolution) {
                                VideoResolution.RESOLUTION_640x480 -> "640x480"
                                VideoResolution.RESOLUTION_1280x720 -> "1280x720"
                                VideoResolution.RESOLUTION_1920x1080 -> "1920x1080"
                                VideoResolution.RESOLUTION_3840x2160 -> "3840x2160"
                                else -> "Unknown(${range.resolution})"
                            }
                            val frameRateStr = when (range.frameRate) {
                                VideoFrameRate.RATE_30FPS -> "30fps"
                                VideoFrameRate.RATE_60FPS -> "60fps"
                                else -> "Unknown(${range.frameRate})"
                            }
                            // Return a JSON-like string for each supported setting.
                            "{\"resolution\": \"$resolutionStr\", \"frameRate\": \"$frameRateStr\"}"
                        }?.joinToString(separator = ", ") ?: "No supported ranges available"

                        // Send the full error message to the WebSocket.
                        WebsocketContainer.sendError(
                            "Failed to update VideoInformation. Supported ranges: $friendlyRanges",
                            errorDetails
                        )
                    }
                    override fun onFailure(getValueError: IDJIError) {
                        WebsocketContainer.sendError(
                            "Failed to update VideoInformation and to retrieve supported ranges: ${getValueError}",
                            errorDetails
                        )
                    }
                })
            }
        })
    }
}

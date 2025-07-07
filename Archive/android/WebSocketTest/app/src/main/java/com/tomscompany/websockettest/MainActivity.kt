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
import com.google.android.material.floatingactionbutton.FloatingActionButton

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

    private lateinit var fab: FloatingActionButton

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Data Binding Setup
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val navController = findNavController(R.id.nav_host_fragment_content_main)
        appBarConfiguration = AppBarConfiguration(navController.graph)

        // Initialize FAB
        fab = binding.fab
        fab.setOnClickListener {
            initializeWebSocket()
        }

        // Initialize WebSocket using the singleton instance
        initializeWebSocket()

        // Add connection status listener
        WebsocketContainer.addConnectionStatusListener(::updateFabVisibility)

        // Observe SDK Manager
        observeSDKManager()

        checkAndRequestPermissions()

        // Initialize BasicAircraftControlVM
        basicAircraftControlVM = BasicAircraftControlVM()

        // Initialize UI components
        btnTakeOff = findViewById(R.id.btn_takeoff)
        btnLand = findViewById(R.id.btn_land)

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

    private fun updateFabVisibility(isConnected: Boolean) {
        runOnUiThread {
            if (isConnected) {
                fab.hide()
            } else {
                fab.show()
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
        WebsocketContainer.removeConnectionStatusListener(::updateFabVisibility)
        WebsocketContainer.close()
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
}

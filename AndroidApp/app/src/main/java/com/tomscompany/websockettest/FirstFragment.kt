package com.tomscompany.websockettest

import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.view.ViewGroup
import android.widget.EditText
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.core.app.ActivityCompat
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.viewpager2.adapter.FragmentStateAdapter
import androidx.viewpager2.widget.ViewPager2
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.google.android.material.tabs.TabLayout
import com.google.android.material.tabs.TabLayoutMediator
import com.tomscompany.websockettest.MSDKManager
import com.tomscompany.websockettest.data.ImuData
import com.tomscompany.websockettest.databinding.FragmentFirstBinding
import com.tomscompany.websockettest.virtualstick.OnScreenJoystick
import com.tomscompany.websockettest.virtualstick.OnScreenJoystickListener
import dji.sdk.keyvalue.key.BatteryKey
import dji.sdk.keyvalue.key.FlightControllerKey
import dji.sdk.keyvalue.key.GimbalKey
import dji.sdk.keyvalue.key.KeyTools
import dji.sdk.keyvalue.value.common.Attitude
import dji.sdk.keyvalue.value.common.ComponentIndexType
import dji.sdk.keyvalue.value.common.LocationCoordinate2D
import dji.sdk.keyvalue.value.common.Velocity3D
import dji.sdk.keyvalue.value.flightcontroller.IMUCalibrationInfo
import dji.sdk.keyvalue.value.gimbal.GimbalCalibrationStatusInfo
import dji.v5.manager.KeyManager
import dji.v5.manager.aircraft.virtualstick.Stick
import org.json.JSONObject
import kotlin.math.abs
import android.os.Handler
import android.os.Looper
import dji.v5.common.callback.CommonCallbacks.CompletionCallbackWithParam
import dji.sdk.keyvalue.value.common.EmptyMsg
import dji.v5.common.error.IDJIError
import dji.v5.utils.common.LogUtils
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Spinner
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import android.widget.Button
import android.widget.ProgressBar
import androidx.lifecycle.lifecycleScope



/**
 * A simple [Fragment] subclass as the default destination in the navigation.
 */
class FirstFragment : Fragment() {

    private var _binding: FragmentFirstBinding? = null
    
    // Set the minimum stick movement to register
    private val deviation: Double = 0.02

    // This property is only valid between onCreateView and
    // onDestroyView.
    private val binding get() = _binding!!

    private val virtualStickVM: VirtualStickVM by activityViewModels()
    private val liveStreamVM: LiveStreamVM by activityViewModels()
    private val h264VideoStreamVM: H264VideoStreamVM by activityViewModels()
    private val basicAircraftControlVM: BasicAircraftControlVM by activityViewModels()
    private val videoFileStreamVM: VideoFileStreamVM by activityViewModels()

    private lateinit var batteryStatusText: TextView
    private lateinit var surfaceView: SurfaceView

    // Add at the top with other properties
    private var latestImuData = ImuData()
    private lateinit var fabIpSettings: Button
    private lateinit var btnVirtualStick: Button
    private lateinit var altitudeValue: TextView
    private lateinit var speedDisplayValue: TextView
    private lateinit var batteryBar: ProgressBar
    private var isDroneFlying = false
    private var isDroneConnected = false

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {

        _binding = FragmentFirstBinding.inflate(inflater, container, false)
        return binding.root

    }


    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        liveStreamVM.setFragment(this)
        setupJoysticks()
        // setupStreamButton() - Removed JPEG streaming
        setupSpeedSlider()
        setupBatteryStatus()
        setupVideoPreview()
        // Temporarily disabled IMU data collection
        // setupIMUListener()
        // setupRecordButton() - Removed for HUD UI
        // setupResolutionSpinner() - Removed for HUD UI
        setupH264StreamButton()
        virtualStickVM.setSpeedLevel(0.001)
        setupConfigButton()
        observeWebSocketConnection()
        setupCalibrationListeners() 
        disableGimbal()
        setupTelemetryDisplay()
        setupTakeoffButton()
        listenToFlightState()
        setupVirtualStickButton()
        observeDroneConnection()
    }

    // Initialize on-screen joystick listeners
    private fun setupJoysticks() {
        // Set up left joystick listener
        binding.joystickLeft.setJoystickListener(object : OnScreenJoystickListener {
            override fun onTouch(joystick: OnScreenJoystick?, pX: Float, pY: Float) {
                var leftPx = 0F
                var leftPy = 0F

                // Apply deviation threshold
                if (abs(pX) >= deviation) {
                    leftPx = pX
                }

                if (abs(pY) >= deviation) {
                    leftPy = pY
                }

                // Update left joystick position
                virtualStickVM.setLeftPosition(
                    (leftPx * Stick.MAX_STICK_POSITION_ABS).toInt(),
                    (leftPy * Stick.MAX_STICK_POSITION_ABS).toInt()
                )
            }
        })

        // Set up right joystick listener
        binding.joystickRight.setJoystickListener(object : OnScreenJoystickListener {
            override fun onTouch(joystick: OnScreenJoystick?, pX: Float, pY: Float) {
                var rightPx = 0F
                var rightPy = 0F

                // Apply deviation threshold
                if (abs(pX) >= deviation) {
                    rightPx = pX
                }

                if (abs(pY) >= deviation) {
                    rightPy = pY
                }

                // Update right joystick position
                virtualStickVM.setRightPosition(
                    (rightPx * Stick.MAX_STICK_POSITION_ABS).toInt(),
                    (rightPy * Stick.MAX_STICK_POSITION_ABS).toInt()
                )
            }
        })
    }

    // Removed setupStreamButton() - JPEG streaming no longer needed

    // Removed JPEG streaming methods

    override fun onDestroyView() {
        // Stop H.264 streaming if active
        if (h264VideoStreamVM.isStreaming()) {
            h264VideoStreamVM.stopH264Streaming()
        }
        
        // Stop video file streaming if active
        if (videoFileStreamVM.isStreaming.value == true) {
            videoFileStreamVM.stopStreaming()
        }
        
        liveStreamVM.setVideoSurface(null, 0, 0)
        super.onDestroyView()
        KeyManager.getInstance().cancelListen(this)
        _binding = null
        WebsocketContainer.removeConnectionStatusListener { }
    }

    private fun setupSpeedSlider() {
        val speedSlider = binding.speedSlider
        val speedValueText = binding.speedValue

        // Set initial speed to 0.001
        speedSlider.progress = 0
        speedValueText.text = "THROTTLE: 0.001"

        speedSlider.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                val speedValue = (progress + 1).toDouble() / 1000.0 
                val formattedSpeed = String.format("%.3f", speedValue)
                speedValueText.text = "THROTTLE: $formattedSpeed"
                virtualStickVM.setSpeedLevel(speedValue)
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
    }

    private fun setupBatteryStatus() {
        batteryStatusText = binding.batteryStatus
        batteryBar = binding.batteryBar
        
        // Set initial state to N/A
        batteryStatusText.text = "PWR: N/A"
        batteryStatusText.setTextColor(Color.GRAY)
        batteryBar.progress = 0
        batteryBar.progressDrawable.setColorFilter(
            Color.GRAY,
            android.graphics.PorterDuff.Mode.SRC_IN
        )

        // Listen for battery level changes
        KeyManager.getInstance().listen(
            KeyTools.createKey(
                BatteryKey.KeyChargeRemainingInPercent, ComponentIndexType.LEFT_OR_MAIN
            ),
            this
        ) { _, newValue ->
            WebsocketContainer.send("Battery level changed: $newValue")
            if (newValue != null && isDroneConnected) {
                updateBatteryStatus(newValue as Int)
            }
        }
    }

    private fun updateBatteryStatus(batteryPercentage: Int) {
        activity?.runOnUiThread {
            if (!isDroneConnected) {
                batteryStatusText.text = "PWR: N/A"
                batteryStatusText.setTextColor(Color.GRAY)
                if (::batteryBar.isInitialized) {
                    batteryBar.progress = 0
                }
                return@runOnUiThread
            }
            
            val statusText = "PWR: $batteryPercentage%"
            batteryStatusText.text = statusText
            
            // Set text color based on battery percentage
            when {
                batteryPercentage <= 20 -> batteryStatusText.setTextColor(Color.RED)
                batteryPercentage <= 50 -> batteryStatusText.setTextColor(Color.rgb(255, 165, 0)) // Orange
                else -> batteryStatusText.setTextColor(Color.WHITE)
            }
            
            // Update battery bar if initialized
            if (::batteryBar.isInitialized) {
                batteryBar.progress = batteryPercentage
                when {
                    batteryPercentage <= 20 -> batteryBar.progressDrawable.setColorFilter(
                        resources.getColor(R.color.hud_battery_low, null),
                        android.graphics.PorterDuff.Mode.SRC_IN
                    )
                    batteryPercentage <= 50 -> batteryBar.progressDrawable.setColorFilter(
                        resources.getColor(R.color.hud_battery_medium, null),
                        android.graphics.PorterDuff.Mode.SRC_IN
                    )
                    else -> batteryBar.progressDrawable.setColorFilter(
                        resources.getColor(R.color.hud_battery_high, null),
                        android.graphics.PorterDuff.Mode.SRC_IN
                    )
                }
            }
            
            val batteryJson = JSONObject().apply {
                put("type", "battery_status")
                put("percentage", batteryPercentage)
                put("status_text", statusText)
                put("connected", isDroneConnected)
                put("timestamp", System.currentTimeMillis())
            }
            WebsocketContainer.send(batteryJson.toString())
        }
    }

    private fun setupVideoPreview() {
        surfaceView = binding.videoSurfaceView
        altitudeValue = binding.altitudeValue
        speedDisplayValue = binding.speedValueDisplay
        batteryBar = binding.batteryBar
        surfaceView.holder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceCreated(holder: SurfaceHolder) {
                // When surface is created, set it as the video preview surface
                liveStreamVM.setVideoSurface(holder.surface, surfaceView.width, surfaceView.height)
            }

            override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
                // Update the video surface when the surface changes
                liveStreamVM.setVideoSurface(holder.surface, width, height)
            }

            override fun surfaceDestroyed(holder: SurfaceHolder) {
                // Clean up when surface is destroyed
                liveStreamVM.setVideoSurface(null, 0, 0)
            }
        })
    }

    private fun setupIMUListener() {
        // Listen for flight controller state changes
        KeyManager.getInstance().listen(
            KeyTools.createKey(
                FlightControllerKey.KeyConnection,
                ComponentIndexType.LEFT_OR_MAIN
            ),
            this
        ) { _: Boolean?, newValue: Boolean? -> 
            if (newValue == true) {
                listenToFlightControllerData()
            }
        }

        // Listen for gimbal connection status
        KeyManager.getInstance().listen(
            KeyTools.createKey(
                GimbalKey.KeyConnection,
                ComponentIndexType.LEFT_OR_MAIN
            ),
            this
        ) { _: Boolean?, newValue: Boolean? ->
            if (newValue == true) {
                listenToGimbalData()
            }
        }
    }

    private fun listenToFlightControllerData() {
        // Listen for velocity data
        KeyManager.getInstance().listen(
            KeyTools.createKey(
                FlightControllerKey.KeyAircraftVelocity,
                ComponentIndexType.LEFT_OR_MAIN
            ),
            this
        ) { _: Velocity3D?, newValue: Velocity3D? ->
            newValue?.let { sendVelocityData(it) }
        }

        // Listen for aircraft location (using LocationCoordinate2D)
        KeyManager.getInstance().listen(
            KeyTools.createKey(
                FlightControllerKey.KeyAircraftLocation,
                ComponentIndexType.LEFT_OR_MAIN
            ),
            this
        ) { _: LocationCoordinate2D?, newValue: LocationCoordinate2D? ->
            newValue?.let { sendLocationData(it) }
        }

        // Listen for attitude data
        KeyManager.getInstance().listen(
            KeyTools.createKey(
                FlightControllerKey.KeyAircraftAttitude,
                ComponentIndexType.LEFT_OR_MAIN
            ),
            this
        ) { _: Attitude?, newValue: Attitude? ->
            newValue?.let { sendAttitudeData(it) }
        }
    }

    private fun listenToGimbalData() {
        // Listen for gimbal attitude data
        KeyManager.getInstance().listen(
            KeyTools.createKey(
                GimbalKey.KeyGimbalAttitude,
                ComponentIndexType.LEFT_OR_MAIN
            ),
            this
        ) { _: Attitude?, newValue: Attitude? ->
            newValue?.let { sendGimbalAttitudeData(it) }
        }

        // Listen for yaw relative to aircraft
        KeyManager.getInstance().listen(
            KeyTools.createKey(
                GimbalKey.KeyYawRelativeToAircraftHeading,
                ComponentIndexType.LEFT_OR_MAIN
            ),
            this
        ) { _: Double?, newValue: Double? ->
            newValue?.let { yaw ->
                latestImuData = latestImuData.copy(
                    gimbalAttitude = latestImuData.gimbalAttitude.copy(
                        yawRelativeToAircraft = yaw
                    ),
                    timestamp = System.currentTimeMillis()
                )
            }
        }
    }

    private fun sendVelocityData(velocity: Velocity3D) {
        latestImuData = latestImuData.copy(
            velocity = ImuData.Velocity(
                velocity.x.toDouble(),
                velocity.y.toDouble(),
                velocity.z.toDouble()
            ),
            timestamp = System.currentTimeMillis()
        )
        
        // Update speed display
        val speed = kotlin.math.sqrt(velocity.x * velocity.x + velocity.y * velocity.y).toFloat()
        updateSpeedDisplay(speed)
    }

    private fun sendLocationData(location: LocationCoordinate2D) {
        latestImuData = latestImuData.copy(
            location = ImuData.Location(
                location.latitude,
                location.longitude
            ),
            timestamp = System.currentTimeMillis()
        )
    }

    private fun sendAttitudeData(attitude: Attitude) {
        latestImuData = latestImuData.copy(
            attitude = ImuData.Attitude(
                attitude.pitch.toDouble(),
                attitude.roll.toDouble(),
                attitude.yaw.toDouble()
            ),
            timestamp = System.currentTimeMillis()
        )
    }

    private fun sendGimbalAttitudeData(attitude: Attitude) {
        latestImuData = latestImuData.copy(
            gimbalAttitude = ImuData.Attitude(
                attitude.pitch.toDouble(),
                attitude.roll.toDouble(),
                attitude.yaw.toDouble()
            ),
            timestamp = System.currentTimeMillis()
        )
    }

    fun getLatestImuData(): ImuData = latestImuData.withCurrentTimestamp()

    private fun setupConfigButton() {
        binding.fabIpSettings.setOnClickListener {
            showIpSettingsDialog()
        }
    }

    private fun observeWebSocketConnection() {
        WebsocketContainer.addConnectionStatusListener { isConnected ->
            // No visibility changes here anymore
        }
    }
    
    private fun observeDroneConnection() {
        // Observe drone connection state from MSDKManager
        MSDKManager.lvProductConnectionState.observe(viewLifecycleOwner) { resultPair ->
            isDroneConnected = resultPair.first
            
            activity?.runOnUiThread {
                if (!isDroneConnected) {
                    // Update battery display to N/A when disconnected
                    batteryStatusText.text = "PWR: N/A"
                    batteryStatusText.setTextColor(Color.GRAY)
                    if (::batteryBar.isInitialized) {
                        batteryBar.progress = 0
                        batteryBar.progressDrawable.setColorFilter(
                            Color.GRAY,
                            android.graphics.PorterDuff.Mode.SRC_IN
                        )
                    }
                    
                    // Send disconnection status via WebSocket
                    val disconnectJson = JSONObject().apply {
                        put("type", "battery_status")
                        put("percentage", -1)
                        put("status_text", "PWR: N/A")
                        put("connected", false)
                        put("timestamp", System.currentTimeMillis())
                    }
                    WebsocketContainer.send(disconnectJson.toString())
                }
            }
        }
    }

    private fun showIpSettingsDialog() {
        val dialogView = layoutInflater.inflate(R.layout.dialog_settings_tabbed, null)
        val tabLayout = dialogView.findViewById<TabLayout>(R.id.tab_layout)
        val viewPager = dialogView.findViewById<ViewPager2>(R.id.view_pager)

        // Set up the ViewPager2 adapter with our two fragments.
        viewPager.adapter = object : FragmentStateAdapter(this) {
            override fun getItemCount() = 2
            override fun createFragment(position: Int): Fragment {
                return when (position) {
                    0 -> ServerSettingsFragment()
                    1 -> CameraSettingsFragment()
                    else -> throw IllegalStateException("Invalid position $position")
                }
            }
        }
        TabLayoutMediator(tabLayout, viewPager) { tab, position ->
            tab.text = when (position) {
                0 -> "Server Settings"
                1 -> "Camera Settings"
                else -> null
            }
        }.attach()

        // (Optional) perform an initial update if needed
        updateCameraInfo(dialogView)

        // Create the AlertDialog with a positive (Save) and negative (Cancel) button.
        val dialog = AlertDialog.Builder(requireContext())
            .setTitle("Settings")
            .setView(dialogView)
            .setPositiveButton("Save", null) // We will override the click listener later
            .setNegativeButton("Cancel", null)
            .create()

        dialog.setOnShowListener {
            val positiveButton = dialog.getButton(AlertDialog.BUTTON_POSITIVE)
            // Initially update the button text based on the currently selected tab
            when (tabLayout.selectedTabPosition) {
                0 -> {
                    positiveButton.text = "Save"
                    positiveButton.visibility = View.VISIBLE
                }
                1 -> {
                    positiveButton.text = "Update"
                    positiveButton.visibility = View.VISIBLE
                }
            }
            // Listen for tab changes and update the button text accordingly
            tabLayout.addOnTabSelectedListener(object : TabLayout.OnTabSelectedListener {
                override fun onTabSelected(tab: TabLayout.Tab?) {
                    when (tab?.position) {
                        0 -> {
                            positiveButton.text = "Save"
                            positiveButton.visibility = View.VISIBLE
                        }
                        1 -> {
                            positiveButton.text = "Update"
                            positiveButton.visibility = View.VISIBLE
                        }
                    }
                }
                override fun onTabUnselected(tab: TabLayout.Tab?) {}
                override fun onTabReselected(tab: TabLayout.Tab?) {}
            })

            // Set click listener for the positive button
            positiveButton.setOnClickListener {
                when (tabLayout.selectedTabPosition) {
                    // Server Settings tab (position 0) uses the existing "Save" logic:
                    0 -> {
                        val serverView = (viewPager.getChildAt(0) as? ViewGroup)?.getChildAt(0)
                        if (serverView != null) {
                            val ipInput = serverView.findViewById<EditText>(R.id.et_ip_address)
                            val portInput = serverView.findViewById<EditText>(R.id.et_port)
                            val newIp = ipInput.text.toString()
                            val newPort = portInput.text.toString()

                            if (isValidIpAddress(newIp) && isValidPort(newPort)) {
                                WebsocketContainer.setServerAddress(newIp, newPort)
                                WebsocketContainer.connect { message ->
                                    activity?.runOnUiThread {
                                        Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
                                    }
                                }
                                dialog.dismiss()
                            } else {
                                Toast.makeText(context, "Invalid IP address or port", Toast.LENGTH_SHORT).show()
                            }
                        }
                    }
                    // Camera Settings tab (position 1) now calls setVideoModeAndResolution()
                    1 -> {
                        (activity as? MainActivity)?.setVideoModeAndResolution()
                        Toast.makeText(context, "Camera settings updated", Toast.LENGTH_SHORT).show()
                        dialog.dismiss()
                    }
                }
            }
        }

        dialog.show()
    }

    
    private fun isValidIpAddress(ip: String): Boolean {
        return try {
            val parts = ip.split(".")
            parts.size == 4 && parts.all { it.toInt() in 0..255 }
        } catch (e: Exception) {
            false
        }
    }
    
    private fun isValidPort(port: String): Boolean {
        return try {
            port.toInt() in 1..65535
        } catch (e: Exception) {
            false
        }
    }

    private fun updateCameraInfo(dialogView: View) {
        // Get the latest stream info
        val streamInfo = liveStreamVM.getCameraStreamInfo()
        // Find the ViewPager2 from the dialog view
        val viewPager = dialogView.findViewById<ViewPager2>(R.id.view_pager)
        
        // Post the update so that the ViewPager2 has completed its layout
        viewPager.post {
            // Loop through the ViewPager2's children to find the one that contains the camera settings TextViews.
            for (i in 0 until viewPager.childCount) {
                val child = viewPager.getChildAt(i)
                // Try to find the TextView for frame rate
                val frameRateText = child.findViewById<TextView>(R.id.tv_frame_rate)
                if (frameRateText != null) {
                    // Update the camera info text views
                    frameRateText.text = "Frame Rate: ${streamInfo.frameRate} fps"
                    child.findViewById<TextView>(R.id.tv_width)?.text = "Width: ${streamInfo.width} px"
                    child.findViewById<TextView>(R.id.tv_height)?.text = "Height: ${streamInfo.height} px"
                    break
                }
            }
        }
    }


    // Removed setupRecordButton() - Not needed for HUD UI
    
    private fun setupH264StreamButton() {
        // Initialize button state
        binding.btnH264Stream.isEnabled = true
        binding.btnH264Stream.text = "STREAM"
        
        binding.btnH264Stream.setOnClickListener {
            // Disable button during operation
            binding.btnH264Stream.isEnabled = false
            
            if (!h264VideoStreamVM.isStreaming()) {
                startH264Streaming()
            } else {
                stopH264Streaming()
            }
            
            // Re-enable after short delay
            Handler(Looper.getMainLooper()).postDelayed({
                binding.btnH264Stream.isEnabled = true
            }, 1000)
        }
        
        // Observe H.264 stream statistics
        h264VideoStreamVM.streamStats.observe(viewLifecycleOwner) { stats ->
            // Update UI with stream stats
            val statsText = String.format(
                "VIDEO: %.1f MBPS | %.1f FPS",
                stats.bitrate / 1_000_000,
                stats.fps
            ).uppercase()
            binding.h264StatsText.text = statsText
            binding.h264StatsText.visibility = View.VISIBLE
        }
        
        // Observe stream errors
        h264VideoStreamVM.streamError.observe(viewLifecycleOwner) { error ->
            error?.let {
                val errorMsg = when (it) {
                    H264VideoStreamVM.StreamError.ENCODER_ERROR -> "Encoder error"
                    H264VideoStreamVM.StreamError.NETWORK_ERROR -> "Network error"
                    H264VideoStreamVM.StreamError.RESOLUTION_CHANGE -> "Resolution changed"
                    H264VideoStreamVM.StreamError.FRAME_TIMEOUT -> "No frames received"
                }
                Log.e("FirstFragment", "H.264 stream error: $errorMsg")
                
                // Show error in stats text
                binding.h264StatsText.text = "H.264: Error - $errorMsg"
                binding.h264StatsText.setTextColor(resources.getColor(R.color.hud_battery_low, null))
            } ?: run {
                // Error cleared, restore normal color
                binding.h264StatsText.setTextColor(resources.getColor(R.color.hud_text_label, null))
            }
        }
    }
    
    private fun startH264Streaming() {
        // Ensure WebSocket is connected
        if (!WebsocketContainer.isConnected()) {
            Toast.makeText(context, "WebSocket not connected", Toast.LENGTH_SHORT).show()
            return
        }
        
        // Ensure Video WebSocket is connected
        if (!VideoWebsocketContainer.isConnected()) {
            Toast.makeText(context, "Video WebSocket not connected", Toast.LENGTH_SHORT).show()
            return
        }
        
        // Check if JPEG streaming is active
        if (liveStreamVM.isStreaming()) {
            Toast.makeText(context, "Please stop JPEG streaming first", Toast.LENGTH_SHORT).show()
            return
        }
        
        // Check if we have a drone connection
        if (!isDroneConnected) {
            // No drone connected, stream video file instead
            try {
                videoFileStreamVM.startVideoFileStreaming(
                    requireContext(),
                    "FULL_HALLWAY.mp4"
                ) { videoData ->
                    // Send video data via WebSocket on IO thread
                    lifecycleScope.launch(Dispatchers.IO) {
                        try {
                            VideoWebsocketContainer.sendBinary(videoData)
                        } catch (e: Exception) {
                            Log.e("FirstFragment", "Error sending video file data", e)
                        }
                    }
                }
                
                binding.btnH264Stream.text = "PAUSE"
                binding.h264StatsText.visibility = View.VISIBLE
                Toast.makeText(context, "Streaming video file: FULL_HALLWAY.mp4", Toast.LENGTH_SHORT).show()
                
                // Observe video file stream statistics
                videoFileStreamVM.streamStats.observe(viewLifecycleOwner) { stats ->
                    val statsText = String.format(
                        "FILE VIDEO: %.1f MBPS | %.1f FPS",
                        stats.bitrate / 1_000_000,
                        stats.fps
                    ).uppercase()
                    binding.h264StatsText.text = statsText
                }
                
                // Send notification via WebSocket
                val streamJson = JSONObject().apply {
                    put("type", "video_file_stream_status")
                    put("status", "started")
                    put("filename", "FULL_HALLWAY.mp4")
                    put("timestamp", System.currentTimeMillis())
                }
                WebsocketContainer.send(streamJson.toString())
                
                return
            } catch (e: Exception) {
                Log.e("FirstFragment", "Error starting video file streaming", e)
                Toast.makeText(context, "Error: ${e.message}", Toast.LENGTH_LONG).show()
                return
            }
        }
        
        // Check if encoder is supported for drone streaming
        if (!h264VideoStreamVM.validateEncoderSupport()) {
            Toast.makeText(context, "H.264 encoder not supported on this device", Toast.LENGTH_LONG).show()
            return
        }
        
        try {
            h264VideoStreamVM.startH264Streaming { videoData ->
                // Send H.264 data via WebSocket on IO thread
                lifecycleScope.launch(Dispatchers.IO) {
                    try {
                        VideoWebsocketContainer.sendBinary(videoData)
                    } catch (e: Exception) {
                        Log.e("FirstFragment", "Error sending H.264 data", e)
                    }
                }
            }
        } catch (e: Exception) {
            Log.e("FirstFragment", "Failed to start H.264 streaming", e)
            Toast.makeText(context, "Failed to start H.264 streaming: ${e.message}", Toast.LENGTH_LONG).show()
            return
        }
        
        binding.btnH264Stream.text = "STOP"
        Toast.makeText(context, "H.264 streaming started", Toast.LENGTH_SHORT).show()
        
        // Send notification via WebSocket
        val streamJson = JSONObject().apply {
            put("type", "h264_stream_status")
            put("status", "started")
            put("timestamp", System.currentTimeMillis())
        }
        WebsocketContainer.send(streamJson.toString())
    }
    
    private fun stopH264Streaming() {
        // Check if we're streaming a video file
        if (videoFileStreamVM.isStreaming.value == true) {
            if (videoFileStreamVM.isPaused.value == true) {
                // Resume video file streaming
                videoFileStreamVM.resumeStreaming()
                binding.btnH264Stream.text = "PAUSE"
                Toast.makeText(context, "Video file streaming resumed", Toast.LENGTH_SHORT).show()
            } else {
                // Pause video file streaming
                videoFileStreamVM.pauseStreaming()
                binding.btnH264Stream.text = "RESUME"
                Toast.makeText(context, "Video file streaming paused", Toast.LENGTH_SHORT).show()
            }
            return
        }
        
        // Stop drone H.264 streaming
        h264VideoStreamVM.stopH264Streaming()
        binding.btnH264Stream.text = "STREAM"
        binding.h264StatsText.visibility = View.GONE
        Toast.makeText(context, "H.264 streaming stopped", Toast.LENGTH_SHORT).show()
        
        // Send notification via WebSocket
        val streamJson = JSONObject().apply {
            put("type", "h264_stream_status")
            put("status", "stopped")
            put("timestamp", System.currentTimeMillis())
        }
        WebsocketContainer.send(streamJson.toString())
    }

    private fun checkStoragePermissions(): Boolean {
        return if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
            android.os.Environment.isExternalStorageManager()
        } else {
            val write = androidx.core.content.ContextCompat.checkSelfPermission(
                requireContext(),
                android.Manifest.permission.WRITE_EXTERNAL_STORAGE
            )
            val read = androidx.core.content.ContextCompat.checkSelfPermission(
                requireContext(),
                android.Manifest.permission.READ_EXTERNAL_STORAGE
            )
            write == android.content.pm.PackageManager.PERMISSION_GRANTED &&
            read == android.content.pm.PackageManager.PERMISSION_GRANTED
        }
    }

    private fun requestStoragePermissions() {
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
            try {
                val intent = android.content.Intent(android.provider.Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION)
                intent.addCategory("android.intent.category.DEFAULT")
                intent.data = android.net.Uri.parse("package:${requireContext().packageName}")
                startActivityForResult(intent, 2296)
            } catch (e: Exception) {
                val intent = android.content.Intent()
                intent.action = android.provider.Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION
                startActivityForResult(intent, 2296)
            }
        } else {
            ActivityCompat.requestPermissions(
                requireActivity(),
                arrayOf(
                    android.Manifest.permission.WRITE_EXTERNAL_STORAGE,
                    android.Manifest.permission.READ_EXTERNAL_STORAGE
                ),
                1
            )
        }
    }

    private fun setupCalibrationListeners() {
        // Listen for IMU calibration status
        KeyManager.getInstance().listen(
            KeyTools.createKey(
                FlightControllerKey.KeyIMUCalibrationInfo,
                ComponentIndexType.LEFT_OR_MAIN
            ),
            this
        ) { _: IMUCalibrationInfo?, newValue: IMUCalibrationInfo? ->
            newValue?.let { updateImuCalibrationStatus(it) }
        }

        // Listen for Gimbal calibration status
        KeyManager.getInstance().listen(
            KeyTools.createKey(
                GimbalKey.KeyGimbalCalibrationStatus,
                ComponentIndexType.LEFT_OR_MAIN
            ),
            this
        ) { _: GimbalCalibrationStatusInfo?, newValue: GimbalCalibrationStatusInfo? ->
            newValue?.let { updateGimbalCalibrationStatus(it) }
        }
    }

    private fun updateImuCalibrationStatus(info: IMUCalibrationInfo) {
        latestImuData = latestImuData.copy(
            imuCalibrationStatus = ImuData.IMUCalibrationStatus(
                state = info.calibrationState.name,
                orientationsNeeded = info.orientationsToCalibrate.map { it.name },
            )
        )
    }

    private fun updateGimbalCalibrationStatus(info: GimbalCalibrationStatusInfo) {
        latestImuData = latestImuData.copy(
            gimbalCalibrationStatus = ImuData.GimbalCalibrationStatus(
                state = info.status.name,
                progress = info.progress
            )
        )
    }




    private fun disableGimbal() {
        val actionKey = KeyTools.createKey(
            GimbalKey.KeyTurnOffGimbal,
            ComponentIndexType.LEFT_OR_MAIN  // adjust as needed for your gimbal index
        )
        KeyManager.getInstance().performAction(
            actionKey,
            EmptyMsg(), // No parameters required
            object : CompletionCallbackWithParam<EmptyMsg> {
                override fun onSuccess(result: EmptyMsg) {
                    LogUtils.d("Gimbal", "turnOffGimbal: success")
                }
                override fun onFailure(error: IDJIError) {
                    LogUtils.e("Gimbal", "turnOffGimbal: failed - ${error.description()}")
                }
            }
        )
    }

    // Removed setupResolutionSpinner() - Not needed for HUD UI

    private fun setupVirtualStickButton() {
        btnVirtualStick = binding.btnVirtualStick

        // Observe virtual stick state
        virtualStickVM.isVirtualStickEnabled.observe(viewLifecycleOwner) { isEnabled ->
            btnVirtualStick.text = if (isEnabled) "V-STICK ON" else "V-STICK"
            // Show/hide joysticks based on virtual stick state
            binding.joystickLeft.visibility = if (isEnabled) View.VISIBLE else View.GONE
            binding.joystickRight.visibility = if (isEnabled) View.VISIBLE else View.GONE
        }

        btnVirtualStick.setOnClickListener {
            if (virtualStickVM.isVirtualStickEnabled.value == true) {
                virtualStickVM.disableVirtualStick { success, error ->
                    activity?.runOnUiThread {
                        if (success) {
                            Toast.makeText(context, "Virtual Stick disabled", Toast.LENGTH_SHORT).show()
                        } else {
                            Toast.makeText(context, "Failed to disable Virtual Stick: $error", Toast.LENGTH_SHORT).show()
                        }
                    }
                }
            } else {
                virtualStickVM.enableVirtualStick { success, error ->
                    activity?.runOnUiThread {
                        if (success) {
                            Toast.makeText(context, "Virtual Stick enabled", Toast.LENGTH_SHORT).show()
                        } else {
                            Toast.makeText(context, "Failed to enable Virtual Stick: $error", Toast.LENGTH_SHORT).show()
                        }
                    }
                }
            }
        }
    }
    
    private fun setupTelemetryDisplay() {
        // Initialize telemetry displays with formatted values
        altitudeValue.text = formatAltitude(0.0)
        speedDisplayValue.text = formatSpeed(0.0f)
        
        // Listen for altitude updates
        KeyManager.getInstance().listen(
            KeyTools.createKey(
                FlightControllerKey.KeyAltitude,
                ComponentIndexType.LEFT_OR_MAIN
            ),
            this
        ) { _: Double?, newValue: Double? ->
            newValue?.let { updateAltitudeDisplay(it.toFloat()) }
        }
    }
    
    private fun setupTakeoffButton() {
        binding.btnTakeoff.setOnClickListener {
            if (!isDroneFlying) {
                // Takeoff logic
                binding.btnTakeoff.isEnabled = false
                basicAircraftControlVM.startTakeOff(object : CompletionCallbackWithParam<EmptyMsg> {
                    override fun onSuccess(p0: EmptyMsg) {
                        activity?.runOnUiThread {
                            binding.btnTakeoff.text = "LAND"
                            binding.btnTakeoff.isEnabled = true
                            isDroneFlying = true
                            Toast.makeText(context, "Takeoff successful", Toast.LENGTH_SHORT).show()
                        }
                    }
                    
                    override fun onFailure(error: IDJIError) {
                        activity?.runOnUiThread {
                            binding.btnTakeoff.isEnabled = true
                            Toast.makeText(context, "Takeoff failed: ${error.description()}", Toast.LENGTH_LONG).show()
                        }
                    }
                })
            } else {
                // Land logic
                binding.btnTakeoff.isEnabled = false
                basicAircraftControlVM.startLanding(object : CompletionCallbackWithParam<EmptyMsg> {
                    override fun onSuccess(p0: EmptyMsg) {
                        activity?.runOnUiThread {
                            binding.btnTakeoff.text = "TAKE OFF"
                            binding.btnTakeoff.isEnabled = true
                            isDroneFlying = false
                            Toast.makeText(context, "Landing successful", Toast.LENGTH_SHORT).show()
                        }
                    }
                    
                    override fun onFailure(error: IDJIError) {
                        activity?.runOnUiThread {
                            binding.btnTakeoff.isEnabled = true
                            Toast.makeText(context, "Landing failed: ${error.description()}", Toast.LENGTH_LONG).show()
                        }
                    }
                })
            }
        }
    }
    
    private fun updateAltitudeDisplay(altitude: Float) {
        activity?.runOnUiThread {
            altitudeValue.text = formatAltitude(altitude.toDouble())
        }
    }
    
    private fun updateSpeedDisplay(speed: Float) {
        activity?.runOnUiThread {
            speedDisplayValue.text = formatSpeed(speed)
        }
    }
    
    private fun formatAltitude(altitude: Double): String {
        return String.format("%05.1f M", altitude)
    }
    
    private fun formatSpeed(speed: Float): String {
        return String.format("%04.1f M/S", speed)
    }
    
    private fun listenToFlightState() {
        // Listen for flying state
        KeyManager.getInstance().listen(
            KeyTools.createKey(
                FlightControllerKey.KeyIsFlying,
                ComponentIndexType.LEFT_OR_MAIN
            ),
            this
        ) { _: Boolean?, newValue: Boolean? ->
            newValue?.let { isFlying ->
                activity?.runOnUiThread {
                    isDroneFlying = isFlying
                    binding.btnTakeoff.text = if (isFlying) "LAND" else "TAKE OFF"
                }
            }
        }
    }
}

class ServerSettingsFragment : Fragment() {
    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.dialog_server_settings, container, false)
    }
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        // Pre-populate the fields from WebsocketContainer
        val (ip, port) = WebsocketContainer.getServerAddress()
        view.findViewById<EditText>(R.id.et_ip_address)?.setText(ip)
        view.findViewById<EditText>(R.id.et_port)?.setText(port)
    }
}


class CameraSettingsFragment : Fragment() {
    // Assuming LiveStreamVM is defined elsewhere in your project.
    private val liveStreamVM: LiveStreamVM by activityViewModels()

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: android.os.Bundle?
    ): View? {
        // Inflate your camera settings dialog layout.
        return inflater.inflate(R.layout.dialog_camera_settings, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: android.os.Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Observe the streamInfo LiveData and update the TextViews.
        liveStreamVM.streamInfoLiveData.observe(viewLifecycleOwner) { streamInfo ->
            view.findViewById<TextView>(R.id.tv_frame_rate)?.text =
                "Frame Rate: " + if (streamInfo.frameRate > 0) "${streamInfo.frameRate} fps" else "N/A"
            view.findViewById<TextView>(R.id.tv_width)?.text =
                "Width: " + if (streamInfo.width > 0) "${streamInfo.width} px" else "N/A"
            view.findViewById<TextView>(R.id.tv_height)?.text =
                "Height: " + if (streamInfo.height > 0) "${streamInfo.height} px" else "N/A"
        }
    }
}

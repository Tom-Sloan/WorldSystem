package com.tomscompany.websockettest

import android.os.Bundle
import android.view.LayoutInflater
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.view.ViewGroup
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.tomscompany.websockettest.databinding.FragmentFirstBinding
import com.tomscompany.websockettest.virtualstick.OnScreenJoystick
import com.tomscompany.websockettest.virtualstick.OnScreenJoystickListener
import dji.sdk.keyvalue.key.BatteryKey
import dji.sdk.keyvalue.key.FlightControllerKey
import dji.sdk.keyvalue.key.KeyTools
import dji.sdk.keyvalue.key.GimbalKey
import dji.sdk.keyvalue.value.common.ComponentIndexType
import dji.sdk.keyvalue.value.common.Attitude
import dji.sdk.keyvalue.value.common.Velocity3D
import dji.sdk.keyvalue.value.flightcontroller.IMUCalibrationInfo
import dji.sdk.keyvalue.value.gimbal.GimbalCalibrationStatusInfo
import dji.v5.manager.KeyManager
import dji.v5.manager.aircraft.virtualstick.Stick
import kotlin.math.abs
import android.graphics.Color
import org.json.JSONObject
import dji.sdk.keyvalue.value.common.LocationCoordinate2D
import com.tomscompany.websockettest.data.ImuData
import android.widget.EditText
import androidx.core.app.ActivityCompat

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

    private lateinit var batteryStatusText: TextView
    private lateinit var surfaceView: SurfaceView

    // Add at the top with other properties
    private var latestImuData = ImuData()
    private lateinit var fabIpSettings: FloatingActionButton

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
        setupStreamButton()
        setupSpeedSlider()
        setupBatteryStatus()
        setupVideoPreview()
        setupIMUListener()
        setupRecordButton()
        virtualStickVM.setSpeedLevel(0.001)
        setupIpSettingsFab()
        observeWebSocketConnection()
        setupCalibrationListeners()


        
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

    private fun setupStreamButton() {
        binding.btnStartstream.setOnClickListener {
            val buttonJson = JSONObject().apply {
                put("type", "button_event")
                put("button", "stream_button")
                put("action", "clicked")
                put("timestamp", System.currentTimeMillis())
            }
            WebsocketContainer.send(buttonJson.toString())
            
            if (!liveStreamVM.isStreaming()) {
                startStream()
            } else {
                stopStream()
            }
        }
    }

    private fun startStream() {
        if (liveStreamVM.isStreaming()) {
            WebsocketContainer.sendStatus("Stream is already running")
            return
        }

        // Set the surface and start WebSocket streaming
        liveStreamVM.setVideoSurface(surfaceView.holder.surface, surfaceView.width, surfaceView.height)
        liveStreamVM.startFrameStreaming()
        
        activity?.runOnUiThread {
            WebsocketContainer.sendStatus("Stream started successfully")
            binding.btnStartstream.text = "Stop Stream"
        }
    }

    private fun stopStream() {
        liveStreamVM.stopFrameStreaming()
        
        activity?.runOnUiThread {
            WebsocketContainer.sendStatus("Stream stopped successfully")
            binding.btnStartstream.text = "Start Stream"
        }
    }

    override fun onDestroyView() {
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
        speedValueText.text = "Speed: 0.001"

        speedSlider.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                val speedValue = (progress + 1).toDouble() / 1000.0 
                val formattedSpeed = String.format("%.3f", speedValue)
                speedValueText.text = "Speed: $formattedSpeed"
                virtualStickVM.setSpeedLevel(speedValue)
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
    }

    private fun setupBatteryStatus() {
        batteryStatusText = binding.batteryStatus

        // Listen for battery level changes
        KeyManager.getInstance().listen(
            KeyTools.createKey(
                BatteryKey.KeyChargeRemainingInPercent, ComponentIndexType.LEFT_OR_MAIN
            ),
            this
        ) { _, newValue ->
            WebsocketContainer.send("Battery level changed: $newValue")
            if (newValue != null) {
                updateBatteryStatus(newValue as Int)
            }
        }
    }

    private fun updateBatteryStatus(batteryPercentage: Int) {
        activity?.runOnUiThread {
            val statusText = "Battery: $batteryPercentage%"
            batteryStatusText.text = statusText
            
            // Set text color based on battery percentage
            when {
                batteryPercentage <= 20 -> batteryStatusText.setTextColor(Color.RED)
                batteryPercentage <= 50 -> batteryStatusText.setTextColor(Color.rgb(255, 165, 0)) // Orange
                else -> batteryStatusText.setTextColor(Color.BLACK)
            }
            
            val batteryJson = JSONObject().apply {
                put("type", "battery_status")
                put("percentage", batteryPercentage)
                put("status_text", statusText)
                put("timestamp", System.currentTimeMillis())
            }
            WebsocketContainer.send(batteryJson.toString())
        }
    }

    private fun setupVideoPreview() {
        surfaceView = binding.videoSurfaceView
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

    private fun setupIpSettingsFab() {
        fabIpSettings = binding.fabIpSettings
        fabIpSettings.setOnClickListener {
            showIpSettingsDialog()
        }
    }

    private fun observeWebSocketConnection() {
        WebsocketContainer.addConnectionStatusListener { isConnected ->
            activity?.runOnUiThread {
                fabIpSettings.visibility = if (isConnected) View.GONE else View.VISIBLE
            }
        }
    }

    private fun showIpSettingsDialog() {
        val (currentIp, currentPort) = WebsocketContainer.getServerAddress()
        
        val dialogView = layoutInflater.inflate(R.layout.dialog_ip_settings, null)
        val ipInput = dialogView.findViewById<EditText>(R.id.et_ip_address)
        val portInput = dialogView.findViewById<EditText>(R.id.et_port)
        
        ipInput.setText(currentIp)
        portInput.setText(currentPort)
        
        AlertDialog.Builder(requireContext())
            .setTitle("Server Settings")
            .setView(dialogView)
            .setPositiveButton("Save") { _, _ ->
                val newIp = ipInput.text.toString()
                val newPort = portInput.text.toString()
                
                if (isValidIpAddress(newIp) && isValidPort(newPort)) {
                    WebsocketContainer.setServerAddress(newIp, newPort)
                    WebsocketContainer.connect { message ->
                        activity?.runOnUiThread {
                            Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
                        }
                    }
                } else {
                    Toast.makeText(context, "Invalid IP address or port", Toast.LENGTH_SHORT).show()
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
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

    private fun setupRecordButton() {
        binding.btnRecord.setOnClickListener {
            if (checkStoragePermissions()) {
                if (!liveStreamVM.isRecording()) {
                    liveStreamVM.startRecording(requireContext())
                    binding.btnRecord.text = "Stop Recording"
                    Toast.makeText(context, "Recording started", Toast.LENGTH_SHORT).show()
                } else {
                    liveStreamVM.stopRecording(requireContext())
                    binding.btnRecord.text = "Start Recording"
                    Toast.makeText(context, "Recording stopped", Toast.LENGTH_SHORT).show()
                }
            } else {
                requestStoragePermissions()
            }
        }
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
                // message = when (info.calibrationState) {
                //     IMUCalibrationState.NOT_CALIBRATING -> "Ready"
                //     IMUCalibrationState.CALIBRATING -> "Calibration in progress"
                //     IMUCalibrationState.SUCCESS -> "Calibration successful"
                //     IMUCalibrationState.FAILED -> "Calibration failed"
                //     else -> "Unknown calibration state"
                // }
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
}

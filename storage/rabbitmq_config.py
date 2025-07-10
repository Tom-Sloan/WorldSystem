"""
RabbitMQ Exchange Configuration
Centralized configuration for all RabbitMQ exchanges and routing keys
"""

# Exchange definitions
EXCHANGES = {
    'sensor_data': {
        'name': 'sensor_data',
        'type': 'topic',
        'durable': True,
        'auto_delete': False
    },
    'processing_results': {
        'name': 'processing_results', 
        'type': 'topic',
        'durable': True,
        'auto_delete': False
    },
    'control_commands': {
        'name': 'control_commands',
        'type': 'topic', 
        'durable': True,
        'auto_delete': False
    },
    'assets': {
        'name': 'assets',
        'type': 'topic',
        'durable': True,
        'auto_delete': False
    }
}

# Routing key definitions
ROUTING_KEYS = {
    # Sensor data
    'VIDEO_FRAMES': 'sensor.video',
    
    # Processing results
    'YOLO_FRAMES': 'result.frames.yolo',
    'SLAM_POSE': 'result.slam.pose',
    'SLAM_POINTCLOUD': 'result.slam.pointcloud',
    'SLAM_MESH': 'result.slam.mesh',
    
    # Control commands
    'RESTART': 'control.restart',
    'ANALYSIS_MODE': 'control.analysis_mode',
    'SLAM_RESET': 'control.slam.reset',
    
    # Assets
    'PLY_FILE': 'assets.ply',
    'TRAJECTORY': 'assets.trajectory'
}

# Helper function to declare all exchanges
async def declare_exchanges(channel):
    """Declare all exchanges on the given channel."""
    for exchange_config in EXCHANGES.values():
        await channel.exchange_declare(
            exchange=exchange_config['name'],
            exchange_type=exchange_config['type'],
            durable=exchange_config['durable'],
            auto_delete=exchange_config['auto_delete']
        )
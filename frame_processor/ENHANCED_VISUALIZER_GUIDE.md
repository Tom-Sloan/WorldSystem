# Enhanced Rerun Visualizer Guide

## Overview

The Enhanced Rerun Visualizer provides a dual-page visualization system for the frame processor, offering both live monitoring and processed object gallery views. This implementation is designed for Rerun 0.23.2 and provides an intuitive interface for tracking objects through their entire processing pipeline.

## Features

### Page 1: Live Monitoring
- **Live Camera View**: Real-time video feed with integrated object detection overlays
- **Object Overlays**: Color-coded bounding boxes showing tracking stage:
  - 🔵 Blue: Objects being tracked (with countdown to processing)
  - 🟠 Orange: Objects currently being processed
  - 🟢 Green: Objects successfully identified with dimensions
- **Selected Object Details**: Detailed information panel showing:
  - Object identification status
  - Real-world dimensions (when available)
  - Frame quality analysis scores
  - Processing history
- **Processing Timeline**: Time-series visualization of:
  - FPS (Frames Per Second)
  - Active object count
  - Processing stages for each tracked object
- **Live Statistics**: Real-time performance metrics

### Page 2: Process View
- **Processed Gallery**: Grid display of enhanced object images
- **Identification Results**: Detailed product information and dimensions
- **Processing Statistics**: Success rates, processing times, and scene scale
- **Scene Understanding**: Real-world scale factor calculation based on identified objects

## Installation

The enhanced visualizer is automatically integrated when the frame processor starts. No additional installation is required beyond the standard frame processor dependencies.

## Usage

### Basic Operation

The visualizer starts automatically with both pages visible side by side. The frame processor will:

1. Display live camera feed with object overlays on Page 1
2. Show processed objects in the gallery on Page 2
3. Update statistics and timeline in real-time

### View Mode Control

You can switch between different view modes programmatically:

```python
# Show both pages (default)
show_both_pages(processor.visualizer)

# Show only live monitoring page
show_live_page_only(processor.visualizer)

# Show only process/gallery page
show_process_page_only(processor.visualizer)
```

### In the Rerun Viewer

While the application is running, you can also control visibility in the Rerun viewer:

1. **Collapse/Expand Panels**: Click on panel headers to minimize/maximize
2. **Resize Panels**: Drag the dividers between panels
3. **Entity Tree Control**: Use the entity tree on the left to:
   - Check/uncheck specific data paths
   - Collapse entire sections (e.g., `/page1/*` or `/page2/*`)

## Data Organization

The visualizer organizes data into clear hierarchical paths:

### Page 1 (Live Monitoring)
- `/page1/live/camera` - Raw camera feed
- `/page1/live/overlays` - Detection bounding boxes
- `/page1/selected` - Selected object details (Markdown)
- `/page1/timeline/*` - Time-series data for each object
- `/page1/stats` - Live statistics (Markdown)

### Page 2 (Process View)
- `/page2/gallery/item_*` - Processed object images
- `/page2/results` - Identification results (Markdown)
- `/page2/stats` - Processing statistics (Markdown)  
- `/page2/scale` - Scene scale information (Markdown)

## Object Processing Pipeline

1. **Detection**: Objects are detected using YOLO
2. **Tracking**: Objects are tracked for 1.5 seconds to ensure stability
3. **Quality Assessment**: Best frame is selected based on:
   - Sharpness
   - Exposure
   - Size in frame
   - Centering
4. **Enhancement**: Image is enhanced for better API processing
5. **Identification**: API identifies product and dimensions
6. **Gallery Display**: Processed object appears in gallery
7. **Scale Calculation**: Dimensions contribute to scene scale estimation

## Customization

### Modifying View Layout

To create a custom layout, edit the blueprint creation methods in `enhanced_rerun_visualizer.py`:

```python
def _create_custom_blueprint(self) -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Vertical(
            # Your custom layout here
        )
    )
```

### Adjusting Gallery Size

Modify the `max_gallery_items` parameter:

```python
visualizer.max_gallery_items = 30  # Default is 20
```

### Custom Overlay Colors

Modify the color selection in `log_frame_with_overlays()`:

```python
# Example: Use custom colors
if track.estimated_dimensions:
    colors.append([0, 255, 0])  # RGB values
```

## Performance Considerations

- **Compression**: Images are compressed with JPEG quality 85 for efficient transmission
- **Gallery Limit**: Only the most recent 20 objects are kept in gallery
- **Update Frequency**: Statistics update with each frame
- **Memory Usage**: Old gallery items are automatically removed

## Troubleshooting

### No Visualization Appearing
1. Check that `RERUN_ENABLED=true` in environment
2. Verify Rerun viewer is running at configured address
3. Check console logs for connection errors

### Overlays Not Showing
1. Ensure objects are being detected (check YOLO is enabled)
2. Verify tracking is working (check console logs)
3. Check that frame dimensions are valid

### Gallery Not Updating
1. Verify objects are staying in frame for 1.5+ seconds
2. Check API connectivity for dimension processing
3. Ensure enhancement processing is enabled

## API Integration

The visualizer automatically integrates with:
- **YOLO**: For object detection
- **Object Tracker**: For maintaining object identity
- **Frame Scorer**: For quality assessment
- **API Client**: For product identification
- **Scene Scaler**: For real-world measurements

## Development

To extend the visualizer:

1. Add new data logging in appropriate methods
2. Update blueprint to include new views
3. Follow the entity path naming convention
4. Use appropriate Rerun archetypes for your data type

## Best Practices

1. **Entity Paths**: Keep paths organized and hierarchical
2. **Markdown Formatting**: Use tables and styling for clear information display
3. **Performance**: Log only necessary data to avoid overwhelming the viewer
4. **Colors**: Use consistent color coding across the interface
5. **Labels**: Keep labels concise but informative

## Example Integration

```python
# In your processing loop
if RERUN_ENABLED:
    # The enhanced visualizer handles all logging
    # Just ensure your processor has the visualizer integrated
    processor.process_frame(frame, properties, frame_number)
    # Visualization happens automatically!
```

## Summary

The Enhanced Rerun Visualizer provides a comprehensive view of the object detection and identification pipeline. By organizing data into two focused pages, users can monitor live performance while reviewing processed results. The flexible view system allows focusing on specific aspects of the pipeline as needed.
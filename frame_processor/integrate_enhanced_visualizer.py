"""
Integration script to use the enhanced Rerun visualizer with the frame processor.
This replaces the existing Rerun logging with the new dual-page visualization.
"""

import sys
import os

# Add the frame_processor directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhanced_rerun_visualizer import (
    EnhancedRerunVisualizer, 
    show_both_pages,
    show_live_page_only, 
    show_process_page_only
)
import time


def integrate_enhanced_visualization(processor):
    """
    Integrate the enhanced visualizer with the existing frame processor.
    
    This function modifies the processor to use the new visualization system.
    """
    # Create visualizer instance
    visualizer = EnhancedRerunVisualizer()
    processor.visualizer = visualizer
    
    # Set configurable parameters
    visualizer.max_gallery_items = int(os.getenv("MAX_GALLERY_ITEMS", "20"))
    
    # Store original methods
    original_process_frame = processor.process_frame
    original_enhancement_worker = processor.enhancement_worker
    original_log_tracking_status = processor.log_tracking_status
    original_log_dimension_results = processor.log_dimension_results
    
    # Enhanced process_frame that uses new visualizer
    def enhanced_process_frame(frame, properties, frame_number):
        # Call original processing
        result = original_process_frame(frame, properties, frame_number)
        
        # Get timestamp if available from properties headers
        timestamp_ns = None
        try:
            if hasattr(properties, 'headers') and properties.headers:
                timestamp_ns = properties.headers.get('timestamp_ns')
                if timestamp_ns and isinstance(timestamp_ns, str):
                    timestamp_ns = int(timestamp_ns)
        except Exception as e:
            # If there's any issue accessing headers, continue without timestamp
            print(f"[EnhancedVisualizer] Could not extract timestamp: {e}")
            timestamp_ns = None
        
        # Log frame with overlays using new visualizer
        visualizer.log_frame_with_overlays(
            frame, 
            processor.tracker.tracked_objects,
            frame_number,
            timestamp_ns
        )
        
        # Pass tracker configuration to visualizer
        visualizer.set_tracker_config(
            process_after_seconds=processor.tracker.process_after_seconds,
            reprocess_interval_seconds=processor.tracker.reprocess_interval_seconds
        )
        
        # Update queue size for statistics
        visualizer.set_queue_size(processor.enhancement_queue.qsize())
        
        # Update selected object (select first object with dimensions or being processed)
        selected_track = None
        for track in processor.tracker.tracked_objects.values():
            if track.estimated_dimensions or track.is_being_processed:
                selected_track = track
                break
        
        if selected_track:
            visualizer.log_selected_object(selected_track)
        elif processor.tracker.tracked_objects:
            # Select first tracked object if none are processed yet
            first_track = next(iter(processor.tracker.tracked_objects.values()))
            visualizer.log_selected_object(first_track)
        
        return result
    
    # Enhanced enhancement worker that logs to gallery
    def enhanced_enhancement_worker():
        """Enhanced worker that logs processed objects to gallery."""
        while True:
            try:
                track = processor.enhancement_queue.get(timeout=1)
                
                # Log processing start
                print(f"[Enhancement] Processing object {track.id} ({track.class_name})")
                
                # Extract best frame region
                x1, y1, x2, y2 = track.best_bbox
                roi = track.best_frame[y1:y2, x1:x2]
                
                # Enhance if enabled
                if processor.enhancement_enabled:
                    enhanced_roi = processor.enhancer.enhance_frame(roi)
                else:
                    enhanced_roi = roi
                
                # Process with API for dimensions
                start_time = time.time()
                dimension_result = processor.api_client.process_object_for_dimensions(
                    enhanced_roi, track.id, track.class_name
                )
                processing_time = time.time() - start_time
                
                if dimension_result:
                    # Update track with dimension info
                    track.identified_products = dimension_result.get('all_products', [])
                    track.estimated_dimensions = dimension_result.get('dimensions')
                    track.processing_time = processing_time
                    
                    # Update visualizer processing time
                    visualizer.set_last_processing_time(processing_time * 1000)
                    
                    # Add to scene scaler
                    processor.scene_scaler.add_dimension_estimate(
                        track.id, track.class_name, dimension_result, track.confidence
                    )
                    
                    # Log to gallery using new visualizer
                    visualizer.log_processed_object_to_gallery(track, enhanced_roi)
                    
                    # Update scene scale
                    scale_info = processor.scene_scaler.calculate_weighted_scale()
                    if scale_info['confidence'] > 0:
                        visualizer.log_scene_scale(scale_info)
                    
                    # Publish results
                    processor.update_and_publish_scene_scale()
                
                # Mark processing complete
                track.is_being_processed = False
                
            except Exception as e:
                if e.__class__.__name__ != 'Empty':
                    print(f"[Enhancement] Error: {e}")
    
    # Replace methods
    processor.process_frame = enhanced_process_frame
    processor.enhancement_worker = enhanced_enhancement_worker
    
    # Disable old Rerun logging methods
    processor.log_tracking_status = lambda: None
    processor.log_dimension_results = lambda track, result: None
    
    # Restart enhancement thread with new worker
    if hasattr(processor, 'enhancement_thread'):
        # The old thread will exit on its own due to daemon=True
        import threading
        processor.enhancement_thread = threading.Thread(target=enhanced_enhancement_worker)
        processor.enhancement_thread.daemon = True
        processor.enhancement_thread.start()
    
    print("[EnhancedRerunVisualizer] Integration complete!")
    print("[EnhancedRerunVisualizer] Available view modes:")
    print("  - show_both_pages(processor.visualizer)     # Default: both pages side by side")
    print("  - show_live_page_only(processor.visualizer) # Only live monitoring")
    print("  - show_process_page_only(processor.visualizer) # Only gallery/results")
    
    return visualizer


# Example usage in main processing loop
if __name__ == "__main__":
    print("""
    To integrate the enhanced visualizer with your frame processor:
    
    1. Import this integration function:
       from integrate_enhanced_visualizer import integrate_enhanced_visualization
    
    2. After creating your processor, integrate the visualizer:
       processor = EnhancedFrameProcessor()
       visualizer = integrate_enhanced_visualization(processor)
    
    3. The visualizer will automatically handle all Rerun logging with the new dual-page layout.
    
    4. You can switch view modes at any time:
       show_live_page_only(visualizer)    # Focus on live monitoring
       show_process_page_only(visualizer)  # Focus on processed gallery
       show_both_pages(visualizer)         # Show both pages (default)
    
    5. The visualization includes:
       - Page 1: Live camera with integrated overlays, selected object details, timeline
       - Page 2: Gallery of processed objects, identification results, statistics
    """)
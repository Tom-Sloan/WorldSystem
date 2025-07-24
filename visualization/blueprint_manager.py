"""
Blueprint manager for Rerun visualization layouts.
Creates structured views matching the original frame_processor visualization.
"""

import rerun as rr
import rerun.blueprint as rrb
from enum import Enum
from typing import Optional


class ViewMode(Enum):
    """Visualization view modes."""
    BOTH = "both"
    LIVE_ONLY = "live"
    PROCESS_ONLY = "process"


class BlueprintManager:
    """Manages Rerun blueprint layouts for different view modes."""
    
    def __init__(self):
        """Initialize blueprint manager."""
        self.current_mode = ViewMode.BOTH
    
    def create_blueprint(self, mode: ViewMode = ViewMode.BOTH) -> rrb.Blueprint:
        """
        Create blueprint for specified view mode.
        
        Args:
            mode: Visualization mode (BOTH, LIVE_ONLY, PROCESS_ONLY)
            
        Returns:
            Rerun blueprint
        """
        self.current_mode = mode
        
        if mode == ViewMode.BOTH:
            return self._create_both_view()
        elif mode == ViewMode.LIVE_ONLY:
            return self._create_live_view()
        elif mode == ViewMode.PROCESS_ONLY:
            return self._create_process_view()
    
    def _create_both_view(self) -> rrb.Blueprint:
        """Create blueprint showing both live and process pages side by side."""
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                # Page 1: Live Monitoring
                rrb.Vertical(
                    rrb.Horizontal(
                        # Live Camera View
                        rrb.Spatial2DView(
                            origin="page1/live_camera",
                            name="📹 Live Camera",
                            contents=[
                                "page1/live_camera/frame",
                                "page1/live_camera/visualization",
                                "page1/live_camera/masks"
                            ]
                        ),
                        # Frame Grid
                        rrb.Spatial2DView(
                            origin="page1/frame_grid",
                            name="🎬 Frame Grid",
                            contents=["page1/frame_grid/**"]
                        ),
                        column_shares=[2, 1]
                    ),
                    rrb.Horizontal(
                        # Selected Object Info
                        rrb.TextDocumentView(
                            origin="page1/selected",
                            name="🎯 Selected Object",
                            contents=["page1/selected/**"]
                        ),
                        # Timeline
                        rrb.TimeSeriesView(
                            origin="page1/timeline",
                            name="⏱️ Timeline",
                            contents=["page1/timeline/**"]
                        ),
                        column_shares=[1, 1]
                    ),
                    # Live Stats
                    rrb.TextDocumentView(
                        origin="page1/stats",
                        name="📊 Statistics",
                        contents=["page1/stats/**"]
                    ),
                    row_shares=[3, 2, 1]
                ),
                
                # Page 2: Process Gallery
                rrb.Vertical(
                    # Processed Gallery
                    rrb.Spatial2DView(
                        origin="page2/gallery",
                        name="🖼️ Processed Gallery",
                        contents=["page2/gallery/**"]
                    ),
                    rrb.Horizontal(
                        # ID Results
                        rrb.TextDocumentView(
                            origin="page2/results",
                            name="📋 ID Results",
                            contents=["page2/results/**"]
                        ),
                        # Process Stats
                        rrb.TextDocumentView(
                            origin="page2/stats",
                            name="📈 Statistics",
                            contents=["page2/stats/**"]
                        ),
                        column_shares=[2, 1]
                    ),
                    row_shares=[3, 1]
                ),
                column_shares=[1, 1]
            ),
            auto_layout=False
        )
        return blueprint
    
    def _create_live_view(self) -> rrb.Blueprint:
        """Create blueprint showing only live monitoring page."""
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    # Live Camera View (larger when alone)
                    rrb.Spatial2DView(
                        origin="page1/live_camera",
                        name="📹 Live Camera",
                        contents=[
                            "page1/live_camera/frame",
                            "page1/live_camera/visualization",
                            "page1/live_camera/masks"
                        ]
                    ),
                    # Frame Grid
                    rrb.Spatial2DView(
                        origin="page1/frame_grid",
                        name="🎬 Frame Grid",
                        contents=["page1/frame_grid/**"]
                    ),
                    column_shares=[3, 1]
                ),
                rrb.Horizontal(
                    # Selected Object Info
                    rrb.TextDocumentView(
                        origin="page1/selected",
                        name="🎯 Selected Object",
                        contents=["page1/selected/**"]
                    ),
                    # Timeline
                    rrb.TimeSeriesView(
                        origin="page1/timeline",
                        name="⏱️ Timeline",
                        contents=["page1/timeline/**"]
                    ),
                    # Live Stats
                    rrb.TextDocumentView(
                        origin="page1/stats",
                        name="📊 Statistics",
                        contents=["page1/stats/**"]
                    ),
                    column_shares=[1, 1, 1]
                ),
                row_shares=[3, 1]
            ),
            auto_layout=False
        )
        return blueprint
    
    def _create_process_view(self) -> rrb.Blueprint:
        """Create blueprint showing only process gallery page."""
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                # Processed Gallery (full width)
                rrb.Spatial2DView(
                    origin="page2/gallery",
                    name="🖼️ Processed Gallery",
                    contents=["page2/gallery/**"]
                ),
                rrb.Horizontal(
                    # ID Results
                    rrb.TextDocumentView(
                        origin="page2/results",
                        name="📋 ID Results",
                        contents=["page2/results/**"]
                    ),
                    # Process Stats
                    rrb.TextDocumentView(
                        origin="page2/stats",
                        name="📈 Statistics",
                        contents=["page2/stats/**"]
                    ),
                    # Timeline (added in process view)
                    rrb.TimeSeriesView(
                        origin="page1/timeline",
                        name="⏱️ Object Timeline",
                        contents=["page1/timeline/**"]
                    ),
                    column_shares=[2, 1, 1]
                ),
                row_shares=[3, 1]
            ),
            auto_layout=False
        )
        return blueprint
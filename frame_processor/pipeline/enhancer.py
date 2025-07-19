"""
Image enhancement module.

This module provides image enhancement capabilities to improve quality
before API processing, preserving all functionality from the original
modules/enhancement.py while fitting into the new architecture.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict

from core.utils import get_logger, PerformanceTimer
from core.config import Config


logger = get_logger(__name__)


class ImageEnhancer:
    """
    Enhances image quality for better API recognition.
    
    This preserves the exact enhancement logic from the original implementation
    while adding performance monitoring and better configuration management.
    """
    
    def __init__(self, config: Config):
        """
        Initialize image enhancer with configuration.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.enabled = config.enhancement_enabled
        
        # Enhancement parameters from config
        self.gamma = config.enhancement_gamma
        self.alpha = config.enhancement_alpha  # Contrast
        self.beta = config.enhancement_beta    # Brightness
        
        # Auto-adjustment settings
        self.auto_adjust = config.enhancement_auto_adjust
        
        logger.info(
            f"ImageEnhancer initialized - Enabled: {self.enabled}, "
            f"Gamma: {self.gamma}, Alpha: {self.alpha}, Beta: {self.beta}, "
            f"Auto-adjust: {self.auto_adjust}"
        )
    
    def enhance_frame(self, frame: np.ndarray, 
                     bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Enhance frame quality, optionally focusing on a specific region.
        
        Args:
            frame: Input frame
            bbox: Optional bounding box (x1, y1, x2, y2) to focus enhancement
            
        Returns:
            Enhanced frame
        """
        if not self.enabled:
            return frame
        
        with PerformanceTimer("image_enhancement", logger):
            try:
                enhanced = frame.copy()
                
                # Auto-adjust parameters if enabled
                if self.auto_adjust:
                    target_region = frame if bbox is None else self._extract_roi(frame, bbox)
                    params = self.auto_adjust_parameters(target_region)
                    
                    # Use auto-adjusted parameters
                    gamma = params['gamma']
                    alpha = params['alpha']
                    beta = params['beta']
                    
                    logger.debug(
                        f"Auto-adjusted parameters - "
                        f"Gamma: {gamma:.2f}, Alpha: {alpha:.2f}, Beta: {beta:.1f}"
                    )
                else:
                    # Use configured parameters
                    gamma = self.gamma
                    alpha = self.alpha
                    beta = self.beta
                
                if bbox:
                    # Enhance only the region of interest
                    x1, y1, x2, y2 = bbox
                    
                    # Validate bbox
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    
                    if x2 > x1 and y2 > y1:
                        roi = enhanced[y1:y2, x1:x2]
                        enhanced_roi = self._enhance_region(roi, gamma, alpha, beta)
                        enhanced[y1:y2, x1:x2] = enhanced_roi
                else:
                    # Enhance entire frame
                    enhanced = self._enhance_region(enhanced, gamma, alpha, beta)
                
                return enhanced
                
            except Exception as e:
                logger.error(f"Enhancement failed: {e}")
                return frame
    
    def enhance_roi(self, roi: np.ndarray) -> np.ndarray:
        """
        Enhance a region of interest directly.
        
        Args:
            roi: Region of interest image
            
        Returns:
            Enhanced ROI
        """
        if not self.enabled:
            return roi
        
        with PerformanceTimer("roi_enhancement", logger):
            try:
                # Auto-adjust if enabled
                if self.auto_adjust:
                    params = self.auto_adjust_parameters(roi)
                    gamma = params['gamma']
                    alpha = params['alpha']
                    beta = params['beta']
                else:
                    gamma = self.gamma
                    alpha = self.alpha
                    beta = self.beta
                
                return self._enhance_region(roi, gamma, alpha, beta)
                
            except Exception as e:
                logger.error(f"ROI enhancement failed: {e}")
                return roi
    
    def _enhance_region(self, img: np.ndarray, gamma: float, 
                       alpha: float, beta: float) -> np.ndarray:
        """
        Apply enhancement pipeline to an image region.
        
        This preserves the exact enhancement pipeline from the original.
        """
        if img.size == 0:
            return img
        
        try:
            # Step 1: Apply gamma correction for lighting
            img_gamma = self._apply_gamma_correction(img, gamma)
            
            # Step 2: Apply contrast and brightness adjustment
            img_contrast = cv2.convertScaleAbs(img_gamma, alpha=alpha, beta=beta)
            
            # Step 3: Apply denoising to reduce noise
            if img_contrast.shape[0] > 5 and img_contrast.shape[1] > 5:
                img_denoised = cv2.fastNlMeansDenoisingColored(
                    img_contrast, None, 10, 10, 7, 21
                )
            else:
                # Skip denoising for very small images
                img_denoised = img_contrast
            
            # Step 4: Apply sharpening
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            img_sharpened = cv2.filter2D(img_denoised, -1, kernel)
            
            # Step 5: Ensure values are in valid range
            img_final = np.clip(img_sharpened, 0, 255).astype(np.uint8)
            
            return img_final
            
        except Exception as e:
            logger.error(f"Enhancement pipeline failed: {e}")
            return img
    
    def _apply_gamma_correction(self, img: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply gamma correction to improve lighting.
        
        Lower gamma (<1) brightens dark images.
        Higher gamma (>1) darkens bright images.
        """
        try:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 
                             for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(img, table)
        except Exception as e:
            logger.error(f"Gamma correction failed: {e}")
            return img
    
    def auto_adjust_parameters(self, img: np.ndarray) -> Dict[str, float]:
        """
        Automatically adjust enhancement parameters based on image statistics.
        
        This preserves the exact auto-adjustment logic from the original.
        """
        try:
            # Convert to LAB color space for better brightness analysis
            if len(img.shape) == 3:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l_channel = lab[:, :, 0]
            else:
                l_channel = img
            
            # Calculate statistics
            mean_brightness = np.mean(l_channel)
            std_brightness = np.std(l_channel)
            
            # Adjust gamma based on brightness
            if mean_brightness < 50:  # Dark image
                gamma = 0.7  # Brighten significantly
            elif mean_brightness > 200:  # Bright image
                gamma = 1.5  # Darken slightly
            else:
                gamma = 1.2  # Slight brightening for normal images
            
            # Adjust contrast based on standard deviation
            if std_brightness < 20:  # Low contrast
                alpha = 1.5  # Increase contrast
            elif std_brightness > 50:  # High contrast
                alpha = 1.1  # Slight contrast adjustment
            else:
                alpha = 1.3  # Moderate contrast increase
            
            # Adjust brightness offset
            # Target mean brightness around 127 (middle of range)
            beta = max(0, min(50, 127 - mean_brightness))
            
            logger.debug(
                f"Image stats - Mean: {mean_brightness:.1f}, Std: {std_brightness:.1f}"
            )
            
            return {'gamma': gamma, 'alpha': alpha, 'beta': beta}
            
        except Exception as e:
            logger.error(f"Auto-adjustment failed: {e}")
            # Return default parameters on failure
            return {'gamma': self.gamma, 'alpha': self.alpha, 'beta': self.beta}
    
    def _extract_roi(self, frame: np.ndarray, 
                    bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Safely extract ROI from frame with bounds checking.
        
        Args:
            frame: Source frame
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Extracted ROI
        """
        x1, y1, x2, y2 = bbox
        
        # Ensure valid bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            logger.warning(f"Invalid bbox after bounds checking: {bbox}")
            return frame
        
        return frame[y1:y2, x1:x2]
    
    def get_enhancement_stats(self) -> Dict[str, float]:
        """
        Get current enhancement parameters.
        
        Returns:
            Dictionary with current parameters
        """
        return {
            'enabled': self.enabled,
            'gamma': self.gamma,
            'alpha': self.alpha,
            'beta': self.beta,
            'auto_adjust': self.auto_adjust
        }
    
    def update_parameters(self, gamma: Optional[float] = None,
                         alpha: Optional[float] = None,
                         beta: Optional[float] = None):
        """
        Update enhancement parameters at runtime.
        
        Args:
            gamma: New gamma value
            alpha: New contrast value
            beta: New brightness value
        """
        if gamma is not None:
            self.gamma = gamma
            logger.info(f"Updated gamma to {gamma}")
        
        if alpha is not None:
            self.alpha = alpha
            logger.info(f"Updated alpha to {alpha}")
        
        if beta is not None:
            self.beta = beta
            logger.info(f"Updated beta to {beta}")
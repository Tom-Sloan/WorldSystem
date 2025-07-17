import cv2
import numpy as np
from typing import Tuple, Optional, Dict


class ImageEnhancer:
    """Enhances image quality for better API recognition."""
    
    def __init__(self, gamma=1.2, alpha=1.3, beta=20):
        self.gamma = gamma
        self.alpha = alpha  # Contrast
        self.beta = beta    # Brightness
        
    def enhance_frame(self, frame: np.ndarray, 
                     bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """Enhance frame quality, optionally focusing on a specific region."""
        enhanced = frame.copy()
        
        if bbox:
            x1, y1, x2, y2 = bbox
            roi = enhanced[y1:y2, x1:x2]
            enhanced_roi = self._enhance_region(roi)
            enhanced[y1:y2, x1:x2] = enhanced_roi
        else:
            enhanced = self._enhance_region(enhanced)
            
        return enhanced
    
    def _enhance_region(self, img: np.ndarray) -> np.ndarray:
        """Apply enhancement pipeline to an image region."""
        # Apply gamma correction
        img_gamma = self._apply_gamma_correction(img, self.gamma)
        
        # Apply contrast and brightness
        img_contrast = cv2.convertScaleAbs(img_gamma, alpha=self.alpha, beta=self.beta)
        
        # Apply denoising
        img_denoised = cv2.fastNlMeansDenoisingColored(img_contrast, None, 10, 10, 7, 21)
        
        # Apply sharpening
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        img_sharpened = cv2.filter2D(img_denoised, -1, kernel)
        
        # Ensure values are in valid range
        img_final = np.clip(img_sharpened, 0, 255).astype(np.uint8)
        
        return img_final
    
    def _apply_gamma_correction(self, img: np.ndarray, gamma: float) -> np.ndarray:
        """Apply gamma correction to improve lighting."""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(img, table)
    
    def auto_adjust_parameters(self, img: np.ndarray) -> Dict[str, float]:
        """Automatically adjust enhancement parameters based on image statistics."""
        # Convert to LAB color space for better analysis
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Calculate statistics
        mean_brightness = np.mean(l_channel)
        std_brightness = np.std(l_channel)
        
        # Adjust gamma based on brightness
        if mean_brightness < 50:  # Dark image
            gamma = 0.7
        elif mean_brightness > 200:  # Bright image
            gamma = 1.5
        else:
            gamma = 1.2
            
        # Adjust contrast based on standard deviation
        if std_brightness < 20:  # Low contrast
            alpha = 1.5
        elif std_brightness > 50:  # High contrast
            alpha = 1.1
        else:
            alpha = 1.3
            
        # Adjust brightness offset
        beta = max(0, min(50, 127 - mean_brightness))
        
        return {'gamma': gamma, 'alpha': alpha, 'beta': beta}
import cv2
import numpy as np
import json
import os
from enum import Enum, auto
from typing import Dict, Optional, Tuple, Any

class CalibrationMethod(Enum):
    THRESHOLD = auto()
    RGB = auto()
    HSV = auto()
    LAB = auto()
    YCRCB = auto()
    ADAPTIVE_THRESHOLD = auto()
    CANNY_EDGE = auto()

class Calibrator:
    """A class for calibrating image processing parameters for line detection.
    
    Supports multiple calibration methods (e.g., THRESHOLD, RGB, HSV) with a UI
    for real-time adjustment and settings persistence via JSON files.
    
    Attributes:
        name: Unique identifier for the calibrator instance.
        current_method: Active calibration method.
        settings_file: Path to JSON file for saving/loading settings.
        settings: Current calibration parameters.
    """
    def __init__(self, name: str, method: Optional[CalibrationMethod] = None, settings_file: str = 'line_calibration.json'):
        """Initialize the calibrator.
        
        Args:
            name: Identifier for the calibrator instance.
            method: Specific calibration method to use (disables UI if set).
            settings_file: Path to JSON settings file.
        """
        self._suppress_ui = method is not None
        self.name = name
        self.window_prefix = f"Calibrator_{self.name}_"
        self.current_method = method or CalibrationMethod.THRESHOLD
        self.settings_file = settings_file
        
        self.calibration_methods = {
            CalibrationMethod.THRESHOLD: self._threshold_calibration,
            CalibrationMethod.RGB: self._rgb_calibration,
            CalibrationMethod.HSV: self._hsv_calibration,
            CalibrationMethod.LAB: self._lab_calibration,
            CalibrationMethod.YCRCB: self._ycrcb_calibration,
            CalibrationMethod.ADAPTIVE_THRESHOLD: self._adaptive_threshold_calibration,
            CalibrationMethod.CANNY_EDGE: self._canny_edge_calibration
        }
        
        self.default_settings = {
            CalibrationMethod.THRESHOLD.name: {'thresh': 127, 'max_val': 255},
            CalibrationMethod.RGB.name: {'r_low': 0, 'r_high': 255, 'g_low': 0, 'g_high': 255, 'b_low': 0, 'b_high': 255},
            CalibrationMethod.HSV.name: {'h_low': 0, 'h_high': 179, 's_low': 0, 's_high': 255, 'v_low': 0, 'v_high': 255},
            CalibrationMethod.LAB.name: {'l_low': 0, 'l_high': 255, 'a_low': 0, 'a_high': 255, 'b_low': 0, 'b_high': 255},
            CalibrationMethod.YCRCB.name: {'y_low': 0, 'y_high': 255, 'cr_low': 0, 'cr_high': 255, 'cb_low': 0, 'cb_high': 255},
            CalibrationMethod.ADAPTIVE_THRESHOLD.name: {
                'block_size': 11, 'c': 2, 'method': cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 'thresh_type': cv2.THRESH_BINARY_INV
            },
            CalibrationMethod.CANNY_EDGE.name: {'threshold1': 50, 'threshold2': 150, 'aperture_size': 3}
        }
        
        self.settings = self._load_settings()
        if not self._suppress_ui:
            self._init_ui()
    
    def _init_ui(self) -> None:
        """Initialize UI windows for calibration."""
        cv2.namedWindow(f"{self.window_prefix}Calibration")
        cv2.namedWindow(f"{self.window_prefix}Method Selection")
        cv2.namedWindow(f"{self.window_prefix}Original")
        cv2.namedWindow(f"{self.window_prefix}Processed")
        cv2.setMouseCallback(f"{self.window_prefix}Method Selection", self._method_selection_callback)
        self._create_method_buttons()
        self._setup_trackbars()

    def _threshold_calibration(self, frame: np.ndarray) -> np.ndarray:
        """Apply simple threshold calibration.
        
        Args:
            frame: Input BGR frame.
        
        Returns:
            Binary image after thresholding.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(
            gray, self.settings[CalibrationMethod.THRESHOLD.name]['thresh'],
            self.settings[CalibrationMethod.THRESHOLD.name]['max_val'], cv2.THRESH_BINARY_INV
        )
        return binary

    def _rgb_calibration(self, frame: np.ndarray) -> np.ndarray:
        """Apply RGB color space calibration.
        
        Args:
            frame: Input BGR frame.
        
        Returns:
            Binary mask of RGB range.
        """
        r, g, b = frame[:, :, 2], frame[:, :, 1], frame[:, :, 0]
        r_mask = cv2.inRange(r, self.settings[CalibrationMethod.RGB.name]['r_low'], self.settings[CalibrationMethod.RGB.name]['r_high'])
        g_mask = cv2.inRange(g, self.settings[CalibrationMethod.RGB.name]['g_low'], self.settings[CalibrationMethod.RGB.name]['g_high'])
        b_mask = cv2.inRange(b, self.settings[CalibrationMethod.RGB.name]['b_low'], self.settings[CalibrationMethod.RGB.name]['b_high'])
        return cv2.bitwise_and(r_mask, cv2.bitwise_and(g_mask, b_mask))

    def _hsv_calibration(self, frame: np.ndarray) -> np.ndarray:
        """Apply HSV color space calibration.
        
        Args:
            frame: Input BGR frame.
        
        Returns:
            Binary mask of HSV range.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([self.settings[CalibrationMethod.HSV.name][k] for k in ['h_low', 's_low', 'v_low']])
        upper = np.array([self.settings[CalibrationMethod.HSV.name][k] for k in ['h_high', 's_high', 'v_high']])
        return cv2.inRange(hsv, lower, upper)

    def _lab_calibration(self, frame: np.ndarray) -> np.ndarray:
        """Apply LAB color space calibration.
        
        Args:
            frame: Input BGR frame.
        
        Returns:
            Binary mask of LAB range.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lower = np.array([self.settings[CalibrationMethod.LAB.name][k] for k in ['l_low', 'a_low', 'b_low']])
        upper = np.array([self.settings[CalibrationMethod.LAB.name][k] for k in ['l_high', 'a_high', 'b_high']])
        return cv2.inRange(lab, lower, upper)

    def _ycrcb_calibration(self, frame: np.ndarray) -> np.ndarray:
        """Apply YCrCb color space calibration.
        
        Args:
            frame: Input BGR frame.
        
        Returns:
            Binary mask of YCrCb range.
        """
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lower = np.array([self.settings[CalibrationMethod.YCRCB.name][k] for k in ['y_low', 'cr_low', 'cb_low']])
        upper = np.array([self.settings[CalibrationMethod.YCRCB.name][k] for k in ['y_high', 'cr_high', 'cb_high']])
        return cv2.inRange(ycrcb, lower, upper)

    def _adaptive_threshold_calibration(self, frame: np.ndarray) -> np.ndarray:
        """Apply adaptive threshold calibration.
        
        Args:
            frame: Input BGR frame.
        
        Returns:
            Binary image after adaptive thresholding.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(
            gray, self.settings[CalibrationMethod.ADAPTIVE_THRESHOLD.name]['max_val'],
            self.settings[CalibrationMethod.ADAPTIVE_THRESHOLD.name]['method'],
            self.settings[CalibrationMethod.ADAPTIVE_THRESHOLD.name]['thresh_type'],
            self.settings[CalibrationMethod.ADAPTIVE_THRESHOLD.name]['block_size'],
            self.settings[CalibrationMethod.ADAPTIVE_THRESHOLD.name]['c']
        )

    def _canny_edge_calibration(self, frame: np.ndarray) -> np.ndarray:
        """Apply Canny edge detection calibration.
        
        Args:
            frame: Input BGR frame.
        
        Returns:
            Binary image of edges.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(
            gray, self.settings[CalibrationMethod.CANNY_EDGE.name]['threshold1'],
            self.settings[CalibrationMethod.CANNY_EDGE.name]['threshold2'],
            apertureSize=self.settings[CalibrationMethod.CANNY_EDGE.name]['aperture_size']
        )

    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from JSON file or initialize with defaults.
        
        Returns:
            Dictionary of calibration settings.
        """
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    loaded = json.load(f)
                    return {**self.default_settings, **loaded}
            except Exception as e:
                print(f"Warning: Could not load settings ({e}), using defaults")
        return self.default_settings.copy()

    def save_settings(self) -> bool:
        """Save current settings to JSON file.
        
        Returns:
            True if saved successfully, False otherwise.
        """
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False

    def _method_selection_callback(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Handle mouse clicks for method selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, method in enumerate(CalibrationMethod):
                if 10 <= x <= 290 and 10 + i*40 <= y <= 40 + i*40:
                    if self.current_method != method:
                        self.current_method = method
                        self._setup_trackbars()
                        cv2.waitKey(50)
                    break

    def _create_method_buttons(self) -> None:
        """Create UI buttons for method selection."""
        self.method_bg = np.zeros((len(CalibrationMethod)*40 + 20, 300, 3), dtype=np.uint8)
        for i, method in enumerate(CalibrationMethod):
            color = (0, 255, 0) if method == self.current_method else (100, 100, 100)
            cv2.rectangle(self.method_bg, (10, 10 + i*40), (290, 40 + i*40), color, -1)
            cv2.putText(self.method_bg, method.name.replace('_', ' '), 
                       (20, 30 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow(f'{self.window_prefix}Method Selection', self.method_bg)

    def _setup_trackbars(self) -> None:
        """Initialize trackbars for the current calibration method."""
        cv2.destroyWindow(f"{self.window_prefix}Calibration")
        cv2.namedWindow(f"{self.window_prefix}Calibration")
        self._create_method_buttons()
        
        method_name = self.current_method.name
        params = self.settings[method_name]
        
        def create_trackbar(name: str, value: int, max_val: int) -> None:
            cv2.createTrackbar(name, f"{self.window_prefix}Calibration", value, max_val, lambda x: None)
        
        if self.current_method == CalibrationMethod.THRESHOLD:
            create_trackbar('Threshold', params['thresh'], 255)
            create_trackbar('Max Value', params['max_val'], 255)
        elif self.current_method == CalibrationMethod.RGB:
            for ch, prefix in [('r', 'R'), ('g', 'G'), ('b', 'B')]:
                create_trackbar(f'{prefix} Low', params[f'{ch}_low'], 255)
                create_trackbar(f'{prefix} High', params[f'{ch}_high'], 255)
        elif self.current_method == CalibrationMethod.HSV:
            create_trackbar('H Low', params['h_low'], 179)
            create_trackbar('H High', params['h_high'], 179)
            create_trackbar('S Low', params['s_low'], 255)
            create_trackbar('S High', params['s_high'], 255)
            create_trackbar('V Low', params['v_low'], 255)
            create_trackbar('V High', params['v_high'], 255)
        elif self.current_method == CalibrationMethod.LAB:
            for ch in ['l', 'a', 'b']:
                create_trackbar(f'{ch.upper()} Low', params[f'{ch}_low'], 255)
                create_trackbar(f'{ch.upper()} High', params[f'{ch}_high'], 255)
        elif self.current_method == CalibrationMethod.YCRCB:
            for ch in ['y', 'cr', 'cb']:
                create_trackbar(f'{ch.upper()} Low', params[f'{ch}_low'], 255)
                create_trackbar(f'{ch.upper()} High', params[f'{ch}_high'], 255)
        elif self.current_method == CalibrationMethod.ADAPTIVE_THRESHOLD:
            create_trackbar('Block Size', params['block_size'], 255)
            create_trackbar('C Value', params['c'], 10)
        elif self.current_method == CalibrationMethod.CANNY_EDGE:
            create_trackbar('Thresh1', params['threshold1'], 255)
            create_trackbar('Thresh2', params['threshold2'], 255)

    def _get_current_values(self) -> Dict[str, Any]:
        """Get current trackbar values for the active method.
        
        Returns:
            Dictionary of current parameter values.
        """
        method_name = self.current_method.name
        values = {}
        
        def get_trackbar(name: str) -> int:
            return cv2.getTrackbarPos(name, f"{self.window_prefix}Calibration")
        
        if self.current_method == CalibrationMethod.THRESHOLD:
            values.update(thresh=get_trackbar('Threshold'), max_val=get_trackbar('Max Value'))
        elif self.current_method == CalibrationMethod.RGB:
            values.update(
                r_low=get_trackbar('R Low'), r_high=get_trackbar('R High'),
                g_low=get_trackbar('G Low'), g_high=get_trackbar('G High'),
                b_low=get_trackbar('B Low'), b_high=get_trackbar('B High')
            )
        elif self.current_method == CalibrationMethod.HSV:
            values.update(
                h_low=get_trackbar('H Low'), h_high=get_trackbar('H High'),
                s_low=get_trackbar('S Low'), s_high=get_trackbar('S High'),
                v_low=get_trackbar('V Low'), v_high=get_trackbar('V High')
            )
        elif self.current_method == CalibrationMethod.LAB:
            values.update(
                l_low=get_trackbar('L Low'), l_high=get_trackbar('L High'),
                a_low=get_trackbar('A Low'), a_high=get_trackbar('A High'),
                b_low=get_trackbar('B Low'), b_high=get_trackbar('B High')
            )
        elif self.current_method == CalibrationMethod.YCRCB:
            values.update(
                y_low=get_trackbar('Y Low'), y_high=get_trackbar('Y High'),
                cr_low=get_trackbar('Cr Low'), cr_high=get_trackbar('Cr High'),
                cb_low=get_trackbar('Cb Low'), cb_high=get_trackbar('Cb High')
            )
        elif self.current_method == CalibrationMethod.ADAPTIVE_THRESHOLD:
            block_size = get_trackbar('Block Size')
            values.update(
                block_size=max(3, block_size if block_size % 2 == 1 else block_size - 1),
                c=get_trackbar('C Value'), method=self.settings[method_name]['method'],
                thresh_type=self.settings[method_name]['thresh_type'], max_val=255
            )
        elif self.current_method == CalibrationMethod.CANNY_EDGE:
            values.update(
                threshold1=get_trackbar('Thresh1'), threshold2=get_trackbar('Thresh2'), aperture_size=3
            )
        
        return values

    def calibrate(self) -> None:
        """Run the calibration interface with live video feed."""
        if self._suppress_ui:
            return
        
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.backup_settings = self.settings.copy()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.settings[self.current_method.name] = self._get_current_values()
            processed = self.process(frame)
            cv2.imshow(f'{self.window_prefix}Original', frame)
            cv2.imshow(f'{self.window_prefix}Processed', processed)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                if self.save_settings():
                    print("Settings saved successfully!")
                    self.backup_settings = self.settings.copy()
                else:
                    print("Failed to save settings!")
                    self.settings = self.backup_settings
            elif key == 27:  # ESC
                self.settings = self.backup_settings
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process a frame using current calibration settings.
        
        Args:
            frame: Input BGR frame.
        
        Returns:
            Processed binary image.
        """
        return self.calibration_methods[self.current_method](frame)

    def get_settings(self) -> Dict[str, Any]:
        """Get a copy of the current calibration settings.
        
        Returns:
            Dictionary of settings.
        """
        return self.settings.copy()

    def set_method(self, method: CalibrationMethod) -> None:
        """Set the current calibration method.
        
        Args:
            method: Calibration method to use.
        """
        if method in CalibrationMethod:
            self.current_method = method
            self._setup_trackbars()

    def __del__(self) -> None:
        """Clean up UI windows on instance destruction."""
        for win in [f"{self.window_prefix}{s}" for s in ["Calibration", "Method Selection", "Original", "Processed"]]:
            cv2.destroyWindow(win)
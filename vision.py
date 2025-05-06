import cv2
import numpy as np
from typing import List, Tuple, Optional, Union
from calibration import Calibrator
from config import (
    MIN_CONTOUR_AREA, MAX_INTENSITY, NUM_SEGMENTS, KERNEL_SIZE,
    ERODE_ITERATIONS, DILATE_ITERATIONS, GREEN_ERODE_ITERATIONS, GREEN_DILATE_ITERATIONS
)

def extract_roi(frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Extract a region of interest from the frame.
    
    Args:
        frame: Input image frame (BGR).
        x, y: Top-left corner coordinates of ROI.
        w, h: Width and height of ROI.
    
    Returns:
        ROI sub-image.
    """
    return frame[y:y+h, x:x+w]

def extract_min_area_contours(contours: List[np.ndarray], min_area: float = MIN_CONTOUR_AREA) -> List[np.ndarray]:
    """Filter contours by minimum area.
    
    Args:
        contours: List of contours from cv2.findContours.
        min_area: Minimum contour area threshold.
    
    Returns:
        List of contours with area greater than min_area.
    """
    return [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

def extract_left_mid_right_points(contour: np.ndarray) -> Tuple[Optional[int], Optional[Tuple[int, int]], Optional[int]]:
    """Extract leftmost, midpoint, and rightmost points from a contour.
    
    Args:
        contour: Single contour array.
    
    Returns:
        Tuple of (left_x, (mid_x, mid_y), right_x). Returns (None, None, None) if contour is invalid.
    """
    if contour is None:
        return None, None, None
    
    min_x_idx = contour[:, :, 0].argmin()
    max_x_idx = contour[:, :, 0].argmax()
    leftmost = tuple(contour[min_x_idx][0])
    rightmost = tuple(contour[max_x_idx][0])
    
    mid_x = (leftmost[0] + rightmost[0]) // 2
    mid_y = (leftmost[1] + rightmost[1]) // 2
    
    return leftmost[0], (mid_x, mid_y), rightmost[0]

def extract_min_dx_contour(contours: List[np.ndarray], prev_contour: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Select contour with minimal x-coordinate variation from previous contour.
    
    Args:
        contours: List of contours to filter.
        prev_contour: Previous contour for reference.
    
    Returns:
        Contour with least x variation, or None if no contours.
    """
    if not contours:
        return prev_contour
    if len(contours) == 1:
        return contours[0]
    
    if prev_contour is not None:
        prev_x, _, prev_w, _ = cv2.boundingRect(prev_contour)
        prev_center_x = prev_x + prev_w // 2
        return min(contours, key=lambda c: abs(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] // 2 - prev_center_x))
    
    return contours[0]

def extract_min_y_contours(contours: List[np.ndarray]) -> List[np.ndarray]:
    """Filter contours by minimum y-coordinate.
    
    Args:
        contours: List of contours to filter.
    
    Returns:
        List of contours with the minimum y-coordinate.
    """
    if not contours:
        return []
    if len(contours) == 1:
        return contours
    
    sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3], reverse=True)
    min_y = cv2.boundingRect(sorted_contours[0])[1] + cv2.boundingRect(sorted_contours[0])[3]
    return [c for c in sorted_contours if cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] == min_y]

def extract_min_intensity_contours(contours: List[np.ndarray], gray: np.ndarray, max_intensity: float = MAX_INTENSITY) -> List[np.ndarray]:
    """Filter contours by maximum intensity in grayscale image.
    
    Args:
        contours: List of contours to filter.
        gray: Grayscale image for intensity calculation.
        max_intensity: Maximum allowed intensity.
    
    Returns:
        List of contours with average intensity below max_intensity.
    """
    filtered = []
    mask = np.zeros(gray.shape[:2], dtype=np.uint8)
    
    for cnt in contours:
        mask.fill(0)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_val = cv2.mean(gray, mask=mask)[0]
        if mean_val < max_intensity:
            filtered.append(cnt)
    
    return filtered

def get_final_contour(
    gray: np.ndarray,
    contours: List[np.ndarray],
    prev_contour: Optional[np.ndarray],
    min_area: float,
    max_intensity: float,
    min_y: bool = False
) -> Tuple[Optional[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Apply a series of filters to select the final contour.
    
    Args:
        gray: Grayscale image for intensity filtering.
        contours: Input contours to filter.
        prev_contour: Previous contour for continuity.
        min_area: Minimum contour area threshold.
        max_intensity: Maximum intensity threshold.
        min_y: If True, filter by minimum y-coordinate.
    
    Returns:
        Tuple of (final_contour, min_y_contours, intensity_filtered_contours, area_filtered_contours).
    """
    area_filtered = extract_min_area_contours(contours, min_area)
    intensity_filtered = extract_min_intensity_contours(area_filtered, gray, max_intensity)
    y_filtered = extract_min_y_contours(intensity_filtered) if min_y else intensity_filtered
    final_contour = extract_min_dx_contour(y_filtered, prev_contour)
    
    return final_contour, y_filtered, intensity_filtered, area_filtered

def segment_contour(
    contour: np.ndarray,
    prev_segment_contours: List[Optional[np.ndarray]],
    num_segments: int,
    thresh: np.ndarray,
    roi: np.ndarray
) -> Tuple[List[int], List[Tuple[int, int]], List[int], List, List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Segment a contour into parts and extract left, mid, right points for each segment.
    
    Args:
        contour: Input contour to segment.
        prev_segment_contours: List of previous segment contours for continuity.
        num_segments: Number of segments to divide the contour into.
        thresh: Binary threshold image for contour detection.
        roi: Grayscale ROI image for intensity filtering.
    
    Returns:
        Tuple of (left_points, mid_points, right_points, segment_contours_data, rect_top_left, rect_bottom_right).
    """
    midpoints, left_points, right_points = [], [], []
    segment_contours_data = []
    rect_top_left, rect_bottom_right = [], []
    
    x, y, w, h = cv2.boundingRect(contour)
    segment_height = max(1, h // num_segments)
    
    for i in range(num_segments):
        top = y + i * segment_height
        bottom = top + segment_height
        
        mask = np.zeros_like(thresh)
        cv2.rectangle(mask, (x, top), (x + w, bottom), 255, -1)
        
        segment_contours, _ = cv2.findContours(
            cv2.bitwise_and(thresh, mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        selected_contour, _, intensity_contours, area_contours = get_final_contour(
            roi, segment_contours, prev_segment_contours[i], min_area=5, max_intensity=MAX_INTENSITY
        )
        
        segment_contours_data.append([segment_contours, area_contours, intensity_contours, None, selected_contour])
        prev_segment_contours[i] = selected_contour
        
        if selected_contour is not None:
            left, mid, right = extract_left_mid_right_points(selected_contour)
            left_points.append(left)
            midpoints.append(mid)
            right_points.append(right)
            rect_top_left.append((left, top))
            rect_bottom_right.append((right, bottom))
    
    return left_points, midpoints, right_points, segment_contours_data, rect_top_left, rect_bottom_right

def process_frame(
    frame: np.ndarray,
    state: dict,
    black_calibrator: Calibrator,
    green_calibrator: Calibrator,
    ser: Optional['serial.Serial']
) -> np.ndarray:
    """Process a single frame to detect black and green lines and navigate.
    
    Args:
        frame: Input BGR frame.
        state: Line follower state dictionary.
        black_calibrator: Calibrator for black line detection.
        green_calibrator: Calibrator for green line detection.
        ser: Serial connection to Arduino.
    
    Returns:
        Debug grid image for visualization.
    """
    from navigation import process_navigation_logic
    from debug import debug_choice
    
    roi = extract_roi(frame, 0, 300, 640, 180)
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    
    thresh = black_calibrator.process(roi)
    green = green_calibrator.process(roi)
    
    kernel = np.ones(KERNEL_SIZE, np.uint8)
    eroded_thresh = cv2.erode(thresh, kernel, iterations=ERODE_ITERATIONS)
    dilated_thresh = cv2.dilate(eroded_thresh, kernel, iterations=DILATE_ITERATIONS)
    dilated_thresh = cv2.erode(dilated_thresh, kernel, iterations=5)
    
    eroded_green = cv2.erode(green, kernel, iterations=GREEN_ERODE_ITERATIONS)
    dilated_green = cv2.dilate(eroded_green, kernel, iterations=GREEN_DILATE_ITERATIONS)
    
    contours, _ = cv2.findContours(dilated_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_contours, _ = cv2.findContours(dilated_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    largest_contour, _, intensity, area = get_final_contour(
        blurred, contours, state["prev_big_contour"], min_area=5000, max_intensity=50
    )
    _, _, _, green_area = get_final_contour(
        blurred, green_contours, state["prev_big_contour"], min_area=200, max_intensity=50
    )
    
    state["prev_big_contour"] = largest_contour
    
    left_points, mid_points, right_points, segment_contours, rect_pt1, rect_pt2 = segment_contour(
        largest_contour, state["prev_contours"], NUM_SEGMENTS, dilated_thresh, blurred
    )
    
    green_mids = []
    for contour in green_area or []:
        x, y, w, h = cv2.boundingRect(contour)
        mid_x, mid_y = x + w // 2, y + h // 2
        cv2.circle(roi, (mid_x, mid_y), 3, (0, 0, 255), -1)
        cv2.putText(
            roi, str(cv2.contourArea(contour)), (mid_x, mid_y),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA
        )
        green_mids.append((mid_x, mid_y))
    
    anchor = 3
    green_mids_below_diff = []
    if len(mid_points) >= 9:
        anchor_x, anchor_y = mid_points[anchor]
        pt1, pt2 = mid_points[anchor + 5], mid_points[anchor + 6]
        cv2.line(roi, pt1, pt2, (0, 0, 255), 2)
        x1, y1 = pt1
        x2, y2 = pt2
        
        for mid in green_mids:
            if mid[1] > anchor_y:
                x, y = mid
                cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
                green_mids_below_diff.append(cross)
    
    state = process_navigation_logic(
        state, roi, mid_points, left_points, right_points, anchor, "", ser, green_mids_below_diff
    )
    
    params = {
        "roi": roi,
        "contours": contours,
        "dx": largest_contour,
        "thresh": thresh,
        "eroded": eroded_thresh,
        "dilated": dilated_thresh,
        "min_area": area,
        "max_intensity": intensity,
        "frame": dilated_green,
        "segment_contours": [x[0] for x in segment_contours],
        "min_area_": [x[1] for x in segment_contours],
        "max_intensity_": [x[2] for x in segment_contours],
        "dx_": [x[4] for x in segment_contours],
        "lmr": [left_points, mid_points, right_points],
        "rect_pt1": rect_pt1,
        "rect_pt2": rect_pt2,
        **{k: state.get(k) for k in [
            "dr", "dl", "w", "wnext", "response", "process_left", "process_right",
            "imp_l_state", "imp_r_state", "ddleft", "ddright", "previous_leftmost",
            "previous_rightmost", "error"
        ]}
    }
    
    bool_params = {
        "frame": True,
        "roi": True,
        "thresh": True,
        "eroded": True,
        "dilated": True,
        "contours": True,
        "min_area": True,
        "max_intensity": True,
        "min_y": False,
        "dx": True,
        "segment_contours": True,
        "min_area_": True,
        "max_intensity_": True,
        "dx_": True,
        "lmr": True,
        "response": True,
        "navigation": True
    }
    
    data_params = {
        "contours": None,
        "dx": None,
        "thresh": None,
        "eroded": None,
        "dilated": None,
        "min_area": None,
        "max_intensity": None,
        "frame": None,
        "min_area_": None,
        "max_intensity_": None,
        "segment_contours": None,
        "dx_": None,
        "lmr": None,
        "navigation": None
    }
    
    return debug_choice(bool_params, params, data_params)
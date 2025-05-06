import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional,Any

def prepare_image(img: Optional[np.ndarray], label: str, size: Tuple[int, int], frame_number: int) -> np.ndarray:
    """Prepare an image for debug visualization with a label and frame number.
    
    Args:
        img: Input image (grayscale or BGR).
        label: Text label to display.
        size: Target size (width, height).
        frame_number: Frame number to display (e.g., 1, 2).
    
    Returns:
        Resized BGR image with label and number.
    """
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        img = np.zeros((*size, 3), dtype=np.uint8)
    elif len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    # Display frame number and label in red (e.g., "1: Default")
    display_label = f"{frame_number}: {label}"
    cv2.putText(img, display_label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    return img

def debug_choice(bool_params: Dict[str, bool], params: Dict[str, Any], data_params: Dict[str, Optional[np.ndarray]]) -> np.ndarray:
    """Create a debug visualization grid based on selected parameters.
    
    Args:
        bool_params: Dictionary indicating which debug images to include.
        params: Dictionary of debug data (images, contours, etc.).
        data_params: Dictionary of optional base images for contour drawing.
    
    Returns:
        Debug grid image.
    """
    debug_images = []
    width, height = 300, 150
    grid_cell_size = (width, height)
    frame_counter = 1  # Start numbering from 1
    
    # Initialize default surface, fallback to zero image if roi is invalid
    default_surface = params.get("roi")
    if default_surface is None or not isinstance(default_surface, np.ndarray) or default_surface.size == 0:
        default_surface = np.zeros((height, width, 3), dtype=np.uint8)
    
    debug_images.append(prepare_image(default_surface.copy(), "Default", grid_cell_size, frame_counter))
    frame_counter += 1
    
    for key, enabled in bool_params.items():
        if not enabled:
            continue
        if key == "navigation":
            # Use default_surface as base for navigation text
            img = data_params.get(key)
            if img is None or not isinstance(img, np.ndarray) or img.size == 0:
                img = default_surface.copy()
            h = 20
            for i, (param_key, label) in enumerate([
                ("dr", "dr"), ("dl", "dl"), ("w", "w"), ("wnext", "wnext"), ("response", "response"),
                ("pr", "pr"), ("pl", "pl"), ("left", "r"), ("right", "l"), ("ddleft", "ddleft"),
                ("ddright", "ddright"), ("previous_left", "prev_l"), ("previous_right", "prev_r"), ("error", "error")
            ]):
                if params.get(param_key) is not None:
                    cv2.putText(img, f"{label}: {params[param_key]}", (10, 10 + i*h),
                               cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
            debug_images.append(prepare_image(img, "Navigation", grid_cell_size, frame_counter))
            frame_counter += 1
        elif key == "lmr":
            # Ensure valid image for LMR drawing
            img = data_params.get(key)
            if img is None or not isinstance(img, np.ndarray) or img.size == 0:
                print(f"Warning: Invalid image for key '{key}', using default surface")
                img = default_surface.copy()
            else:
                img = img.copy()
            for i in range(len(params["lmr"][0])):
                cv2.rectangle(img, params["rect_pt1"][i], params["rect_pt2"][i], (255, 0, 0), 1)
                cv2.circle(img, params["lmr"][1][i], 5, (0, 255, 255), -1)
            debug_images.append(prepare_image(img, "LMR", grid_cell_size, frame_counter))
            frame_counter += 1
        elif key in ["segment_contours", "min_area_", "max_intensity_", "dx_"]:
            # Ensure valid image for contour drawing
            img = data_params.get(key)
            if img is None or not isinstance(img, np.ndarray) or img.size == 0:
                print(f"Warning: Invalid image for key '{key}', using default surface")
                img = default_surface.copy()
            else:
                img = img.copy()
            # Use distinct, bright colors for segment contours
            color = {
                "segment_contours": (255, 255, 0),  # Yellow: bright, distinct
                "min_area_": (255, 165, 0),        # Orange: unique, visible
                "max_intensity_": (128, 0, 128),   # Purple: distinct from others
                "dx_": (255, 0, 255)               # Magenta: bright, clear
            }[key]
            for cnts in params[key]:
                cv2.drawContours(img, [cnts] if key == "dx_" else cnts, -1, color, 4)
            debug_images.append(prepare_image(img, key.replace('_', ' ').title(), grid_cell_size, frame_counter))
            frame_counter += 1
        elif key == "dx":
            # Ensure valid image for dx contour
            img = data_params.get(key)
            if img is None or not isinstance(img, np.ndarray) or img.size == 0:
                print(f"Warning: Invalid image for key '{key}', using default surface")
                img = default_surface.copy()
            else:
                img = img.copy()
            # White for dx: already clear
            cv2.drawContours(img, [params[key]], -1, (255, 255, 255), 4)
            debug_images.append(prepare_image(img, "DX", grid_cell_size, frame_counter))
            frame_counter += 1
        elif key in ["contours", "min_area", "max_intensity"]:
            # Ensure valid image for contour drawing
            img = data_params.get(key)
            if img is None or not isinstance(img, np.ndarray) or img.size == 0:
                print(f"Warning: Invalid image for key '{key}', using default surface")
                img = default_surface.copy()
            else:
                img = img.copy()
            # Use distinct, bright colors for main contours
            color = {
                "contours": (0, 255, 255),     # Cyan: bright blue, visible
                "min_area": (255, 0, 0),       # Red: pure, distinct
                "max_intensity": (0, 255, 0)   # Green: pure, high contrast
            }[key]
            cv2.drawContours(img, params[key], -1, color, 4)
            debug_images.append(prepare_image(img, key.replace('_', ' ').title(), grid_cell_size, frame_counter))
            frame_counter += 1
        elif key == "frame":
            # Explicitly handle green lines (dilated_green from vision.py)
            img = params.get(key)
            if img is None or not isinstance(img, np.ndarray) or img.size == 0:
                print("Warning: Invalid green lines image, using default surface")
                img = default_surface.copy()
            debug_images.append(prepare_image(img, "Green Lines", grid_cell_size, frame_counter))
            frame_counter += 1
        elif key in ["roi", "thresh", "eroded", "dilated"]:
            # Direct params[key] should be valid, but prepare_image handles None
            debug_images.append(prepare_image(params[key], key.capitalize(), grid_cell_size, frame_counter))
            frame_counter += 1
    
    if not debug_images:
        print("Debug: No images selected for display.")
        cv2.destroyWindow("Debug Grid")
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    num_images = len(debug_images)
    num_rows = (num_images + 4 - 1) // 4
    debug_images.extend([np.zeros((height, width, 3), dtype=np.uint8)] * ((num_rows * 4) - num_images))
    
    image_rows = [cv2.hconcat(debug_images[i*4:(i+1)*4]) for i in range(num_rows)]
    return cv2.vconcat(image_rows)
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from config import MAX_RECT_WIDTH, MIN_RIGHT_DEVIATION, MIN_LEFT_DEVIATION, MIN_DDERROR
from arduino import send_to_arduino

def create_line_follower_state() -> Dict[str, Any]:
    """Initialize the state dictionary for line following.
    
    Returns:
        Dictionary with initial state values for contour tracking and navigation.
    """
    return {
        'prev_big_contour': None,
        'prev_contours': [None] * 10,
        'previous_leftmost': 0,
        'previous_rightmost': 0,
        'process_right': False,
        'process_left': False,
        'imp_l_state': 0,
        'imp_r_state': 0,
        'previous_width': 0,
        'w': 0,
        'wnext': 0,
        'ddleft': 0,
        'ddright': 0,
        'dl': 0,
        'dr': 0,
        'right': 0,
        'left': 0,
        'error': None,
        'response': "",
        'max_rect_width': MAX_RECT_WIDTH,
        'min_right_deviation': MIN_RIGHT_DEVIATION,
        'min_left_deviation': MIN_LEFT_DEVIATION,
        'min_dderror': MIN_DDERROR
    }

def decide(green_mids_below_diff: List[float]) -> Optional[str]:
    """Determine navigation direction based on green markers.
    
    Args:
        green_mids_below_diff: List of cross-product differences for green markers.
    
    Returns:
        Direction ('l', 'r', 'u') or None.
    """
    if len(green_mids_below_diff) == 2:
        return "u"
    if len(green_mids_below_diff) == 1:
        return "l" if green_mids_below_diff[0] > 0 else "r"
    return None

def handle_junction(ser: Optional['serial.Serial'], green_mids_below_diff: List[float], state: Dict[str, Any]) -> None:
    """Handle junction navigation based on green markers.
    
    Args:
        ser: Serial connection to Arduino.
        green_mids_below_diff: List of cross-product differences for green markers.
        state: Line follower state dictionary.
    """
    result = decide(green_mids_below_diff)
    if result == "r":
        print("Junction right")
        state["response"] = "JR"
        send_to_arduino(ser, "JR")
    elif result == "l":
        print("Junction left")
        state["response"] = "JL"
        send_to_arduino(ser, "JL")
    elif result == "u":
        print("Junction U-turn")
        state["response"] = "U"
        send_to_arduino(ser, "U")
    else:
        print("Junction forward")
        state["response"] = "JF"
        send_to_arduino(ser, "JF")

def process_navigation_logic(
    state: Dict[str, Any],
    roi: np.ndarray,
    midpoints: List[Tuple[int, int]],
    leftmost_points: List[int],
    rightmost_points: List[int],
    anchor: int,
    data_text: str,
    ser: Optional['serial.Serial'],
    green_mids_below_diff: List[float]
) -> Dict[str, Any]:
    """Process navigation logic for line following.
    
    Args:
        state: Line follower state dictionary.
        roi: Region of interest image.
        midpoints: List of midpoint coordinates.
        leftmost_points: List of leftmost x-coordinates.
        rightmost_points: List of rightmost x-coordinates.
        anchor: Index of anchor segment.
        data_text: Debug text (unused).
        ser: Serial connection to Arduino.
        green_mids_below_diff: List of cross-product differences for green markers.
    
    Returns:
        Updated state dictionary.
    """
    wait_left = state.get('wait_left', False)
    wait_right = state.get('wait_right', False)
    
    if leftmost_points and len(leftmost_points) > (anchor + 1):
        dl = -(leftmost_points[anchor] - state['previous_leftmost'])
        dr = rightmost_points[anchor] - state['previous_rightmost']
        state["dl"], state["dr"] = dl, dr
        
        w = abs(rightmost_points[anchor] - leftmost_points[anchor])
        w_next = abs(rightmost_points[anchor + 1] - leftmost_points[anchor + 1])
        w_turn_anchor = abs(rightmost_points[anchor + 5] - leftmost_points[anchor + 5])
        w_first_anchor = abs(rightmost_points[0] - leftmost_points[0])
        state["w"], state["wnext"] = w, w_next
        
        anchor_width = w > state['max_rect_width']
        next_anchor_width = w_next > state['max_rect_width']
        turn_anchor = w_turn_anchor > state['max_rect_width']
        first_anchor = w_first_anchor > state['max_rect_width']
        
        if anchor_width:
            right_turn = dr > state['min_right_deviation']
            left_turn = dl > state['min_left_deviation']
            junction = right_turn and left_turn
            
            if junction:
                handle_junction(ser, green_mids_below_diff, state)
            elif right_turn and not state['process_left'] and not state['process_right'] and state['previous_width'] < state['max_rect_width']:
                state["response"] = "Possible right turn"
                state['imp_l_state'] = state['previous_leftmost']
                state['process_right'] = True
            elif left_turn and not state['process_right'] and not state['process_left'] and state['previous_width'] < state['max_rect_width']:
                state["response"] = "Possible left turn"
                state['imp_r_state'] = state['previous_rightmost']
                state['process_left'] = True
            
            if next_anchor_width:
                if state['process_right'] and not state['process_left']:
                    state['ddleft'] = state['imp_l_state'] - leftmost_points[anchor]
                    if abs(state['ddleft']) > state['min_dderror']:
                        handle_junction(ser, green_mids_below_diff, state)
                    else:
                        result = decide(green_mids_below_diff)
                        if result == "r":
                            state["response"] = "RC"
                            print("Right color")
                            send_to_arduino(ser, "RC")
                        elif not wait_left:
                            print("Waiting for right")
                            state['wait_right'] = True
                    state['process_right'] = False
                if state['process_left'] and not state['process_right']:
                    state['ddright'] = state['imp_r_state'] - rightmost_points[anchor]
                    if abs(state['ddright']) > state['min_dderror']:
                        handle_junction(ser, green_mids_below_diff, state)
                    else:
                        result = decide(green_mids_below_diff)
                        if result == "l":
                            state["response"] = "LEFT after green"
                            print("Left color")
                            send_to_arduino(ser, "LC")
                        elif not wait_right:
                            print("Waiting for left")
                            state['wait_left'] = True
                    state['process_left'] = False
        
        if turn_anchor and wait_right and not wait_left:
            if first_anchor:
                print("Normal right")
                state["response"] = "NR"
                send_to_arduino(ser, "NR")
            else:
                print("Forward right")
                state["response"] = "FR"
                send_to_arduino(ser, "FR")
            state['wait_right'] = False
        if turn_anchor and wait_left and not wait_right:
            if first_anchor:
                print("Normal left")
                state["response"] = "NL"
                send_to_arduino(ser, "NL")
            else:
                print("Forward left")
                state["response"] = "FL"
                send_to_arduino(ser, "FL")
            state['wait_left'] = False
        
        center_x = roi.shape[1] // 2
        error = midpoints[anchor][0] - center_x
        state["error"] = error
        
        state['previous_rightmost'] = rightmost_points[anchor]
        state['previous_leftmost'] = leftmost_points[anchor]
        state['previous_width'] = w
        
        send_to_arduino(ser, f"PID:{error}")
    
    return state
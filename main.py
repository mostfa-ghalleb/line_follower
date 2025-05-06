from calibration import Calibrator
import cv2
import time
from vision import process_frame
from navigation import create_line_follower_state
from arduino import connect_to_arduino

def main() -> None:
    """Main function to run the line-following robot."""
    print("Starting calibration for green...")
    green = Calibrator(name="green", settings_file="green.json")
    green.calibrate()
    print("Green calibration completed.")

    print("Starting calibration for black...")
    black = Calibrator(name="black", settings_file="black.json")
    black.calibrate()
    print("Black calibration completed.")
    
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    
    ser = connect_to_arduino()
    time.sleep(1)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (300*4, 150*4))
    
    state = create_line_follower_state()
    prev_time = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            final_grid = process_frame(frame, state, black, green, ser)
            
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(final_grid, f"FPS: {int(fps)}", (30, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            out.write(final_grid)
            cv2.imshow("Debug Grid", final_grid)
            
            if cv2.waitKey(1) == 27:  # ESC key
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        out.release()
        print("Program terminated")

if __name__ == "__main__":
    main()
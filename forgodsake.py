import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
from picamera2 import Picamera2
from movement import MotorController, find_line_position

# Pin configuration
ENA = 12  # Enable A
ENB = 13  # Enable B
IN1 = 5   # Motor A - Input 1
IN2 = 6   # Motor A - Input 2
IN3 = 19  # Motor B - Input 3
IN4 = 16  # Motor B - Input 4
encR = 25 # Right encoder
encL = 8  # Left encoder

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setup([ENA, ENB, IN1, IN2, IN3, IN4], GPIO.OUT)
GPIO.setup(encR, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(encL, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# PWM setup for speed control
pwm_a = GPIO.PWM(ENA, 1000)  # Frequency: 1kHz
pwm_b = GPIO.PWM(ENB, 1000)
pwm_a.start(0)
pwm_b.start(0)

def compute_angle(pt1, pt2, pt3):
    vec1 = pt1 - pt2
    vec2 = pt3 - pt2
    dot_prod = np.dot(vec1, vec2)
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)
    if mag1 == 0 or mag2 == 0:
        return 180
    cos_theta = dot_prod / (mag1 * mag2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))
    return angle

def get_contour_color(hsv, contour):
    """Determine the dominant color within a contour"""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    
    color_counts = {}
    # Check for each of our target colors
    for color, ranges in color_ranges.items():
        color_mask = np.zeros_like(mask)
        for lower, upper in ranges:
            color_mask |= cv2.inRange(hsv, lower, upper)
        color_count = cv2.countNonZero(cv2.bitwise_and(color_mask, mask))
        color_counts[color] = color_count
    
    if color_counts:
        dominant_color = max(color_counts, key=color_counts.get)
        if color_counts[dominant_color] > 50:  # Minimum pixel threshold
            return dominant_color
    return None  # Return None instead of 'unknown' to filter out non-colored objects

def create_color_mask(hsv):
    """Create a mask that only shows regions of our target colors"""
    color_mask = np.zeros((hsv.shape[0], hsv.shape[1]), dtype=np.uint8)
    
    # Combine all target color ranges
    for color, ranges in color_ranges.items():
        for lower, upper in ranges:
            color_mask |= cv2.inRange(hsv, lower, upper)
    
    return color_mask

def detect_object(frame, hsv, min_area=2000):
    height, width = frame.shape[:2]
    
    # Use frame's top 1/3th for object detection
    object_roi_start = 0
    object_roi_end = int(height * 2/3)
    
    # Extract object detection ROI
    frame_roi = frame[object_roi_start:object_roi_end, :]
    hsv_roi = hsv[object_roi_start:object_roi_end, :]
    
    # Create a debug image
    debug_image = frame_roi.copy()
    
    # Create a mask that only shows our target colors
    color_mask = create_color_mask(hsv_roi)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the color mask
    contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and shape
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            # Filter out contours that are too elongated (likely track parts)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / (min(w, h) + 0.01)  # Avoid division by zero
            if aspect_ratio < 5:  # Not too elongated
                valid_contours.append(contour)
    
    if not valid_contours:
        return None, None, None, debug_image
    
    # Sort contours by area (largest first)
    main_contour = max(valid_contours, key=cv2.contourArea)
    area = cv2.contourArea(main_contour)
    
    # Determine object color
    object_color = get_contour_color(hsv_roi, main_contour)
    
    # If no valid color detected, return None
    if object_color is None:
        return None, None, None, debug_image
    
    # Draw the main contour on debug image
    cv2.drawContours(debug_image, [main_contour], -1, (0, 0, 255), 2)
    
    # Calculate shape properties
    epsilon = 0.02 * cv2.arcLength(main_contour, True)
    approx = cv2.approxPolyDP(main_contour, epsilon, True)
    num_vertices = len(approx)
    approx_points = [pt[0] for pt in approx]
    
    # Draw approximated polygon on debug image
    for i, pt in enumerate(approx_points):
        cv2.circle(debug_image, tuple(pt), 5, (255, 0, 0), -1)
        cv2.putText(debug_image, str(i), tuple(pt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Calculate angles for shape detection
    angles = []
    for i in range(len(approx_points)):
        pt1 = np.array(approx_points[(i-1) % len(approx_points)])
        pt2 = np.array(approx_points[i])
        pt3 = np.array(approx_points[(i+1) % len(approx_points)])
        angle = compute_angle(pt1, pt2, pt3)
        angles.append(angle)
    
    response = None
    
    # Determine shape based on vertices and angles
    if num_vertices == 3:
        # Check if it's a valid triangle (no extremely sharp angles)
        min_angle = min(angles) if angles else 0
        
        # For triangles, we expect all angles to be > 30 degrees
        if min_angle > 30:
            response = "triangle"
        else:
            # Might be an arrow but incorrectly detected as triangle
            response = "triangle"  # Keep it as triangle as requested
    elif num_vertices == 4:
        x, y, w, h = cv2.boundingRect(np.array(approx_points))
        aspect_ratio = w / float(h)
        response = "square" if 0.95 <= aspect_ratio <= 1.05 else "rectangle"
    elif num_vertices == 5:
        response = "pentagon"
    elif num_vertices == 6:
        response = "hexagon"
    else:
        # Check for arrow shape (sharp point, concave shape)
        if angles and min(angles) < 50:
            # Additional checks for arrow shape
            hull = cv2.convexHull(main_contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            if 0.7 <= solidity <= 0.9:
                # Calculate arrow direction
                tip_index = angles.index(min(angles))
                tip_point = approx_points[tip_index]
                
                # Calculate centroid
                M = cv2.moments(main_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Calculate direction
                    dx = tip_point[0] - cx
                    dy = tip_point[1] - cy
                    
                    # Draw the arrow direction on debug image
                    cv2.line(debug_image, (cx, cy), tuple(tip_point), (0, 255, 255), 2)
                    cv2.circle(debug_image, tuple(tip_point), 7, (0, 255, 255), -1)
                    cv2.circle(debug_image, (cx, cy), 5, (255, 0, 255), -1)
                    
                    if abs(dx) > abs(dy):
                        direction = "right" if dx > 0 else "left"
                    else:
                        direction = "up" if dy < 0 else "down"
                    response = f"arrow pointing {direction}"
        else:
            # Check for circle
            perimeter = cv2.arcLength(main_contour, True)
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            
            # Display shape metrics
            cv2.putText(debug_image, f"Vertices: {num_vertices}", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(debug_image, f"Circularity: {circularity:.2f}", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            if circularity > 0.8:
                response = "circle"
            else:
                # Changed unknown shape to pacman as requested
                response = "pacman"
    
    # Add shape label to debug image
    if response is not None:
        detection_text = f"{response} ({object_color})"
        cv2.putText(debug_image, detection_text, (10, debug_image.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Adjust contour coordinates to match the full frame
    if main_contour is not None:
        main_contour = main_contour.copy()
        for point in main_contour:
            point[0][1] += object_roi_start
    
    return response, main_contour, object_color, debug_image

# Initialize motor controller
motor_controller = MotorController(pwm_a, pwm_b, IN1, IN2, IN3, IN4)

# Define HSV ranges
black_lower = np.array([0, 0, 0])
black_upper = np.array([180, 255, 50])

color_ranges = {
    'red': [(np.array([0, 100, 100]), np.array([10, 255, 255])), 
            (np.array([170, 100, 100]), np.array([180, 255, 255]))],
    'blue': [(np.array([100, 100, 100]), np.array([130, 255, 255]))],  # Expanded blue range
    'green': [(np.array([40, 50, 50]), np.array([80, 255, 255]))],
    'yellow': [(np.array([20, 100, 100]), np.array([35, 255, 255]))]   # Expanded yellow range
}

# Colors to track
chosen_colors = ['yellow']
object_colors = ['red', 'green', 'blue']

# Speed settings - do not change as requested
normal_speed = 42  # Default speed
turn_speed = 63    # Turn speed

# Initialize Camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))  # Increased resolution
picam2.start()

# State variables
following_color_track = False
current_track_color = 'black'
current_object_info = None
last_detected_time = 0
object_display_duration = 3.0  # Show object detection for 3 seconds

# Create window with larger size
cv2.namedWindow("Robot Vision", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Robot Vision", 800, 600)  # Make the window bigger

try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width = frame.shape[:2]
        
        # Define line tracking ROI boundaries
        line_roi_start = int(height * 3/4)  # Lower 1/4th of the image for line tracking
        
        # Create visualization display
        display = frame.copy()
        
        # Create track masks
        black_mask = cv2.inRange(hsv, black_lower, black_upper)
        color_mask = np.zeros_like(black_mask, dtype=np.uint8)
        individual_color_masks = {}
        
        for color in chosen_colors:
            color_mask_for_this_color = np.zeros_like(black_mask, dtype=np.uint8)
            for lower, upper in color_ranges[color]:
                color_mask_for_this_color |= cv2.inRange(hsv, lower, upper)
            individual_color_masks[color] = color_mask_for_this_color
            color_mask |= color_mask_for_this_color
        
        # Extract line tracking ROI
        black_roi = black_mask[line_roi_start:, :]
        color_roi = color_mask[line_roi_start:, :]
        
        # Process masks for line tracking
        kernel = np.ones((3, 3), np.uint8)
        black_processed = cv2.morphologyEx(black_roi, cv2.MORPH_OPEN, kernel)
        color_processed = cv2.morphologyEx(color_roi, cv2.MORPH_OPEN, kernel)
        
        # Find line positions
        black_line_x, black_cy = find_line_position(black_processed)
        color_line_x, color_cy = find_line_position(color_processed)
        
        # Object detection - run every frame
        object_response, object_contour, object_color, debug_image = detect_object(frame, hsv, min_area=1500)
        
        # Update object info if new object is detected
        current_time = time.time()
        if object_response is not None and object_color in object_colors:
            # Only update if this is a new detection or different from previous
            if current_object_info is None or current_object_info[0] != object_response or current_object_info[1] != object_color:
                current_object_info = (object_response, object_color, object_contour)
                print(f"Detected: {object_response} ({object_color})")
                last_detected_time = current_time
        
        # Line tracking and object detection separation
        # Draw a dividing line to show line tracking area
        cv2.line(display, (0, line_roi_start), (width, line_roi_start), (255, 255, 255), 1)
        cv2.putText(display, "Line Tracking Area", (10, line_roi_start - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # If we have a recently detected object, display it
        if current_object_info is not None and current_time - last_detected_time < object_display_duration:
            obj_response, obj_color, obj_contour = current_object_info
            
            # Draw a highlight box in the top portion of the screen
            cv2.rectangle(display, (10, 10), (width - 10, 50), (0, 0, 0), -1)
            cv2.rectangle(display, (10, 10), (width - 10, 50), (0, 255, 255), 2)
            
            # Display object info in the highlighted area
            detection_text = f"DETECTED: {obj_response.upper()} ({obj_color.upper()})"
            cv2.putText(display, detection_text, (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw the object contour if available
            if obj_contour is not None:
                cv2.drawContours(display, [obj_contour], -1, (0, 255, 0), 2)
                
                # Label near the contour
                x, y, w, h = cv2.boundingRect(obj_contour)
                cv2.putText(display, f"{obj_response} ({obj_color})",
                          (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Line tracking logic - continue even when object is detected
        if following_color_track:
            # Try to find colored line
            color_track_found = False
            for color in chosen_colors:
                color_roi_mask = individual_color_masks[color][line_roi_start:, :]
                color_roi_processed = cv2.morphologyEx(color_roi_mask, cv2.MORPH_OPEN, kernel)
                color_line_x, color_cy = find_line_position(color_roi_processed)
                
                if color_line_x is not None:
                    color_track_found = True
                    line_y = line_roi_start + color_cy
                    # Show tracking point
                    cv2.circle(display, (color_line_x, line_y), 5, (0, 255, 0), -1)
                    
                    # Follow the colored line
                    error = color_line_x - (width // 2)
                    if abs(error) < 40:
                        motor_controller.move_forward(normal_speed)
                    elif error > 40:
                        motor_controller.turn_right(turn_speed)
                        time.sleep(0.005)
                    else:
                        motor_controller.turn_left(turn_speed)
                        time.sleep(0.005)
                    break
            
            # Check if we should switch back to black line tracking
            if not color_track_found and black_line_x is not None:
                following_color_track = False
                current_track_color = 'black'
                print("Switching back to black line tracking")
        else:
            # Check if colored line is detected
            colored_line_detected = False
            for color in chosen_colors:
                color_roi_mask = individual_color_masks[color][line_roi_start:, :]
                color_roi_processed = cv2.morphologyEx(color_roi_mask, cv2.MORPH_OPEN, kernel)
                color_line_x, color_cy = find_line_position(color_roi_processed)
                
                if color_line_x is not None:
                    colored_line_detected = True
                    following_color_track = True
                    current_track_color = color
                    line_y = line_roi_start + color_cy
                    print(f"Switching to {color} line tracking")
                    break
            
            # If no colored line or already following one, follow black line
            if not colored_line_detected:
                if black_line_x is not None:
                    line_y = line_roi_start + black_cy
                    # Show tracking point
                    cv2.circle(display, (black_line_x, line_y), 5, (0, 255, 0), -1)
                    
                    # Follow the black line
                    error = black_line_x - (width // 2)
                    if abs(error) < 40:
                        motor_controller.move_forward(normal_speed)
                    elif error > 40:
                        motor_controller.turn_right(turn_speed)
                        time.sleep(0.005)
                    else:
                        motor_controller.turn_left(turn_speed)
                        time.sleep(0.005)
                else:
                    # No line found, reverse briefly and stop
                    motor_controller.reverse(normal_speed)
                    time.sleep(0.50)
                    motor_controller.stop()
                    time.sleep(0.1)
        
        # Add tracking status information
        status_text = f"Tracking: {current_track_color} line"
        cv2.putText(display, status_text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Show the display
        cv2.imshow("Robot Vision", display)
        
        # Exit on 'q' key
        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    motor_controller.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()
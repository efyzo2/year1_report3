from picamera2 import Picamera2
import cv2
import numpy as np

def compute_angle(pt1, pt2, pt3):
    """Compute the angle at pt2 given three points."""
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

# Initialize camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

previous_response = None

try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR
        
        # Preprocessing pipeline
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # Minimal blur for sharp contours
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 9, 3)  # High sensitivity
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))  # Minimal closing
        
        # Contour detection
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        response = None
        
        if contours is not None and hierarchy is not None:
            # Find outer contours (parentless)
            outer_contours = [(c, i) for i, c in enumerate(contours) 
                             if hierarchy[0][i][3] == -1]
            
            if outer_contours:
                # Select largest outer contour
                _, main_idx = max(outer_contours, 
                                 key=lambda x: cv2.contourArea(x[0]))
                
                # Collect all inner contours (children)
                inner_contours = []
                child_idx = hierarchy[0][main_idx][2]
                while child_idx != -1:
                    inner_contours.append((contours[child_idx], child_idx))
                    child_idx = hierarchy[0][child_idx][0]  # Next sibling
                
                # Select the largest inner contour
                if inner_contours:
                    main_contour, inner_idx = max(inner_contours, 
                                                 key=lambda x: cv2.contourArea(x[0]))
                    area = cv2.contourArea(main_contour)
                    
                    if area > 1500:
                        epsilon = 0.02 * cv2.arcLength(main_contour, True)
                        approx = cv2.approxPolyDP(main_contour, epsilon, True)
                        approx = [pt[0] for pt in approx]  # Convert to list of [x, y]
                        
                        # Check for arrow by analyzing angles
                        angles = []
                        for i in range(len(approx)):
                            pt1 = np.array(approx[(i-1) % len(approx)])
                            pt2 = np.array(approx[i])
                            pt3 = np.array(approx[(i+1) % len(approx)])
                            angle = compute_angle(pt1, pt2, pt3)
                            angles.append(angle)
                        
                        min_angle = min(angles)
                        if min_angle < 50:  # Threshold for arrow tip
                            # It's an arrow
                            tip_index = angles.index(min_angle)
                            tip_point = approx[tip_index]
                            M = cv2.moments(main_contour)
                            if M['m00'] != 0:
                                cx = int(M['m10'] / M['m00'])
                                cy = int(M['m01'] / M['m00'])
                            else:
                                cx, cy = 0, 0
                            dx = tip_point[0] - cx
                            dy = tip_point[1] - cy
                            if abs(dx) > abs(dy):
                                direction = "right" if dx > 0 else "left"
                            else:
                                direction = "up" if dy < 0 else "down"
                            response = f"arrow pointing {direction}"
                        else:
                            # No arrow, classify as shape
                            shape = "pacman"  # Default shape
                            vertices = len(approx)
                            if vertices == 3:
                                shape = "triangle"
                            elif vertices == 4:
                                x, y, w, h = cv2.boundingRect(np.array(approx))
                                aspect_ratio = w / float(h)
                                shape = "square" if 0.95 <= aspect_ratio <= 1.05 else "rectangle"
                            elif vertices == 5:
                                shape = "pentagon"
                            elif vertices == 6:
                                shape = "hexagon"
                            else:
                                perimeter = cv2.arcLength(main_contour, True)
                                circularity = (4 * np.pi * area) / (perimeter ** 2)
                                if circularity > 0.75:
                                    shape = "circle"

                            response = shape
                        
                        # Update display only when changed and response exists
                        if response and response != previous_response:
                            print(f"Detected: {response}")
                            previous_response = response
                        
                        # Visual feedback: highlight largest inner contour in red
                        cv2.drawContours(frame, [main_contour], -1, (0, 0, 255), 2)  # Red for largest inner contour
                        if response:
                            cv2.putText(frame, response, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()

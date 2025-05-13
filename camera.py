import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
from picamera2 import Picamera2

# Pin configuration
ENA = 12  # Enable A
ENB = 13  # Enable B
IN1 = 5  # Motor A - Input 1
IN2 = 6  # Motor A - Input 2
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

def move_forward(speed=50):
    GPIO.output([IN1], GPIO.HIGH)
    GPIO.output([IN2], GPIO.LOW)
    GPIO.output([IN3], GPIO.LOW)
    GPIO.output([IN4], GPIO.HIGH)
    pwm_a.ChangeDutyCycle(speed)
    pwm_b.ChangeDutyCycle(speed)

def turn_left(speed=80):
    GPIO.output([IN1], GPIO.HIGH)
    GPIO.output([IN2], GPIO.LOW)
    GPIO.output([IN3], GPIO.HIGH)
    GPIO.output([IN4], GPIO.LOW)
    pwm_a.ChangeDutyCycle(speed)
    pwm_b.ChangeDutyCycle(speed)

def turn_right(speed=80):
    GPIO.output([IN1], GPIO.LOW)
    GPIO.output([IN2], GPIO.HIGH)
    GPIO.output([IN3], GPIO.LOW)
    GPIO.output([IN4], GPIO.HIGH)
    pwm_a.ChangeDutyCycle(speed)
    pwm_b.ChangeDutyCycle(speed)

def stop():
    GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)
    pwm_a.ChangeDutyCycle(0)
    pwm_b.ChangeDutyCycle(0)

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    temp, binary = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
    return binary

def find_line_position(binary_frame):
    contours, temp = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            return int(M["m10"] / M["m00"])
    return None

# Initialize Camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (320, 240)}))
picam2.start()

# Main loop
try:
    while True:
        image = picam2.capture_array()
        binary = process_frame(image)
        line_position = find_line_position(binary)
        frame_center = 160

        if line_position is not None:
            error = line_position - frame_center
            if abs(error) < 20:
                move_forward(50)
                time.sleep(0.01)
            elif error > 20:
                turn_right(50)
                time.sleep(0.01)
            else:
                turn_left(50)
                time.sleep(0.01)
        else:
            stop()

except KeyboardInterrupt:
    stop()
    GPIO.cleanup()


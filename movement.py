import cv2
import numpy as np
import RPi.GPIO as GPIO

class MotorController:
    def __init__(self, pwm_a, pwm_b, IN1, IN2, IN3, IN4):
        self.pwm_a = pwm_a
        self.pwm_b = pwm_b
        self.IN1 = IN1
        self.IN2 = IN2
        self.IN3 = IN3
        self.IN4 = IN4

    def move_forward(self, speed=50):
        GPIO.output([self.IN1], GPIO.HIGH)
        GPIO.output([self.IN2], GPIO.LOW)
        GPIO.output([self.IN3], GPIO.LOW)
        GPIO.output([self.IN4], GPIO.HIGH)
        self.pwm_a.ChangeDutyCycle(speed)
        self.pwm_b.ChangeDutyCycle(speed)

    def turn_left(self, speed=80):
        GPIO.output([self.IN1], GPIO.HIGH)
        GPIO.output([self.IN2], GPIO.LOW)
        GPIO.output([self.IN3], GPIO.HIGH)
        GPIO.output([self.IN4], GPIO.LOW)
        self.pwm_a.ChangeDutyCycle(speed)
        self.pwm_b.ChangeDutyCycle(speed)

    def turn_right(self, speed=80):
        GPIO.output([self.IN1], GPIO.LOW)
        GPIO.output([self.IN2], GPIO.HIGH)
        GPIO.output([self.IN3], GPIO.LOW)
        GPIO.output([self.IN4], GPIO.HIGH)
        self.pwm_a.ChangeDutyCycle(speed)
        self.pwm_b.ChangeDutyCycle(speed)

    def reverse(self, speed=50):
        GPIO.output([self.IN1], GPIO.LOW)
        GPIO.output([self.IN2], GPIO.HIGH)
        GPIO.output([self.IN3], GPIO.HIGH)
        GPIO.output([self.IN4], GPIO.LOW)
        self.pwm_a.ChangeDutyCycle(speed)
        self.pwm_b.ChangeDutyCycle(speed)

    def stop(self):
        GPIO.output([self.IN1, self.IN2, self.IN3, self.IN4], GPIO.LOW)
        self.pwm_a.ChangeDutyCycle(0)
        self.pwm_b.ChangeDutyCycle(0)

def find_line_position(binary_frame, min_area=100):
    """
    Find the centroid of the line in the binary frame.
    Returns (x, y_roi) where y_roi is relative to the ROI.
    """
    moments = cv2.moments(binary_frame)
    if moments["m00"] > min_area:  # Ensure sufficient area to filter noise
        cx = int(moments["m10"] / moments["m00"])
        cy_roi = int(moments["m01"] / moments["m00"])
        return cx, cy_roi
    return None, None
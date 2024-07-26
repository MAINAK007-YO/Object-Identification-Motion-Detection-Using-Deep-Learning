import threading
import cv2
import imutils
import smtplib
import supervision as sv
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from ultralytics import YOLO
import requests
import os

class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()

URL = "http://192.168.58.140"
cap = VideoCapture(URL + ":81/stream")

# Size of the output camera (adjust if needed)
cap.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initializing the 1st frame with which we want to differentiate the current frame
_, start_frame = cap.read()
start_frame = imutils.resize(start_frame, width=500)
start_frame = cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY)
start_frame = cv2.GaussianBlur(start_frame, (21, 21), 0)

# Setting alarm
alarm = False
alarm_mode = False
alarm_counter = 0
recording = False
video_writer = None

# Email configuration (replace with your credentials)
password = "zjqlaajibfwrlinz"
sender_email = "bhattacharjeemainak20@gmail.com"
receiver_email = "mainakbhattacharya.2108@gmail.com"

# Function to send email notification with attachment
def send_email(subject, body, attachment_path=None):
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    if attachment_path:
        attachment = open(attachment_path, "rb")
        part = MIMEBase("application", "octet-stream")
        part.set_payload((attachment).read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {os.path.basename(attachment_path)}",
        )
        message.attach(part)
        attachment.close()

    try:
        server = smtplib.SMTP("smtp.gmail.com:587")
        server.ehlo()
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())
        server.quit()
        print(f"Email sent: {subject}")
    except Exception as e:
        print(f"Error sending email: {e}")

# Object detection model (replace with your model path)
model = YOLO("../../yolov8s.pt")
bbox_annotator = sv.BoxAnnotator()

# Function to send email notification with motion detection
def beep_alarm(object_name, threshold_sum, video_path):
    global alarm
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    email_body = (
        f"Motion detected with object identified: {object_name} at {current_time}\n"
        f"Threshold sum: {threshold_sum}"
    )
    send_email("Motion Alert - Object Detected", email_body, video_path)
    alarm = False

if __name__ == '__main__':
    requests.get(URL + "/control?var=framesize&val={}".format(8))

while True:
    ret, frame = cap.read()

    if ret:
        frame = imutils.resize(frame, width=500)

        if alarm_mode:
            frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_bw = cv2.GaussianBlur(frame_bw, (21, 21), 0)

            difference = cv2.absdiff(start_frame, frame_bw)
            threshold = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]
            threshold = cv2.dilate(threshold, None, iterations=2)
            contours, _ = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            threshold_sum = threshold.sum()

            motion_detected = False

            for contour in contours:
                if cv2.contourArea(contour) < 500:  # Minimum area to be considered motion
                    continue
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                motion_detected = True

            if motion_detected:
                alarm_counter += 1
                if not recording:
                    # Start recording
                    recording = True
                    video_filename = f"motion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
                    video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'XVID'), 20, (frame.shape[1], frame.shape[0]))
            else:
                if alarm_counter > 0:
                    alarm_counter -= 1

            cv2.imshow("cam", frame)

            if recording:
                video_writer.write(frame)

            if alarm_counter == 0 and recording:
                # Stop recording and save the video
                recording = False
                video_writer.release()
                video_writer = None

        else:
            cv2.imshow("cam", frame)

        if alarm_counter > 20:
            if not alarm:
                alarm = True
                result = model(frame)[0]
                detections = sv.Detections.from_ultralytics(result)
                detections = detections[detections.confidence > 0.5]

                if detections:
                    labels = [
                        result.names[class_id] for class_id in detections.class_id
                    ]
                    for object_name in labels:
                        threading.Thread(target=beep_alarm, args=(object_name, threshold_sum, video_filename)).start()

        key_pressed = cv2.waitKey(30)
        if key_pressed == ord("t"):
            alarm_mode = not alarm_mode
            alarm_counter = 0
        if key_pressed == ord("q"):
            alarm_mode = False
            break
        cv2.imshow("gray_frame", start_frame)

cap.release()
cv2.destroyAllWindows()

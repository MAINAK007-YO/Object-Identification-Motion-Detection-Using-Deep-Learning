import threading
import cv2
import imutils
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Setting up the camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Here 0 is the index no. of the camera

# size of the output camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# initializing the 1st frame with which we want to differentiate the current frame
_, start_frame = cap.read()
start_frame = imutils.resize(start_frame, width=500)
start_frame = cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY)  # Converting the color from BGR to gray scale
start_frame = cv2.GaussianBlur(start_frame, (21, 21), 0)

# setting alarm
alarm = False  # by default the alarm is set to false
alarm_mode = False  # to toggle the alarm
alarm_counter = 0  # how much duration is the movement to ring the alarm


# any msg or hardware device to ring the alarm or send a notification
def beep_alarm():
    global alarm

    password="zjqlaajibfwrlinz"
    me="bhattacharjeemainak20@gmail.com"
    you= "mainakbhattacharya.2108@gmail.com"
    email_body="""<html><body><p>motion detected</p></body></html>"""
    message= MIMEMultipart('alternative', None,[MIMEText (email_body, 'html')])
    message['Subject'] = 'Test email send'
    message['From'] = me
    message['To'] = you

    try:
        server=smtplib.SMTP('smtp.gmail.com:587')
        server.ehlo()
        server.starttls()
        server.login(me,password)
        server.sendmail(me,you,message.as_string())
        server.quit()
        print(f'Email sent: {email_body}')
    except Exception as e:
        print(f'Error in sending email: {e}')

    alarm = False


while True:
    _, frame = cap.read()
    frame = imutils.resize(frame, width=500)

    if alarm_mode:
        frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_bw = cv2.GaussianBlur(frame_bw, (5, 5), 0)

        difference = cv2.absdiff(frame_bw, start_frame)  # finding the differences with initial frame and current frame
        threshold = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]  # setting a threshold value
        start_frame = frame_bw  # frame for next iteration
        if threshold.sum() > 400000:
            # if we notify any movement more than threshold then we increment the alarm counter by 1
            alarm_counter += 1
        else:
            # if we don't notify any movement more than threshold then we decrement the alarm counter by 1
            if alarm_counter > 0:
                alarm_counter -= 1

        cv2.imshow("cam", threshold)  # Gray scale screen
    else:
        cv2.imshow("cam", frame)   # Gray scale screen
    if alarm_counter > 20:
        if not alarm:
            alarm = True
            threading.Thread(target=beep_alarm).start()

    key_pressed = cv2.waitKey(30)
    if key_pressed == ord("t"):
        alarm_mode = not alarm_mode   # press t for gray scale screen
        alarm_counter = 0
    if key_pressed == ord("q"):
        alarm_mode = False       # press q for close
        break
    cv2.imshow("gray_frame", start_frame)  # original output frame

cap.release()
cv2.destroyAllWindows()

import cv2
import supervision as sv
from ultralytics import YOLO

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

model = YOLO("yolov8s.pt")
bbox_annotator = sv.BoxAnnotator()

while video.isOpened():
    ret, frame = video.read()
    if ret == True:
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        confidence = detections.confidence
        detections = detections[confidence > 0.5]
        labels = []
        for class_id, confidence in zip(detections.class_id, detections.confidence):
            object_name = model.names[class_id]
            confidence_text = f"{object_name}: {confidence:.2f}%"
            labels.append(confidence_text)
        frame = bbox_annotator.annotate(scene=frame, detections=detections, labels=labels)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

video.release

cv2.destroyAllWindows
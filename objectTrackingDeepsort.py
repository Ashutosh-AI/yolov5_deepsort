import cv2
import numpy as np
import os
import time
import torch

from deep_sort_realtime.deepsort_tracker import DeepSort

class YoloDetector():

    def __init__(self, model_name):

        self.model = self.load_model(model_name)
        self.classes = self.model.names
        #print(self.classes)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using Device: ", self.device)


    def load_model(self, model_name):

        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)

        return model


    def score_frame(self, frame):

        self.model.to(self.device)
        downscale_factor = 2
        #print("frame shape", frame.shape)
        width = int(frame.shape[1] / downscale_factor)
        height = int(frame.shape[0] / downscale_factor)
        frame = cv2.resize(frame, (width, height))

        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        #print("labels cord", labels, cord)

        return labels, cord


    def class_to_label(self, x):

        return self.classes[int(x)]


    def plot_boxes(self, results, frame, height, width, score_thresh=0.3):

        labels, cord = results
        detections = []

        x_shape, y_shape = width, height
        n = len(labels)

        for i in range(n):
            row = cord[i]

            if row[4] >= score_thresh:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)

                if self.class_to_label(labels[i]) == "person":

                    x_center = x1 + ((x2-x1) / 2)
                    y_center = y1 + ((y2-y1) / 2)


                    tlwh = np.asarray([x1, y1, int(x2-x1), int(y2-y1)], dtype=np.float32)
                    confidence = float(row[4].item())
                    feature = "person"

                    detections.append(([x1,y1,int(x2-x1),int(y2-y1)], row[4].item(), "person"))

        return frame, detections





cap = cv2.VideoCapture(0)


detector = YoloDetector(model_name=None)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

object_tracker = DeepSort(max_age=30,      # if an object is not detected for 30 consecutive frames, its track will be terminated
                          n_init=2,
                          nms_max_overlap=1.0,
                          max_cosine_distance=0.3,
                          nn_budget=None,
                          override_track_class=None,
                          embedder="mobilenet",
                          half=True,
                          bgr=True,
                          embedder_gpu=True,
                          embedder_model_name=None,
                          embedder_wts=None,
                          polygon=False,
                          today=None)

while cap.isOpened():

    ret, img = cap.read()

    start_time = time.time()

    results = detector.score_frame(img)
    img, detections = detector.plot_boxes(results, img, height=img.shape[0], width=img.shape[1], score_thresh=0.5)

    #print(detections)

    tracks = object_tracker.update_tracks(detections, frame=img)   # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        #print(track_id)

        ltrb = track.to_ltrb()
        #print(ltrb)

        bbox = ltrb

        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), 2)
        cv2.putText(img, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    end_time = time.time()
    total_time = end_time - start_time
    fps = 1 / total_time

    cv2.putText(img, f"FPS: {int(fps)}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
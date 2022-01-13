import tensorflow as tf
import cv2 as cv
import numpy
import pandas as pd

detector = tf.saved_model.load("/home/dawid/repos/OpenCV-Features/Pretrained")
label_path = "/home/dawid/repos/OpenCV-Features/Pretrained/labels.csv"

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    
    ret, rgb = cap.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)
    rgb_tensor = tf.expand_dims(rgb_tensor, 0)
    rgb_tensor

    label = pd.read_csv(label_path, sep=';', index_col='ID')
    labels = label['OBJECT (2017 REL.)']
    
    boxes, scores, classes, num_detection = detector(rgb_tensor)
    num_detection

    pred_labels = classes.numpy().astype('int')[0]
    pred_labels = [labels[i] for i in pred_labels]

    pred_boxes = boxes.numpy()[0].astype('int')
    pred_scores = scores.numpy()[0]

    for score, (ymin, xmin, ymax, xmax), label in zip(pred_scores, pred_boxes, pred_labels):
        if score < 0.5:
            continue

        img_boxes = cv.rectangle(rgb, (xmin, ymin), (xmax,ymax), (0,255,0), 2)
        font = cv.FONT_HERSHEY_SIMPLEX

        score_txt = f"{100 * round(score)}%"
        cv.rectangle(img_boxes, (xmin, ymin), (xmax, ymin-24), (150, 255, 0), -1)
        cv.putText(img_boxes, "%s"%label+" %s"%score_txt, (xmin, ymin-5), font, 0.7, (0, 0, 0), 1, cv.LINE_AA)

    cv.imshow('frame', rgb)
    if cv.waitKey(1) == ord('q'):
        cap.release()
        cv.destroyAllWindows()
        break
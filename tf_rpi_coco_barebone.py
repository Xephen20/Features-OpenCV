from flask import Flask, Response, request, jsonify, render_template, stream_with_context
from flask_cors import CORS
import threading, argparse, time, cv2, math, json, glob, pickle, subprocess, os
from neopixel import *
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 90
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Define input and output tensors (i.e. data) for the object detection classifier

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
lock = threading.Lock()
app = Flask(__name__)
CORS(app) #allow foreign origin

camera = PiCamera()
camera.resolution = (1360, 768)

IM_WIDTH = 1360
IM_HEIGHT = 768

def main(runlevel):
    global grayFrame

    while True:
        start_time = time.time()
        while camera.analog_gain <= 1:
            time.sleep(0.1)
        
        rawCapture = PiRGBArray(camera, size=(1360, 768))        
        camera.capture(rawCapture, format="bgr", use_video_port=True)
        frame1 = rawCapture.array 
        frame = np.copy(frame1)
        frame.setflags(write=1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
        
        coordinates = vis_util.return_coordinates(
                        frame,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8,
                        min_score_thresh=0.80)
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.40)
        print(coordinates)
        with lock:
            grayFrame = frame.copy()
        #print("--- %s seconds ---" % (time.time() - start_time))

@stream_with_context
def gray():
    global grayFrame, lock
    while True:
        with lock:
            if grayFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", grayFrame)
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')

@app.route("/feed_gray")
def feed_gray():
    return Response(gray(), mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
        help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
        help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-r", "--runlevel", type=int, default=1,
        help="runlevel")
    args = vars(ap.parse_args())
    t = threading.Thread(target=main, args=(args["runlevel"],))
    t.daemon = True
    t.start()
    app.run(host=args["ip"], port=args["port"], debug=True,
        threaded=True, use_reloader=False)

camera.close()

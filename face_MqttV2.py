from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
import facenet
import requests
import detect_face
import os
import time
import pickle
from PIL import Image
import tensorflow.compat.v1 as tf
import paho.mqtt.client as mqtt  # Import MQTT library

# MQTT settings
mqtt_broker = "broker.emqx.io"
mqtt_port = 1883
mqtt_topic = "Face-detect/recognite"

# Create an MQTT client instance
client = mqtt.Client()

# MQTT connection check
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker successfully")
    else:
        print(f"Failed to connect, return code {rc}")

client.on_connect = on_connect

# Connect to the broker
client.connect(mqtt_broker, mqtt_port, 60)

# Start the MQTT loop
client.loop_start()

# Variables for controlling detection display and message sending interval
last_detection_print_time = time.time()
detection_display_interval = 10  # Display every 5 seconds
last_sent_time = time.time()  # Initial last sent time for MQTT messages
send_interval = 60  # Interval for sending the same name in seconds
sent_names = {}  # Dictionary to track names and last sent times

# Function to print detection message only if the interval has passed
def print_detection_message(name, probability):
    global last_detection_print_time
    current_time = time.time()
    if current_time - last_detection_print_time >= detection_display_interval:
        print(f"Predictions: [ name: {name} , accuracy: {probability:.3f} ]")
        last_detection_print_time = current_time

# Function to send MQTT message for detected face with interval control
def send_mqtt_message(name):
    current_time = time.time()
    if name not in sent_names or current_time - sent_names[name] >= send_interval:
        result = client.publish(mqtt_topic, f"Face is {name}")
        status = result[0]
        if status == 0:
            print(f"Sent {name} to topic {mqtt_topic}")
            sent_names[name] = current_time
        else:
            print(f"Failed to send message to topic {mqtt_topic}")

video = 0  # Camera source (0 for default camera)

# URL ของ Camera Webserver ที่ได้จาก ESP32-CAM
video_url = "http://192.168.183.191/capture"
# video_url = "http://192.168.183.191"

# TensorFlow and face detection setup
modeldir = './model/20180402-114759.pb'
classifier_filename = './class/classifier.pkl'
npy = './npy'
train_img = "./train_img"

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
        minsize = 30  # minimum size of face
        threshold = [0.7, 0.8, 0.8]  # three steps' threshold
        factor = 0.709  # scale factor
        margin = 44
        batch_size = 100
        image_size = 182
        input_image_size = 160
        HumanNames = os.listdir(train_img)
        HumanNames.sort()
        print('Loading Model')
        facenet.load_model(modeldir)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile, encoding='latin1')

        #video_capture = cv2.VideoCapture(video)
        print('Start Recognition')
        while True:
            # ดึงภาพจาก Camera Webserver ผ่าน URL
            img_resp = requests.get(video_url)
            img_array = np.array(bytearray(img_resp.content), dtype=np.uint8)
            frame = cv2.imdecode(img_array, -1)

            '''ret, frame = video_capture.read()'''
            timer = time.time()
            if frame.ndim == 2:
                frame = facenet.to_rgb(frame)
            bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
            faceNum = bounding_boxes.shape[0]
            if faceNum > 0:
                for i in range(faceNum):
                    emb_array = np.zeros((1, embedding_size))
                    xmin = int(bounding_boxes[i][0])
                    ymin = int(bounding_boxes[i][1])
                    xmax = int(bounding_boxes[i][2])
                    ymax = int(bounding_boxes[i][3])
                    if xmin <= 0 or ymin <= 0 or xmax >= len(frame[0]) or ymax >= len(frame):
                        print('Face is very close!')
                        continue
                    cropped = frame[ymin:ymax, xmin:xmax, :]
                    cropped = facenet.flip(cropped, False)
                    scaled = np.array(Image.fromarray(cropped).resize((image_size, image_size)))
                    scaled = cv2.resize(scaled, (input_image_size, input_image_size), interpolation=cv2.INTER_CUBIC)
                    scaled = facenet.prewhiten(scaled)
                    scaled_reshape = scaled.reshape(-1, input_image_size, input_image_size, 3)
                    feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                    emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                    predictions = model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                    
                    # Check probability threshold
                    if best_class_probabilities > 0.87:
                        result_names = HumanNames[best_class_indices[0]]
                        print_detection_message(result_names, best_class_probabilities[0])
                        send_mqtt_message(result_names)
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        cv2.rectangle(frame, (xmin, ymin - 20), (xmax, ymin - 2), (0, 255, 255), -1)
                        cv2.putText(frame, result_names, (xmin, ymin - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (0, 0, 0), thickness=1, lineType=1)
                    else:
                        print_detection_message("Unknown", best_class_probabilities[0])

            endtimer = time.time()
            fps = 1 / (endtimer - timer)
            cv2.rectangle(frame, (15, 30), (135, 60), (0, 255, 255), -1)
            cv2.putText(frame, "fps: {:.2f}".format(fps), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.imshow('Face Recognition', frame)
            key = cv2.waitKey(1)
            if key == 113:  # "q" to quit
                break

        video_capture.release()
        cv2.destroyAllWindows()

# Stop the MQTT loop and disconnect after program ends
client.loop_stop()
client.disconnect()

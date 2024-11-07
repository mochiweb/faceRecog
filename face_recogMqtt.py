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

video = 0

# URL ของ Camera Webserver ที่ได้จาก ESP32-CAM
# video_url = "http://192.168.183.191/capture"
# video_url = "http://192.168.183.191"

modeldir = './model/20180402-114759.pb'
classifier_filename = './class/classifier.pkl'
npy = './npy'
train_img = "./train_img"

# Initial value of the last time a message was sent
last_sent_time = time.time()

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
        minsize = 30  # minimum size of face
        threshold = [0.7, 0.8, 0.8]  # three steps' threshold
        factor = 0.709  # scale factor
        margin = 44
        batch_size = 100  # 1000
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

        video_capture = cv2.VideoCapture(video)
        print('Start Recognition')
        while True:
            # ดึงภาพจาก Camera Webserver ผ่าน URL
            '''img_resp = requests.get(video_url)
            img_array = np.array(bytearray(img_resp.content), dtype=np.uint8)
            frame = cv2.imdecode(img_array, -1)'''

            ret, frame = video_capture.read()
            timer = time.time()
            if frame.ndim == 2:
                frame = facenet.to_rgb(frame)
            bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
            faceNum = bounding_boxes.shape[0]
            if faceNum > 0:
                det = bounding_boxes[:, 0:4]
                img_size = np.asarray(frame.shape)[0:2]
                cropped = []
                scaled = []
                scaled_reshape = []
                for i in range(faceNum):
                    emb_array = np.zeros((1, embedding_size))
                    xmin = int(det[i][0])
                    ymin = int(det[i][1])
                    xmax = int(det[i][2])
                    ymax = int(det[i][3])
                    try:
                        if xmin <= 0 or ymin <= 0 or xmax >= len(frame[0]) or ymax >= len(frame):
                            print('Face is very close!')
                            continue
                        cropped.append(frame[ymin:ymax, xmin:xmax, :])
                        cropped[i] = facenet.flip(cropped[i], False)
                        scaled.append(np.array(Image.fromarray(cropped[i]).resize((image_size, image_size))))
                        scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                               interpolation=cv2.INTER_CUBIC)
                        scaled[i] = facenet.prewhiten(scaled[i])
                        scaled_reshape.append(scaled[i].reshape(-1, input_image_size, input_image_size, 3))
                        feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                        predictions = model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        if best_class_probabilities > 0.87:
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # boxing face
                            for H_i in HumanNames:
                                if HumanNames[best_class_indices[0]] == H_i:
                                    result_names = HumanNames[best_class_indices[0]]
                                    print("Predictions : [ name: {} , accuracy: {:.3f} ]".format(HumanNames[best_class_indices[0]], best_class_probabilities[0]))
                                    cv2.rectangle(frame, (xmin, ymin - 20), (xmax, ymin - 2), (0, 255, 255), -1)
                                    cv2.putText(frame, result_names, (xmin, ymin - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (0, 0, 0), thickness=1, lineType=1)

                                    # Check if 2 seconds have passed since the last message was sent
                                    if time.time() - last_sent_time > 2:
                                        # Publish the recognized name to the MQTT broker
                                        result = client.publish(mqtt_topic, f"Faceis {result_names}")
                                        status = result[0]
                                        if status == 0:
                                            print(f"Sent {result_names} to topic {mqtt_topic}")
                                            last_sent_time = time.time()  # Update the last sent time
                                        else:
                                            print(f"Failed to send message to topic {mqtt_topic}")
                        else:
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                            cv2.rectangle(frame, (xmin, ymin - 20), (xmax, ymin - 2), (0, 255, 255), -1)
                            cv2.putText(frame, "?", (xmin, ymin - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (0, 0, 0), thickness=1, lineType=1)
                    except:
                        print("error")

            endtimer = time.time()
            fps = 1 / (endtimer - timer)
            cv2.rectangle(frame, (15, 30), (135, 60), (0, 255, 255), -1)
            cv2.putText(frame, "fps: {:.2f}".format(fps), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.imshow('Face Recognition', frame)
            key = cv2.waitKey(1)
            if key == 113:  # "q"
                break

        video_capture.release()
        cv2.destroyAllWindows()

# Stop the MQTT loop and disconnect after program ends
client.loop_stop()
client.disconnect()

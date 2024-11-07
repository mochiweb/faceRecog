"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy import misc
import os
import tensorflow.compat.v1 as tf

import numpy as np
import facenet
import detect_face
import imageio
from PIL import Image

class preprocesses:
    def __init__(self, input_datadir, output_datadir):
        self.input_datadir = input_datadir
        self.output_datadir = output_datadir

    def collect_data(self):
        output_dir = os.path.expanduser(self.output_datadir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dataset = facenet.get_dataset(self.input_datadir)
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(sess, './npy')

        minsize = 20  # minimum size of face
        threshold = [0.5, 0.6, 0.6]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        image_size = 182

        # Add a random key to the filename to allow alignment using multiple processes
        random_key = np.random.randint(0, high=99999)
        bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)

        with open(bounding_boxes_filename, "w") as text_file:
            nrof_images_total = 0
            nrof_successfully_aligned = 0
            for cls in dataset:
                output_class_dir = os.path.join(output_dir, cls.name)
                if not os.path.exists(output_class_dir):
                    os.makedirs(output_class_dir)
                for image_path in cls.image_paths:
                    nrof_images_total += 1
                    filename = os.path.splitext(os.path.split(image_path)[1])[0]
                    output_filename = os.path.join(output_class_dir, filename + '.png')
                    print("Image: %s" % image_path)
                    if not os.path.exists(output_filename):
                        try:
                            img = imageio.imread(image_path)
                        except (IOError, ValueError, IndexError) as e:
                            errorMessage = '{}: {}'.format(image_path, e)
                            print(errorMessage)
                        else:
                            if img.ndim < 2:
                                print('Unable to align "%s"' % image_path)
                                text_file.write('%s\n' % (output_filename))
                                continue
                            if img.ndim == 2:
                                img = facenet.to_rgb(img)
                                print('to_rgb data dimension: ', img.ndim)
                            img = img[:, :, 0:3]

                            bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold,
                                                                        factor)
                            nrof_faces = bounding_boxes.shape[0]
                            print('No of Detected Face: %d' % nrof_faces)
                            if nrof_faces > 0:
                                det = bounding_boxes[:, 0:4]
                                img_size = np.asarray(img.shape)[0:2]
                                if nrof_faces > 1:
                                    bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                                    img_center = img_size / 2
                                    offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                                         (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                                    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                                    index = np.argmax(
                                        bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                                    det = det[index, :]
                                det = np.squeeze(det)
                                bb_temp = np.zeros(4, dtype=np.int32)

                                bb_temp[0] = det[0]
                                bb_temp[1] = det[1]
                                bb_temp[2] = det[2]
                                bb_temp[3] = det[3]
                                #np.array(Image.fromarray(cropped[i]).resize((image_size, image_size)))
                                cropped_temp = img[bb_temp[1]:bb_temp[3], bb_temp[0]:bb_temp[2], :]
                                #scaled_temp = misc.imresize(cropped_temp, (image_size, image_size), interp='bilinear')
                                scaled_temp =np.array(Image.fromarray(cropped_temp).resize((image_size, image_size)))
                                nrof_successfully_aligned += 1
                                imageio.imwrite(output_filename, scaled_temp)
                                text_file.write('%s %d %d %d %d\n' % (
                                output_filename, bb_temp[0], bb_temp[1], bb_temp[2], bb_temp[3]))
                            else:
                                print('Unable to align "%s"' % image_path)
                                text_file.write('%s\n' % (output_filename))

        return (nrof_images_total,nrof_successfully_aligned)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import tensorflow.compat.v1 as tf
import imageio
from PIL import Image
import cv2
from skimage.util import random_noise
import facenet
import detect_face

class preprocesses:
    def __init__(self, input_datadir, output_datadir):
        self.input_datadir = input_datadir
        self.output_datadir = output_datadir

    def flip_image(self, img):
        # Flip horizontally
        return cv2.flip(img, 1)

    def blur_image(self, img):
        # Apply Gaussian Blur
        return cv2.GaussianBlur(img, (5, 5), 0)

    def add_noise(self, img):
        # Convert image to float before adding noise
        noise_img = random_noise(img, mode='gaussian', var=0.01)
        # Convert back to uint8
        noise_img = np.array(255 * noise_img, dtype='uint8')
        return noise_img

    def to_grayscale(self, img):
        # Convert the image to grayscale
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def collect_data(self):
        output_dir = os.path.expanduser(self.output_datadir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dataset = facenet.get_dataset(self.input_datadir)
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(sess, './npy')

        minsize = 20  # minimum size of face
        threshold = [0.5, 0.6, 0.6]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        image_size = 182

        # Add a random key to the filename to allow alignment using multiple processes
        random_key = np.random.randint(0, high=99999)
        bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)

        with open(bounding_boxes_filename, "w") as text_file:
            nrof_images_total = 0
            nrof_successfully_aligned = 0
            for cls in dataset:
                output_class_dir = os.path.join(output_dir, cls.name)
                if not os.path.exists(output_class_dir):
                    os.makedirs(output_class_dir)
                for image_path in cls.image_paths:
                    nrof_images_total += 1
                    filename = os.path.splitext(os.path.split(image_path)[1])[0]
                    output_filename = os.path.join(output_class_dir, filename + '.png')
                    print("Image: %s" % image_path)
                    if not os.path.exists(output_filename):
                        try:
                            img = imageio.imread(image_path)
                        except (IOError, ValueError, IndexError) as e:
                            errorMessage = '{}: {}'.format(image_path, e)
                            print(errorMessage)
                        else:
                            if img.ndim < 2:
                                print('Unable to align "%s"' % image_path)
                                text_file.write('%s\n' % (output_filename))
                                continue
                            if img.ndim == 2:
                                img = facenet.to_rgb(img)
                                print('to_rgb data dimension: ', img.ndim)
                            img = img[:, :, 0:3]

                            bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                            nrof_faces = bounding_boxes.shape[0]
                            print('No of Detected Face: %d' % nrof_faces)
                            if nrof_faces > 0:
                                det = bounding_boxes[:, 0:4]
                                img_size = np.asarray(img.shape)[0:2]
                                if nrof_faces > 1:
                                    bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                                    img_center = img_size / 2
                                    offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                                         (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                                    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                                    index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                                    det = det[index, :]
                                det = np.squeeze(det)
                                bb_temp = np.zeros(4, dtype=np.int32)

                                bb_temp[0] = det[0]
                                bb_temp[1] = det[1]
                                bb_temp[2] = det[2]
                                bb_temp[3] = det[3]

                                # Crop the aligned image
                                cropped_temp = img[bb_temp[1]:bb_temp[3], bb_temp[0]:bb_temp[2], :]
                                # Resize the cropped image
                                scaled_temp = np.array(Image.fromarray(cropped_temp).resize((image_size, image_size)))

                                # Save aligned image
                                nrof_successfully_aligned += 1
                                imageio.imwrite(output_filename, scaled_temp)
                                text_file.write('%s %d %d %d %d\n' % (
                                    output_filename, bb_temp[0], bb_temp[1], bb_temp[2], bb_temp[3]))

                                # Perform additional augmentations
                                # Flip the image
                                flipped_img = self.flip_image(scaled_temp)
                                flipped_img_path = os.path.join(output_class_dir, f"flipped_{filename}.png")
                                imageio.imwrite(flipped_img_path, flipped_img)

                                # Blur the image
                                blurred_img = self.blur_image(scaled_temp)
                                blurred_img_path = os.path.join(output_class_dir, f"blurred_{filename}.png")
                                imageio.imwrite(blurred_img_path, blurred_img)

                                # Add noise to the image
                                noisy_img = self.add_noise(scaled_temp)
                                noisy_img_path = os.path.join(output_class_dir, f"noisy_{filename}.png")
                                imageio.imwrite(noisy_img_path, noisy_img)

                                # Convert image to grayscale and save
                                grayscale_img = self.to_grayscale(scaled_temp)
                                grayscale_img_path = os.path.join(output_class_dir, f"grayscale_{filename}.png")
                                imageio.imwrite(grayscale_img_path, grayscale_img)

                            else:
                                print('Unable to align "%s"' % image_path)
                                text_file.write('%s\n' % (output_filename))

        return nrof_images_total, nrof_successfully_aligned

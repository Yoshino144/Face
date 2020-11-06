# -*- coding: utf-8 -*-
# @Author  : PengCheng
# @File    : main.py
# @Explain : 最终

import time
import json
from keras import backend as K
import cv2
import tensorflow as tf
from keras.models import load_model
import dlib
import numpy as np

PATH_TO_TENSORFLOW_MODEL = './models/face_mask_detection.pb'
MODEL_PATH = './data/face.model'
IMAGE_SIZE = 128


def load_tf_model(tf_model_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(tf_model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            with detection_graph.as_default():
                sess = tf.Session(graph=detection_graph)
                return sess, detection_graph
#testtbb

def tf_inference(sess, detection_graph, img_arr):
    image_tensor = detection_graph.get_tensor_by_name('data_1:0')
    detection_bboxes = detection_graph.get_tensor_by_name('loc_branch_concat_1/concat:0')
    detection_scores = detection_graph.get_tensor_by_name('cls_branch_concat_1/concat:0')
    bboxes, scores = sess.run([detection_bboxes, detection_scores], feed_dict={image_tensor: img_arr})
    return bboxes, scores


def generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios, offset=0.5):
    anchor_bboxes = []
    for idx, feature_size in enumerate(feature_map_sizes):
        cx = (np.linspace(0, feature_size[0] - 1, feature_size[0]) + 0.5) / feature_size[0]
        cy = (np.linspace(0, feature_size[1] - 1, feature_size[1]) + 0.5) / feature_size[1]
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid_expend = np.expand_dims(cx_grid, axis=-1)
        cy_grid_expend = np.expand_dims(cy_grid, axis=-1)
        center = np.concatenate((cx_grid_expend, cy_grid_expend), axis=-1)
        num_anchors = len(anchor_sizes[idx]) + len(anchor_ratios[idx]) - 1
        center_tiled = np.tile(center, (1, 1, 2 * num_anchors))
        anchor_width_heights = []
        for scale in anchor_sizes[idx]:
            ratio = anchor_ratios[idx][0]
            width = scale * np.sqrt(ratio)
            height = scale / np.sqrt(ratio)
            anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])
        for ratio in anchor_ratios[idx][1:]:
            s1 = anchor_sizes[idx][0]
            width = s1 * np.sqrt(ratio)
            height = s1 / np.sqrt(ratio)
            anchor_width_heights.extend([-width / 2.0, -height / 2.0, width / 2.0, height / 2.0])
        bbox_coords = center_tiled + np.array(anchor_width_heights)
        bbox_coords_reshape = bbox_coords.reshape((-1, 4))
        anchor_bboxes.append(bbox_coords_reshape)
    anchor_bboxes = np.concatenate(anchor_bboxes, axis=0)
    return anchor_bboxes


def decode_bbox(anchors, raw_outputs, variances=None):
    if variances is None:
        variances = [0.1, 0.1, 0.2, 0.2]
    anchor_centers_x = (anchors[:, :, 0:1] + anchors[:, :, 2:3]) / 2
    anchor_centers_y = (anchors[:, :, 1:2] + anchors[:, :, 3:]) / 2
    anchors_w = anchors[:, :, 2:3] - anchors[:, :, 0:1]
    anchors_h = anchors[:, :, 3:] - anchors[:, :, 1:2]
    raw_outputs_rescale = raw_outputs * np.array(variances)
    predict_center_x = raw_outputs_rescale[:, :, 0:1] * anchors_w + anchor_centers_x
    predict_center_y = raw_outputs_rescale[:, :, 1:2] * anchors_h + anchor_centers_y
    predict_w = np.exp(raw_outputs_rescale[:, :, 2:3]) * anchors_w
    predict_h = np.exp(raw_outputs_rescale[:, :, 3:]) * anchors_h
    predict_xmin = predict_center_x - predict_w / 2
    predict_ymin = predict_center_y - predict_h / 2
    predict_xmax = predict_center_x + predict_w / 2
    predict_ymax = predict_center_y + predict_h / 2
    predict_bbox = np.concatenate([predict_xmin, predict_ymin, predict_xmax, predict_ymax], axis=-1)
    return predict_bbox


def single_class_non_max_suppression(bboxes, confidences, conf_thresh=0.2, iou_thresh=0.5, keep_top_k=-1):
    if len(bboxes) == 0: return []
    conf_keep_idx = np.where(confidences > conf_thresh)[0]
    bboxes = bboxes[conf_keep_idx]
    confidences = confidences[conf_keep_idx]
    pick = []
    xmin = bboxes[:, 0]
    ymin = bboxes[:, 1]
    xmax = bboxes[:, 2]
    ymax = bboxes[:, 3]
    area = (xmax - xmin + 1e-3) * (ymax - ymin + 1e-3)
    idxs = np.argsort(confidences)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        if keep_top_k != -1:
            if len(pick) >= keep_top_k:
                break
        overlap_xmin = np.maximum(xmin[i], xmin[idxs[:last]])
        overlap_ymin = np.maximum(ymin[i], ymin[idxs[:last]])
        overlap_xmax = np.minimum(xmax[i], xmax[idxs[:last]])
        overlap_ymax = np.minimum(ymax[i], ymax[idxs[:last]])
        overlap_w = np.maximum(0, overlap_xmax - overlap_xmin)
        overlap_h = np.maximum(0, overlap_ymax - overlap_ymin)
        overlap_area = overlap_w * overlap_h
        overlap_ratio = overlap_area / (area[idxs[:last]] + area[i] - overlap_area)
        need_to_be_deleted_idx = np.concatenate(([last], np.where(overlap_ratio > iou_thresh)[0]))
        idxs = np.delete(idxs, need_to_be_deleted_idx)
    return conf_keep_idx[pick]


def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


class Exp:
    def __init__(self):
        self.detection_model_path = './data/haarcascade_frontalface_default.xml'
        self.emotion_model_path = './data/fer2013_mini_XCEPTION.110-0.65.hdf5'
        self.emotion_labels = {0: 'surprise', 1: 'surprise', 2: 'surprise', 3: 'happy', 4: 'surprise', 5: 'surprise',
                               6: 'neutral'}
        self.face_detection = cv2.CascadeClassifier(self.detection_model_path)
        self.emotion_classifier = load_model(self.emotion_model_path, compile=False)
        self.emotion_target_size = self.emotion_classifier.input_shape[1:3]
        self.emotion_window = []
        self.frame_window = 10

    def exp(self, gray_facee):
        gray_face = cv2.resize(gray_facee, (self.emotion_target_size))
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = self.emotion_classifier.predict(gray_face)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = self.emotion_labels[emotion_label_arg]
        return emotion_text


def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    h, w, _ = image.shape
    longest_edge = max(h, w)
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    return cv2.resize(constant, (height, width))


class Model:
    def __init__(self):
        self.model = None

    def load_model(self, file_path=MODEL_PATH):
        self.model = load_model(file_path)

    def face_predict(self, image):
        if K.image_dim_ordering() == 'th' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_image(image)
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))
        elif K.image_dim_ordering() == 'tf' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        image = image.astype('float32')
        image /= 255
        result_probability = self.model.predict_proba(image)
        # print('[  INFO  ] = [ Result ] = ' + str(result_probability) )
        print('[  INFO  ] = [ MaxResult ] = ' + str(max(result_probability[0])))
        result = self.model.predict_classes(image)
        return max(result_probability[0]), result[0]


sess, graph = load_tf_model('data/face_mask_detection.pb')
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)
anchors_exp = np.expand_dims(anchors, axis=0)
id2class = {0: 'Mask', 1: 'NoMask'}
with open('./data/contrast_table', 'r') as f:
    contrast_table = json.loads(f.read())
model = Model()
model.load_model(file_path='./data/face.model')
detector = dlib.get_frontal_face_detector()
exp = Exp()


def inference(image, conf_thresh=0.5, iou_thresh=0.4, target_shape=(160, 160)):
    output_info = []
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0
    image_exp = np.expand_dims(image_np, axis=0)
    y_bboxes_output, y_cls_output = tf_inference(sess, graph, image_exp)
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)
    keep_idxs = single_class_non_max_suppression(y_bboxes, bbox_max_scores, conf_thresh=conf_thresh,
                                                 iou_thresh=iou_thresh, )
    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)
        frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print("[  INFO  ] = [ Region ] = (" + str(xmin) + "," + str(ymin) + "),(" + str(xmax) + "," + str(ymax) + ")")
        face = image[ymin:ymax, xmin:xmax, ::-1]
        probability, name_number = model.face_predict(face)
        print("[  INFO  ] = [ Number ] = " + str(name_number))
        name = contrast_table[str(name_number)]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        if probability > 0.50:
            print("[  INFO  ] = [ Name ] = " + name)
            cv2.putText(image, name, (xmax + 0, ymax + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            print("[  INFO  ] = [ Name ] = " + 'unknow')
            cv2.putText(image, 'unknow', (xmax + 0, ymax + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        emotion_text = exp.exp(frame_gray[ymin:ymax, xmin:xmax])
        cv2.putText(image, emotion_text, (xmax + 0, ymax + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print("[  INFO  ] = [ Emotion ] = " + emotion_text)
        cv2.putText(image, id2class[class_id], (xmax + 0, ymax + 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print("[  INFO  ] = [ Mask ] = " + id2class[class_id])
        output_info.append([class_id, conf, xmin, ymin, xmax, ymax])
    return output_info


def run_on_video(video_path, conf_thresh):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Video open failed.")
    status = True
    while status:
        print("----------------------------------------------------------------------")
        start_stamp = time.time()
        status, img_raw = cap.read()
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        read_frame_stamp = time.time()
        if status:
            inference(img_raw, conf_thresh, iou_thresh=0.5, target_shape=(260, 260))
            cv2.imshow('image', img_raw[:, :, ::-1])
            cv2.waitKey(1)
            inference_stamp = time.time()
            write_frame_stamp = time.time()
            print("[  INFO  ] = [ MaskState ] = read_frame:%f, infer time:%f, write time:%f" % (
            read_frame_stamp - start_stamp,
            inference_stamp - read_frame_stamp,
            write_frame_stamp - inference_stamp))
        c = cv2.waitKey(1)
        if c == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_on_video(0, conf_thresh=0.5)

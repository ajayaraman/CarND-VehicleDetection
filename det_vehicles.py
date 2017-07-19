import numpy as np
import cv2
import matplotlib.image as mpimg
import os
import sys
import glob
#tensorflow imports
import tensorflow as tf

sys.path.append('../models/object_detection')

#tensorflow object detection api imports
from utils import label_map_util
from utils import visualization_utils as vis_util

from collections import deque

class VehicleDetector:
    def __init__(self, model_file_path, label_map_path):
        
        #Initialize conv net
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_file_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        #Initialize tensorflow session
        with detection_graph.as_default():
            sess = tf.Session(graph=detection_graph)        
        self._sess = sess

        #store tensors to run graph
        self._image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self._boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self._scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self._classes = detection_graph.get_tensor_by_name('detection_classes:0')
        self._num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        
        #load label map and create category index
        NUM_CLASSES = 1        
        label_map = label_map_util.load_labelmap(label_map_path)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self._category_index = label_map_util.create_category_index(categories)

        self.history = deque(maxlen = 5)

    #detect cars
    def detect(self, img):
        image_batch = np.expand_dims(img, axis=0)

        # Actual detection.
        (boxes, scores, classes, num_detections) = self._sess.run(
          [self._boxes, self._scores, self._classes, self._num_detections],
          feed_dict={self._image_tensor: image_batch})

        # Visualization of the results of a detection.
        """
        vis_util.visualize_boxes_and_labels_on_image_array(
          img,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          self._category_index,
          use_normalized_coordinates=True,
          line_thickness=4)
        """
        

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)

        boxes = self.remove_boxes_criteria( boxes, scores)

        boxes = self.smooth_detections(boxes, img)
            
        imgout = self.annotate_image(boxes, img)
        
        return imgout

    def remove_boxes_criteria(self, boxes, scores):
        #only keep boxes above a certain score
        boxes = boxes[ scores > 0.4 ]

        #remove boxes that are not tall enough
        boxes = boxes [ (boxes[:, 2] -  boxes[:, 0]) > 0.08    ]

        return boxes

    def annotate_image(self, boxes, img):
        for box in boxes:
            x, y, w, h = box
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)
        return img

    def smooth_detections(self, boxes, img):
        mask = np.zeros( img.shape[0:2], dtype = np.uint8 )
        xx, yy = np.meshgrid( np.arange(0, img.shape[1]), np.arange(0, img.shape[0]) )
        for i in range(boxes.shape[0]):
            box = tuple(boxes[i].tolist())
            ymin, xmin, ymax, xmax = box
            ymin *= img.shape[0]
            ymax *= img.shape[0]
            xmin *= img.shape[1]
            xmax *= img.shape[1] 
            mask[ (xx >= xmin) & (xx <= xmax) & (yy >= ymin) & (yy <= ymax) ] += 1
        
        self.history.append(mask)

        #use history to calculate mean mask over past few images
        smoothedmask = np.zeros(img.shape[0:2])
        for mask in self.history:
            smoothedmask += mask
        smoothedmask /= len(self.history)

        #threshold the mask
        thresh = 0.01
        smoothedmask [  smoothedmask > thresh ] = 255
        smoothedmask [  smoothedmask <= thresh ] = 0

        #count the box extents
        # find contours
        _, cnt, _ = cv2.findContours(smoothedmask.astype(np.uint8), cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)

        boxes_out  = []
        for c in cnt:
            x,y,w,h = cv2.boundingRect(c)
            boxes_out.append( (x,y,w,h) )

        return boxes_out
        
def run_vehicle_det(videofile, showPlayback = False):
    cap = cv2.VideoCapture(videofile)
    width = int(cap.get(3))
    height = int(cap.get(4))

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    outfilename = os.path.join("output", "out_" + os.path.basename(videofile))
    out = cv2.VideoWriter(outfilename, fourcc, 30.0, (width, height))

    detector = VehicleDetector('output_inference_graph.pb', 'car_label_map.pbtxt')

    count = 0
    while(cap.isOpened()):
        print('frame ' + str(count))
        count = count + 1
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ret == True:
            outframe = detector.detect(frame)
            outframe = cv2.cvtColor(outframe, cv2.COLOR_RGB2BGR)
            out.write(outframe)

            if showPlayback:
                cv2.imshow('frame', outframe)

                if cv2.waitKey(1) & 0xFF == ord('c'):
                    break
        else:
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def test_vehicle_detection(imagefilename):
    detector = VehicleDetector('output_inference_graph.pb', 'car_label_map.pbtxt')
    frame = mpimg.imread(imagefilename)
    outframe = detector.detect(frame)
    mpimg.imsave(os.path.join('output_images', os.path.basename(imagefilename)), outframe)

def run_test_images():
    for filename in glob.glob('test_images/*.jpg'):
        test_vehicle_detection(filename)

#run_test_images()

run_vehicle_det('project_video.mp4', showPlayback = True)


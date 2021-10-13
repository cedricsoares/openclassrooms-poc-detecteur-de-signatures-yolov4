import cv2
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

def load_the_network(yolo_cfg, yolo_weights):
	net = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
	layers = net.getLayerNames()
	output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	return net, output_layers

class SignatureDetector:
	
	def __init__(self):
		YOLO_CONFIG_PATH = dir_path + '/final_model.cfg'
		YOLO_WEIGHTS_PATH = dir_path + '/final_model.weights'
		self.net, self.output_layers = load_the_network(YOLO_CONFIG_PATH, YOLO_WEIGHTS_PATH)

	def get_predictions(self, PIL_image, CONF_THRESH):
	      
	  opencvImage = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)
	  height, width = opencvImage.shape[:2]

	  blob = cv2.dnn.blobFromImage(opencvImage, 0.00392, (608, 608), swapRB=False, crop=False)
	  self.net.setInput(blob)
	  layer_outputs = self.net.forward(self.output_layers)

	  class_ids, confidences, b_boxes = [], [], []
	  for output in layer_outputs:
	      for detection in output:
	          scores = detection[5:]
	          class_id = np.argmax(scores)
	          confidence = scores[class_id]

	          if confidence > CONF_THRESH:
	              center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

	              x = int(center_x - w / 2)
	              y = int(center_y - h / 2)

	              b_boxes.append([x, y, int(w), int(h)])
	              confidences.append(float(confidence))
	              class_ids.append(int(class_id))

	  return class_ids, confidences, b_boxes


	def NMS(self, confidences, b_boxes, CONF_THRESH, NMS_THRESH):
	  results=cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH)
	  if len(results) > 0:
	  	results=results.flatten().tolist()
	  	nms_boxes=[b_boxes[box] for box in results]
	  	return nms_boxes
	  else:
	    nms_boxes=0
	    return nms_boxes



from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
from model import SignatureDetector
import io
from PIL import Image

app = FastAPI(title='API for signature detection',
              description='Fine-tuned Yolo4 model',
              version='0.0.1')

# model is loaded just once on start
detector = SignatureDetector()

@app.get("/")
def index():
    return "Welcome to the Signature detection API! Check /docs for usage."

# transform string to image
def base64str_to_PILImage(base64str):
   base64_img_bytes = base64str.encode('utf-8')
   base64bytes = base64.b64decode(base64_img_bytes)
   bytesObj = io.BytesIO(base64bytes)
   img = Image.open(bytesObj) 
   return img

# define the Input class
class Input(BaseModel):
	base64str : str

@app.post("/predict")
def get_prediction(
	data: Input,
	nms_threshold: float = Query(0.25, description="Should be between 0 and 1"),
	conf_threshold: float = Query(0.25, description="Should be between 0 and 1")):
	'''
	API will take a base 64 image as input and return a json object
	'''
	# Load the image
	img = base64str_to_PILImage(data.base64str)
	class_ids, confidences, b_boxes = detector.get_predictions(img, conf_threshold)
	result = detector.NMS(confidences, b_boxes, conf_threshold, nms_threshold)
	return JSONResponse(status_code=200, content={"prediction": result})
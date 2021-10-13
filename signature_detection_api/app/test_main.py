import json
import base64
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def img_to_base64str(file_path:str):
    with open(file_path, "rb") as image_file:
        base64str = base64.b64encode(image_file.read()).decode("utf-8")
    return base64str

def test_index():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Welcome to the Signature detection API! Check /docs for usage."

def test_docs():
    response = client.get("/docs")
    assert response.status_code == 200
    assert "API for signature detection" in response.text

def test_get_prediction():
    base64str = img_to_base64str('./test.png')
    payload = json.dumps({"base64str": base64str})
    response = client.post("/predict", data=payload)
    assert response.status_code == 200
    assert {'prediction': [[1192, 2460, 1002, 336]]} == response.json()

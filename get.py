import base64
import io
import os
from PIL import Image
from flask import Flask, jsonify, request

from forest_mask import ForestMask

app = Flask(__name__)

@app.route('/api/get', methods=['GET'])
def get():
    image_base64 = request.data

    image_raw = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_raw))
    
    return_image, return_area = ForestMask.get_all(image)

    return_image = image_to_base64(return_image)

    response = {
    'image': return_image,
    'area': return_area
    }
    
    # we return json with the image and area in m2
    return jsonify(response)

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format='PNG')
    img_str = base64.b64encode(buffered.getvalue()).decode('ascii')
    return img_str

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
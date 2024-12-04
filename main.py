import concurrent.futures

from flask import Flask, request, jsonify, Response
import logging
from logging.handlers import TimedRotatingFileHandler
from werkzeug.middleware.proxy_fix import ProxyFix
import boto3
import io
import numpy as np

from image_worker import LabelWorker
local_test = False
app = Flask(__name__)
if not local_test:
    base_work_dir = '/mnt/A/dlp/'
else:
    base_work_dir = '/mnt/data1/zwzhu/A/dlp/'

def configure_logging():
    logger = logging.getLogger('my_flask_app')
    logger.setLevel(logging.INFO)

    handler = TimedRotatingFileHandler('/opt/dlp-worker/logs/logback-info.log', when='d', interval=1, backupCount=7)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)

if not local_test:
    configure_logging()

label_worker = LabelWorker(root=base_work_dir)


@app.route('/')
def home():
    app.logger.info('Hello, Flask logging!')
    return "Hello, Flask logging!"


def _img_id_to_path(img_id):
    return f"/mnt/A/{img_id}.jpg"


@app.post("/get_emb")
def get_emb():
    try:
        # if not request.is_json:
        #     return jsonify({"error": "Request must be JSON"}), 400
        request_data = request.get_json()

        image_id = str(request_data["image_dbid"])
        user_id = str(request_data["user_id"])
        task_id = str(request_data["task_id"])

        app.logger.info(f"Processing image {image_id} for user {user_id} and task {task_id}")

        # Retrieve the embedding, assumed function get_embedding_sam() generates a numpy array
        # emb = label_worker.get_emb_for_sam(user_id, task_id, image_id)
        # ! This is a temporary solution to demo the API
        embeddings = label_worker.get_emb_for_sam(user_id=user_id, task_id="", sel_dbid=image_id) # disabled task id for raw images

        # Save the numpy array to a buffer using np.save, which writes in .npy format
        buffer = io.BytesIO()
        np.save(buffer, embeddings)
        buffer.seek(0)  # Important: reset the buffer's position to the beginning

        # Create a response object with the buffer content and specify the correct MIME type
        response = Response(buffer.getvalue(), mimetype='application/octet-stream')
        return response

        # return jsonify({"message": emb}), 200

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 400


@app.post("/get_similar_mask")
def get_similar_mask():
    try:
        request_data = request.get_json()

        image_id = str(request_data["imageDbId"])
        user_id = str(request_data["userId"])
        task_id = str(request_data["taskId"])

        coordinates = request_data["coordinates"]  # [{"x":34,"y":38,"type":1},{"x":67,"y":61,"type":1}]
        label_space = request_data["labelSpace"]  # ["apple", "plum", "pear"]
        tag = request_data["tag"]  # "apple"

        tinyFlag = request_data["tinyFlag"]

        coordinate_list = [(coord["x"], coord["y"], coord["type"]) for coord in coordinates]

        app.logger.info(
            f"Processing image {image_id} for user {user_id} and task {task_id}, coordinate {coordinate_list}, label_space {label_space}, tag {tag}")

        tiny_image_id_score = label_worker.predict_sam(user_id=user_id, task_id=task_id, sel_dbid=image_id,
                                                     coordinate=coordinate_list,  # currently disabled task id
                                                     label_space=label_space, tag=tag)
        app.logger.info(f"result: {tiny_image_id_score}")
        return jsonify(tiny_image_id_score), 200

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 400

@app.post("/get_similar_image")
def get_similar_image():
    request_data = request.get_json()

    image_id = str(request_data["imageDbId"])
    user_id = str(request_data["userId"])
    task_id = str(request_data["taskId"])

    label_space = request_data["labelSpace"]  # ["apple", "plum", "pear"]
    tag = request_data["tag"]  # "apple"

    try:
        similar_image_id_score = label_worker.predict_image(user_id=user_id, task_id=task_id, sel_dbid=image_id,tag=tag, label_space=label_space)
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 400
    return jsonify(similar_image_id_score), 200


@app.post("/get_similar_text")
def get_similar_text():
    request_data = request.get_json()

    text_id = str(request_data["textDbId"])
    text_content = str(request_data["textContent"])
    user_id = str(request_data["userId"])
    task_id = str(request_data["taskId"])

    label_space = request_data["labelSpace"]  # ["apple", "plum", "pear"]
    tag = request_data["tag"]  # "apple"

    try:
        similar_text_id_score = label_worker.predict_text(user_id=user_id, task_id=task_id, sel_dbid=text_id, sel_text=text_content, tag=tag, label_space=label_space)
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 400
    return jsonify(similar_text_id_score), 200

if __name__ == '__main__':
    app.wsgi_app = ProxyFix(app.wsgi_app)
    app.run(host='0.0.0.0', port=22083, debug=True)

import os
import sqlite3
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import inference
from dotenv import load_dotenv
import cv2
import numpy as np
import faiss
import uuid

load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize Roboflow model
model = inference.get_model("stamps-kh78w/2")

# SQLite and file storage setup
DB_PATH = "stamps.db"
STAMPS_DIR = "stamp_images"
FAISS_INDEX_PATH = "stamps_index.faiss"

if not os.path.exists(STAMPS_DIR):
    os.makedirs(STAMPS_DIR)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS stamps
                 (id TEXT PRIMARY KEY, image_path TEXT)"""
    )
    conn.commit()
    conn.close()


init_db()

# Initialize FAISS index
feature_dimension = 1000  # Adjust based on your feature extraction method
index = faiss.IndexFlatL2(feature_dimension)

# Load existing index if it exists
if os.path.exists(FAISS_INDEX_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)


def extract_features(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to a standard size
    resized = cv2.resize(gray, (128, 128))  # Adjusted to 128x128
    # Apply histogram equalization
    equalized = cv2.equalizeHist(resized)

    # Extract HOG features
    winSize = (128, 128)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    hog_features = hog.compute(equalized)

    # Extract SIFT features
    sift = cv2.SIFT_create()
    _, sift_features = sift.detectAndCompute(equalized, None)

    # Combine HOG and SIFT features
    combined_features = np.concatenate(
        [hog_features.flatten(), sift_features.flatten()]
    )

    # Ensure the feature vector has the correct dimension
    feature_dimension = 1000  # Adjust based on your feature extraction method
    if combined_features.shape[0] > feature_dimension:
        combined_features = combined_features[:feature_dimension]
    elif combined_features.shape[0] < feature_dimension:
        combined_features = np.pad(
            combined_features, (0, feature_dimension - combined_features.shape[0])
        )

    return combined_features.astype("float32")


def find_similar_stamps(features, k=5):
    distances, indices = index.search(features.reshape(1, -1), k)
    similar_stamps = []
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for i, distance in zip(indices[0], distances[0]):
        c.execute("SELECT id FROM stamps WHERE rowid = ?", (int(i) + 1,))
        result = c.fetchone()
        if result:
            similar_stamps.append((result[0], float(1 / (1 + distance))))
    conn.close()
    return similar_stamps


@app.route("/")
def serve_index():
    return send_from_directory("static", "index.html")


@app.route("/detect_stamps", methods=["POST"])
def detect_stamps():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files["image"]
    img_array = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)

    # Detect stamps using Roboflow
    results = model.infer(image=img_array)

    detected_stamps = []
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    for detection in results[0].predictions:
        x, y, w, h = (
            detection.x,
            detection.y,
            detection.width,
            detection.height,
        )
        stamp_img = img_array[
            int(y - h / 2) : int(y + h / 2), int(x - w / 2) : int(x + w / 2)
        ]

        features = extract_features(stamp_img)
        similar_stamps = find_similar_stamps(features)

        # Generate a unique ID for the stamp
        stamp_id = str(uuid.uuid4())

        # Save the stamp image to disk
        image_path = os.path.join(STAMPS_DIR, f"{stamp_id}.jpg")
        cv2.imwrite(image_path, stamp_img)

        # Save stamp data to SQLite
        c.execute(
            "INSERT INTO stamps (id, image_path) VALUES (?, ?)", (stamp_id, image_path)
        )

        # Add features to FAISS index
        index.add(features.reshape(1, -1))

        stamp_data = {
            "id": stamp_id,
            "bbox": [x, y, w, h],
            "similar_stamps": similar_stamps,
        }
        detected_stamps.append(stamp_data)

    conn.commit()
    conn.close()

    # Save updated FAISS index
    faiss.write_index(index, FAISS_INDEX_PATH)

    return jsonify({"stamps": detected_stamps})


@app.route("/detect_against_database", methods=["POST"])
def detect_against_database():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files["image"]
    img_array = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)

    # Detect stamps using Roboflow
    results = model.infer(image=img_array)

    all_similar_stamps = []
    for detection in results[0].predictions:
        x, y, w, h = (
            detection.x,
            detection.y,
            detection.width,
            detection.height,
        )

        stamp_img = img_array[
            int(y - h / 2) : int(y + h / 2), int(x - w / 2) : int(x + w / 2)
        ]

        features = extract_features(stamp_img)
        similar_stamps = find_similar_stamps(
            features, k=10
        )  # Increase limit for database-wide search

        stamp_data = {
            "bbox": [x, y, w, h],
            "similar_stamps": [
                {
                    "id": stamp_id,
                    "similarity": float(similarity),
                }
                for stamp_id, similarity in similar_stamps
            ],
        }
        all_similar_stamps.append(stamp_data)

    return jsonify({"similar_stamps": all_similar_stamps})


@app.route("/stamp_image/<stamp_id>", methods=["GET"])
def get_stamp_image(stamp_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT image_path FROM stamps WHERE id = ?", (stamp_id,))
    result = c.fetchone()
    conn.close()

    if result:
        image_path = result[0]
        return send_file(image_path, mimetype="image/jpeg")
    else:
        return jsonify({"error": "Stamp not found"}), 404


if __name__ == "__main__":
    app.run(debug=True)

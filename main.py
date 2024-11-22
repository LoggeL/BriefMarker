import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sqlite3
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import time
import uuid
from io import BytesIO
import base64

import numpy as np
import cv2
import faiss
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import inference
from dotenv import load_dotenv
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Application constants
FEATURE_DIMENSION = 512
HOG_FEATURES_SIZE = 512
SIFT_FEATURES_SIZE = 512
MAX_IMAGE_SIZE = 4000
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
DEFAULT_CONFIDENCE_THRESHOLD = 0.5


@dataclass
class Config:
    """Application configuration settings"""

    db_path: str = "stamps.db"
    stamps_dir: str = "stamp_images"
    faiss_index_path: str = "stamps_index.faiss"
    max_image_size: int = MAX_IMAGE_SIZE
    feature_dimension: int = FEATURE_DIMENSION
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD


class StampDetectionApp:
    """Main application class for stamp detection and processing"""

    def __init__(self, config: Config):
        """Initialize application components"""
        self.config = config
        self.app = Flask(__name__)
        CORS(self.app)

        # Create required directories
        Path(self.config.stamps_dir).mkdir(parents=True, exist_ok=True)

        # Create static directories
        Path("static").mkdir(parents=True, exist_ok=True)
        Path("static/assets").mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._init_db()
        self.index = self._init_faiss()  # Single index for deep features
        self._init_models()
        self._setup_routes()

    def _init_models(self):
        """Initialize ML models and transforms"""
        # Roboflow model for stamp detection
        self.model = inference.get_model("stamps-kh78w/3")

        # ResNet model for feature extraction
        self.feature_extractor = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )
        self.feature_extractor.fc = torch.nn.Identity()
        self.feature_extractor.eval()

        # Image preprocessing transforms
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _init_db(self) -> None:
        """Initialize SQLite database schema"""
        try:
            with sqlite3.connect(self.config.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS stamps (
                        id TEXT PRIMARY KEY,
                        image_path TEXT,
                        features BLOB,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise

    def _init_faiss(self) -> faiss.Index:
        """Initialize FAISS similarity search index"""
        try:
            index_path = self.config.faiss_index_path

            if os.path.exists(index_path):
                index = faiss.read_index(index_path)
            else:
                index = faiss.IndexFlatL2(FEATURE_DIMENSION)  # For deep features
                faiss.write_index(index, index_path)

            return index

        except Exception as e:
            logger.error(f"FAISS initialization error: {e}")
            raise

    def _setup_routes(self) -> None:
        """Configure Flask routes"""
        self.app.route("/")(self.serve_index)
        self.app.route("/static/<path:filename>")(self.serve_static)
        self.app.route("/detect_stamps", methods=["POST"])(self.detect_stamps)
        self.app.route("/stamp_image/<stamp_id>", methods=["GET"])(self.get_stamp_image)
        self.app.route("/save_stamps", methods=["POST"])(self.save_stamps)
        self.app.route("/stamp_count", methods=["GET"])(self.get_stamp_count)

    @staticmethod
    def _validate_image(image_data: bytes) -> bool:
        """Validate uploaded image format"""
        try:
            img = Image.open(BytesIO(image_data))
            return img.format.lower() in ALLOWED_EXTENSIONS
        except Exception:
            return False

    @staticmethod
    def _preprocess_image(image: np.ndarray) -> np.ndarray:
        """Preprocess image for feature extraction"""
        if image is None or image.size == 0:
            raise ValueError("Invalid image data")

        # Convert to grayscale if needed
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        )

        # Standardize size and enhance contrast
        resized = cv2.resize(gray, (128, 128))
        return cv2.equalizeHist(resized)

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract deep learning features"""
        try:
            with torch.no_grad():
                img_tensor = self.transform(image).unsqueeze(0)
                features = self.feature_extractor(img_tensor).numpy().flatten()
                assert (
                    features.shape[0] == FEATURE_DIMENSION
                ), f"Expected {FEATURE_DIMENSION} features, got {features.shape[0]}"
                features = features / (np.linalg.norm(features) + 1e-7)
                return features.astype("float32")

        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            raise

    def find_similar_stamps(
        self, features: np.ndarray, k: int = 5
    ) -> Dict[str, List[Dict[str, Union[str, float]]]]:
        """Find similar stamps using deep features"""
        try:
            # Check if index is empty
            if self.index.ntotal == 0:
                return {"matches": []}

            # Debug logging
            logger.info(
                f"Feature shape: {features.shape}, Index dimension: {self.index.d}"
            )

            # Ensure features match the expected dimension
            features = features.astype("float32")  # FAISS requires float32
            if len(features.shape) == 1:
                features = features.reshape(1, -1)

            # Verify dimensions match
            if features.shape[1] != self.index.d:
                logger.error(
                    f"Feature dimension mismatch. Expected {self.index.d}, got {features.shape[1]}"
                )
                return {"matches": []}

            distances, indices = self.index.search(features, k)

            matches = []
            with sqlite3.connect(self.config.db_path) as conn:
                cursor = conn.cursor()

                for idx, dist in enumerate(distances[0]):
                    if idx >= len(indices[0]):  # Guard against index out of bounds
                        continue

                    cursor.execute(
                        "SELECT id FROM stamps WHERE rowid = ?",
                        (int(indices[0][idx]) + 1,),
                    )
                    if result := cursor.fetchone():
                        similarity = float(1 / (1 + dist))
                        matches.append({"id": result[0], "similarity": similarity})

            return {"matches": matches}

        except Exception as e:
            logger.error(f"Error finding similar stamps: {str(e)}")
            return {"matches": []}

    def save_stamps(self) -> Dict:
        """Process and save detected stamps"""
        try:
            data = request.get_json()
            if not data or "stamps" not in data:
                return jsonify({"error": "No stamps data provided"}), 400

            stamps = data["stamps"]
            saved_count = 0

            for stamp in stamps:
                try:
                    # Extract base64 image data
                    image_parts = stamp["imageUrl"].split(",")
                    if len(image_parts) != 2:
                        continue
                    image_data = image_parts[1]
                    image_bytes = base64.b64decode(image_data)

                    # Convert to numpy array
                    img_array = cv2.imdecode(
                        np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR
                    )

                    if img_array is None:
                        continue

                    # Generate unique ID and save image
                    stamp_id = str(uuid.uuid4())
                    image_path = os.path.join(self.config.stamps_dir, f"{stamp_id}.jpg")
                    os.makedirs(self.config.stamps_dir, exist_ok=True)

                    if not cv2.imwrite(image_path, img_array):
                        raise IOError(f"Failed to save image to {image_path}")

                    # Process features
                    features = self.extract_features(img_array)

                    # Update database
                    with sqlite3.connect(self.config.db_path) as conn:
                        conn.execute(
                            """INSERT INTO stamps 
                               (id, image_path, features) 
                               VALUES (?, ?, ?)""",
                            (
                                stamp_id,
                                image_path,
                                features.tobytes(),
                            ),
                        )

                    # Add features to index
                    self.index.add(features.reshape(1, -1))
                    saved_count += 1

                except Exception as e:
                    logger.warning(f"Failed to process stamp: {str(e)}")
                    continue

            # Save updated index if any stamps were processed
            if saved_count > 0:
                try:
                    os.makedirs(
                        os.path.dirname(self.config.faiss_index_path), exist_ok=True
                    )
                    faiss.write_index(self.index, self.config.faiss_index_path)
                    logger.info(f"Saved FAISS index: {self.config.faiss_index_path}")
                except Exception as e:
                    logger.error(f"Failed to save FAISS index: {str(e)}")

            return jsonify(
                {"message": "Stamps saved successfully", "saved_count": saved_count}
            )

        except Exception as e:
            logger.error(f"Error saving stamps: {str(e)}")
            return jsonify({"error": f"Failed to save stamps: {str(e)}"}), 500

    def detect_stamps(self) -> Dict:
        """Detect and analyze stamps in uploaded image"""
        try:
            # Validate input
            if "image" not in request.files:
                return jsonify({"error": "No image provided"}), 400

            image_file = request.files["image"]
            image_data = image_file.read()

            if not self._validate_image(image_data):
                return jsonify({"error": "Invalid image format"}), 400

            # Process image
            img_array = cv2.imdecode(
                np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR
            )

            # Write image to file
            cv2.imwrite("temp.jpg", img_array)

            # Detect stamps
            results = self.model.infer(image=img_array)
            detected_stamps = []

            for detection in results[0].predictions:
                if detection.confidence < self.config.confidence_threshold:
                    continue

                # Extract stamp region
                x, y = detection.x, detection.y
                w, h = detection.width, detection.height
                stamp_img = img_array[
                    int(y - h / 2) : int(y + h / 2), int(x - w / 2) : int(x + w / 2)
                ]

                # Find similar stamps
                features = self.extract_features(stamp_img)
                similar_stamps = self.find_similar_stamps(features)

                detected_stamps.append(
                    {
                        "id": str(uuid.uuid4()),
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "confidence": float(detection.confidence),
                        "similar_stamps": similar_stamps,  # Now contains separate deep and SIFT matches
                    }
                )

            return jsonify(
                {"stamps": detected_stamps, "total_detected": len(detected_stamps)}
            )

        except Exception as e:
            logger.error(f"Error detecting stamps: {e}")
            return jsonify({"error": str(e)}), 500

    def serve_index(self):
        """Serve the main application page"""
        return send_from_directory("static", "index.html")

    def get_stamp_image(self, stamp_id: str):
        """Retrieve a stamp image by ID"""
        try:
            with sqlite3.connect(self.config.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT image_path FROM stamps WHERE id = ?", (stamp_id,)
                )
                if result := cursor.fetchone():
                    return send_file(result[0], mimetype="image/jpeg")
                return jsonify({"error": "Stamp not found"}), 404

        except Exception as e:
            logger.error(f"Error retrieving stamp image: {e}")
            return jsonify({"error": str(e)}), 500

    def get_stamp_count(self):
        """Get total number of stamps in database"""
        try:
            with sqlite3.connect(self.config.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM stamps")
                count = cursor.fetchone()[0]
                return jsonify({"count": count})
        except Exception as e:
            logger.error(f"Error getting stamp count: {e}")
            return jsonify({"error": str(e)}), 500

    def serve_static(self, filename):
        """Serve static files"""
        return send_from_directory("static", filename)


def create_app(config: Optional[Config] = None) -> Flask:
    """Create and configure Flask application instance"""
    if config is None:
        config = Config()
    stamp_app = StampDetectionApp(config)
    return stamp_app.app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)

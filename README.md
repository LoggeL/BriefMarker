# Stamp Deduplication Tool

## Overview

The Stamp Deduplication Tool is a web-based application designed to detect and analyze stamps in images, find similar stamps in a database, and help manage large collections of stamps. It uses computer vision techniques and efficient similarity search algorithms to process images containing multiple stamps.

Key features:

- Detect stamps in images using the Roboflow API
- Extract features from detected stamps
- Find similar stamps in the database using FAISS (Facebook AI Similarity Search)
- Store stamp metadata and images for future reference
- Web interface for easy interaction with the tool

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [API Endpoints](#api-endpoints)
6. [Customization](#customization)
7. [Troubleshooting](#troubleshooting)
8. [Contributing](#contributing)
9. [License](#license)

## Requirements

- Python 3.7+
- Flask
- OpenCV
- NumPy
- FAISS
- SQLite
- Roboflow API key

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/stamp-deduplication-tool.git
   cd stamp-deduplication-tool
   ```

2. Create a virtual environment and activate it:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

4. Set up your Roboflow API key:
   Create a `.env` file in the project root and add your API key:

   ```
   ROBOFLOW_API_KEY=your_api_key_here
   ```

5. Initialize the database and FAISS index:
   ```
   python initialize_db.py
   ```

## Usage

1. Start the Flask server:

   ```
   python main.py
   ```

2. Open the `localhost:5000` URL in your web browser to access the Stamp Deduplication Tool.

3. Allow access to your webcam when prompted.

4. Use the "Capture Image" button to detect stamps in the current webcam view.

5. Use the "Detect Against Database" button to find similar stamps in the entire database.

## Project Structure

- `main.py`: Main Flask application
- `index.html`: Frontend web interface
- `stamps.db`: SQLite database for stamp metadata
- `stamps_index.faiss`: FAISS index for efficient similarity search
- `stamp_images/`: Directory for storing detected stamp images

## API Endpoints

- `POST /detect_stamps`: Detect stamps in the uploaded image and find similar stamps
- `POST /detect_against_database`: Detect stamps and search the entire database for similar stamps
- `GET /stamp_image/<stamp_id>`: Retrieve a specific stamp image

## Customization

- To modify the feature extraction method, update the `extract_features` function in `main.py`.
- To change the similarity search parameters, adjust the `find_similar_stamps` function.
- To update the Roboflow model, modify the `model = inference.get_model("stamps-kh78w/2")` line with your new model ID.

## Troubleshooting

- If you encounter issues with FAISS, ensure you have the correct version installed for your system.
- Make sure your Roboflow API key is correctly set in the `.env` file.
- Check that the `stamp_images` directory exists and is writable.

## Contributing

Contributions to the Stamp Deduplication Tool are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to your fork
5. Submit a pull request

## License

This project is licensed under the MIT License.

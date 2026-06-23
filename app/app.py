import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from flask import Flask, render_template, request, send_from_directory

from src.recommender import recommend_image

app = Flask(__name__)

UPLOAD_FOLDER = Path("app/static/uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


BASE_DIR = Path(__file__).resolve().parent.parent

# Deployment configuration
# True  = demo dataset (recommended for deployment)
# False = full cleaned dataset (local development)

USE_DEMO = True 

if USE_DEMO:
    DATASET_IMAGE_FOLDER = BASE_DIR / "data" / "demo" / "images"
else:
    DATASET_IMAGE_FOLDER = BASE_DIR / "data" / "cleaned" / "images"


@app.route("/dataset_image/<path:filename>")
def dataset_image(filename):
   return send_from_directory(DATASET_IMAGE_FOLDER, filename)

@app.route("/", methods=["GET", "POST"])
def home():

    uploaded_image = None
    recommendations = []

    if request.method == "POST":

        image = request.files["image"]

        if image:

            image_path = UPLOAD_FOLDER / image.filename
            image.save(image_path)

            uploaded_image = f"uploads/{image.filename}"

            recommendations = [
                    {
                        "image_name": image_name,
                        "score": score
                    }
                    for image_name, score in recommend_image(image_path)
            ]

    return render_template(
        "index.html",
        uploaded_image=uploaded_image,
        recommendations=recommendations
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.nn.functional import cosine_similarity
from torchvision import models, transforms


# Deployment configuration
# True  = demo dataset
# False = full cleaned dataset

USE_DEMO = True

if USE_DEMO:
    images_path = Path("data/demo/images")
    EMBEDDINGS_PATH = "embeddings/demo_embeddings.pt"
    IMAGE_NAMES_PATH = "embeddings/demo_image_names.csv"
else:
    images_path = Path("data/cleaned/images")
    EMBEDDINGS_PATH = "embeddings/fashion_embeddings.pt"
    IMAGE_NAMES_PATH = "embeddings/fashion_image_names.csv"


# Image preprocessing used by ResNet50
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def get_feature_extractor():
    """
    Load ResNet50 and remove the final
    classification layer so it outputs
    image features instead of class labels.
    """

    model = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    )

    model = torch.nn.Sequential(
        *list(model.children())[:-1]
    )

    model.eval()

    return model


# Load model
model = get_feature_extractor()

# Load saved embeddings
embeddings = torch.load(
    EMBEDDINGS_PATH,
    weights_only=True
)

# Load image names
image_names_df = pd.read_csv(
    IMAGE_NAMES_PATH
)


def recommend_image(query_image_path, top_n=5):

    # Load uploaded image
    image = Image.open(query_image_path).convert("RGB")

    # Apply preprocessing
    transformed_image = image_transform(image)

    # Add batch dimension
    input_tensor = transformed_image.unsqueeze(0)

    # Generate embedding
    with torch.no_grad():
        query_embedding = model(input_tensor)

    query_embedding = query_embedding.squeeze()

    similarities = []

    # Compare uploaded image with every image embedding
    for index, embedding in enumerate(embeddings):

        score = cosine_similarity(
            query_embedding,
            embedding,
            dim=0
        ).item()

        image_name = image_names_df.iloc[index]["image_name"]

        similarities.append(
            (image_name, score)
        )

    # Sort highest similarity first
    similarities.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # Keep only the best recommendations
    unique_recommendations = []

    for image_name, score in similarities:

        # Skip repeated near-identical matches
        if score > 0.95 and len(unique_recommendations) > 0:
            continue

        unique_recommendations.append(
            (image_name, score)
        )

        if len(unique_recommendations) == top_n:
            break

    return unique_recommendations
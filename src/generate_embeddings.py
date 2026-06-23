from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms


# USE_DEMO = True
# Uses the smaller deployment dataset.

# USE_DEMO = False
# Uses the full cleaned dataset.

USE_DEMO = True

if USE_DEMO:
    images_path = Path("data/demo/images")
    embeddings_file = "embeddings/demo_embeddings.pt"
    image_names_file = "embeddings/demo_image_names.csv"
else:
    images_path = Path("data/cleaned/images")
    embeddings_file = "embeddings/fashion_embeddings.pt"
    image_names_file = "embeddings/fashion_image_names.csv"


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

    model = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    )

    model = torch.nn.Sequential(
        *list(model.children())[:-1]
    )

    model.eval()

    return model


image_files = list(
    images_path.glob("*.jpg")
)

model = get_feature_extractor()

all_embeddings = []
image_names = []

for index, image_file in enumerate(image_files, start=1):

    if index % 1000 == 0:
        print(f"Processed {index} images...")

    image = Image.open(
        image_file
    ).convert("RGB")

    transformed_image = image_transform(
        image
    )

    input_tensor = transformed_image.unsqueeze(0)

    with torch.no_grad():
        embedding = model(input_tensor)

    # Convert from [1, 2048, 1, 1] to [2048]
    embedding = embedding.squeeze()

    all_embeddings.append(embedding)
    image_names.append(image_file.name)

# Convert list into a tensor
all_embeddings = torch.stack(
    all_embeddings
)

# Save embeddings
torch.save(
    all_embeddings,
    embeddings_file
)

# Save image names
image_names_df = pd.DataFrame(
    {"image_name": image_names}
)

image_names_df.to_csv(
    image_names_file,
    index=False
)

print("\nEmbedding generation complete")
print(f"\nTotal embeddings generated: {len(image_names)}")
print(f"Embedding tensor shape: {all_embeddings.shape}")
print(f"Saved embeddings to: {embeddings_file}")
print(f"Saved image names to: {image_names_file}")
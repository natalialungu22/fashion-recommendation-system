
import matplotlib.pyplot as plt
from PIL import Image

from recommender import recommend_image, images_path


# Choose one image to test
query_image_path = next(images_path.glob("*.jpg"))

# Get top 5 recommendations
top_recommendations = recommend_image(query_image_path)

# Load query image
query_image = Image.open(query_image_path).convert("RGB")

# Plot input image + recommendations
fig, axes = plt.subplots(1, 6, figsize=(18, 4))

axes[0].imshow(query_image)
axes[0].set_title("Input Image")
axes[0].axis("off")

for ax, (image_name, score) in zip(axes[1:], top_recommendations):
    recommended_image = Image.open(images_path / image_name).convert("RGB")

    ax.imshow(recommended_image)
    ax.set_title(f"Similarity: {score * 100:.2f}%")
    ax.axis("off")

plt.tight_layout()
plt.show()
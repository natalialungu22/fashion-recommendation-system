from pathlib import Path
import shutil

# create a smaller demo version for live web deployment.

cleaned_images_path = Path("data/cleaned/images")
demo_images_path = Path("data/demo/images")

demo_images_path.mkdir(parents=True, exist_ok=True)

# Number of images to use for deployment demo
DEMO_SIZE = 500

image_files = list(cleaned_images_path.glob("*.jpg"))[:DEMO_SIZE]

# Clear old demo images
for old_file in demo_images_path.glob("*.jpg"):
    old_file.unlink()

for image_file in image_files:
    shutil.copy2(image_file, demo_images_path / image_file.name)

print("Demo dataset created")
print(f"\nDemo images copied: {len(image_files)}")
print(f"Saved to: {demo_images_path}")
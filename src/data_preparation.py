from pathlib import Path
from PIL import Image
import hashlib
import shutil

# prepare the raw image dataset for modelling.

raw_images_path = Path("data/raw/images")
cleaned_images_path = Path("data/cleaned/images")

cleaned_images_path.mkdir(parents=True, exist_ok=True)

# Load all JPG image paths from the raw dataset
image_files = list(raw_images_path.glob("*.jpg"))

print("Dataset Summary")
print(f"\nTotal raw JPG images: {len(image_files)}")


# Check invalid images

invalid_images = []

for index, image_file in enumerate(image_files, start=1):

    if index % 1000 == 0:
        print(f"Checked {index} images for validity...")

    try:
        # Verify that the image can be opened correctly
        with Image.open(image_file) as img:
            img.verify()

    except Exception:
        # Store corrupted or unreadable image names
        invalid_images.append(image_file.name)

print("\nInvalid Image Check")
print(f"\nInvalid images found: {len(invalid_images)}")


# Check duplicate images

hashes = {}
duplicate_images = []

for index, image_file in enumerate(image_files, start=1):

    if index % 1000 == 0:
        print(f"Checked {index} images for duplicates...")

    # Read the image as binary data and create a hash fingerprint
    with open(image_file, "rb") as file:
        file_hash = hashlib.md5(file.read()).hexdigest()

    # If the hash already exists, the image is an exact duplicate
    if file_hash in hashes:
        duplicate_images.append(image_file.name)
    else:
        hashes[file_hash] = image_file.name

print("\nDuplicate Image Check")
print(f"\nDuplicate images found: {len(duplicate_images)}")



# Create cleaned dataset

# Clear old cleaned images before creating a new cleaned dataset
for old_file in cleaned_images_path.glob("*.jpg"):
    old_file.unlink()

seen_hashes = set()
cleaned_count = 0

for index, image_file in enumerate(image_files, start=1):

    if index % 1000 == 0:
        print(f"Copied {index} raw images checked for cleaned dataset...")

    # Skip invalid images
    if image_file.name in invalid_images:
        continue

    with open(image_file, "rb") as file:
        file_hash = hashlib.md5(file.read()).hexdigest()

    # Skip duplicate images
    if file_hash in seen_hashes:
        continue

    seen_hashes.add(file_hash)

    # Copy only valid and unique images to the cleaned folder
    output_file = cleaned_images_path / image_file.name
    shutil.copy2(image_file, output_file)

    cleaned_count += 1


print(f"Total raw images checked: {len(image_files)}")
print(f"Invalid images removed: {len(invalid_images)}")
print(f"Duplicate images removed: {len(duplicate_images)}")
print(f"Cleaned images saved: {cleaned_count}")
print(f"Cleaned folder: {cleaned_images_path}")
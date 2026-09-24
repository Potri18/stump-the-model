import os
import glob
from src.matcher import CatalogueMatcher

print("Initializing Production Catalogue Matcher...")
matcher = CatalogueMatcher(catalogue_dir="data/catalogue", artifacts_dir="artifacts")

# Load existing production FAISS index
print("Loading existing production FAISS index...")
matcher.load_index()

# Robust Edge-Case Handling: Automatically find ANY image file starting with 'unknown' 
# regardless of whether it's .jpg, .jpeg, .png, .webp, or .jfif
valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.jfif', '.bmp', '.JPG', '.JPEG', '.PNG')
unknown_files = glob.glob("unknown.*")

unknown_img_path = None
for file in unknown_files:
    if file.endswith(valid_extensions):
        unknown_img_path = file
        break

if unknown_img_path:
    print(f"\nTesting production unknown item rejection on: {unknown_img_path}")
    result = matcher.query(unknown_img_path, top_k=5)
    print("Production Query Results:")
    print(result)
else:
    print("Error: Could not find any valid image file starting with 'unknown' in the root folder!")
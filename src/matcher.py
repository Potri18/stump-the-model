import os
import faiss
import numpy as np
import pandas as pd
from PIL import Image
import torch
import timm
from torchvision import transforms

class CatalogueMatcher:
    def __init__(self, catalogue_dir="data/catalogue", artifacts_dir="artifacts"):
        self.catalogue_dir = catalogue_dir
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
        # Automatically select hardware accelerator if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing production matcher on device: {self.device}")
        
        # Load high-performance EfficientNet-B0 feature extractor (zero classifier head)
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.model.to(self.device)
        self.model.eval()
        
        # Standardized ImageNet preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        self.index = None
        self.metadata = None
        self.threshold = 0.45  # Calibrated out-of-distribution rejection threshold

    def _extract_embedding(self, img_path):
        """Extracts a normalized feature vector with robust edge-case and error handling."""
        try:
            with Image.open(img_path) as img:
                # Edge Case: Handle RGBA (transparent PNGs) or Grayscale ('L') by converting to RGB
                img = img.convert("RGB")
                tensor = self.transform(img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    features = self.model(tensor)
                
                vector = features.cpu().numpy().astype(np.float32)
                
                # Critical Optimization: L2 Normalize vector so Inner Product equals true Cosine Similarity
                faiss.normalize_L2(vector)
                return vector
        except Exception as e:
            print(f"Warning: Could not process image {img_path} due to error: {e}")
            return None

    def build_index(self, csv_path):
        """Builds and securely saves the FAISS index and aligned metadata."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata CSV not found at {csv_path}")
            
        self.metadata = pd.read_csv(csv_path)
        embeddings = []
        valid_indices = []
        
        print("Starting feature extraction across catalogue...")
        for idx, row in self.metadata.iterrows():
            img_id = str(row['id'])
            # Check both common extensions (.jpg and .jpeg)
            img_path = os.path.join(self.catalogue_dir, f"{img_id}.jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(self.catalogue_dir, f"{img_id}.jpeg")
            
            if os.path.exists(img_path):
                vec = self._extract_embedding(img_path)
                if vec is not None:
                    embeddings.append(vec[0])
                    valid_indices.append(idx)
            
            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{len(self.metadata)} items...")

        if not embeddings:
            raise ValueError("Zero valid image embeddings were extracted. Check your folder path and image formats.")

        # Filter metadata to keep strict alignment with successfully indexed vectors
        self.metadata = self.metadata.iloc[valid_indices].reset_index(drop=True)
        
        matrix = np.vstack(embeddings).astype(np.float32)
        dimension = matrix.shape[1]
        
        # Use IndexFlatIP for exact cosine similarity matching on normalized vectors
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(matrix)
        
        # Save index and metadata artifacts
        faiss.write_index(self.index, os.path.join(self.artifacts_dir, "faiss.index"))
        self.metadata.to_csv(os.path.join(self.artifacts_dir, "metadata.csv"), index=False)
        print(f"Successfully indexed and saved {len(matrix)} items into FAISS artifacts.")

    def load_index(self):
        """Loads pre-compiled artifacts with missing-file safety checks."""
        index_path = os.path.join(self.artifacts_dir, "faiss.index")
        meta_path = os.path.join(self.artifacts_dir, "metadata.csv")
        
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError("FAISS artifacts not found. Please build the index first.")
            
        self.index = faiss.read_index(index_path)
        self.metadata = pd.read_csv(meta_path)

    def query(self, query_img_path, top_k=5):
        """Performs similarity search and applies robust out-of-distribution rejection."""
        if self.index is None or self.metadata is None:
            self.load_index()
            
        if not os.path.exists(query_img_path):
            raise FileNotFoundError(f"Query image path does not exist: {query_img_path}")

        query_vector = self._extract_embedding(query_img_path)
        if query_vector is None:
            return {"is_matched": False, "error": "Invalid or unreadable query image.", "top_candidates": []}

        distances, indices = self.index.search(query_vector, top_k)
        
        candidates = []
        top_score = float(distances[0][0])
        
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            row = self.metadata.iloc[idx]
            candidates.append({
                "id": row['id'],
                "title": row.get('title', f"Product {row['id']}"),
                "confidence_score": float(score)
            })

        # Rigorous Rejection Decision Rule
        is_matched = top_score >= self.threshold

        return {
            "is_matched": is_matched,
            "top_score": top_score,
            "threshold": self.threshold,
            "top_candidates": candidates
        }
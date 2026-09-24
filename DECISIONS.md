# Architectural Decisions & Engineering Trade-Offs
**Project:** Visual Search & Out-of-Distribution Rejection Pipeline  
**Candidate:** Potri Selvan

---

## 1. Architecture Choices & What Was Rejected
* **Chosen Backbone (EfficientNet-B0 via `timm`):** Selected for its optimal balance between high representation capacity and lightweight inference speed. Unlike heavier models (e.g., ResNet-50 or ViT-Base), EfficientNet-B0 extracts robust 1280-dimensional feature maps quickly on standard CPU hardware without requiring heavy GPU resources.
* **Rejected Backbone (ResNet-50 / CLIP):** ResNet-50 yields higher vector dimensionality with diminishing returns on retail apparel classification for this scale, while CLIP introduces excessive latency and uncalibrated zero-shot alignment noise for fine-grained retail item matching.
* **Vector Indexing Engine (FAISS `IndexFlatIP`):** Chosen over tree-based or approximate nearest neighbor indices (like HNSW or Annoy) because our catalogue size (~5,100 items) is small enough for brute-force exact search to run in sub-millisecond time. This guarantees **100% recall** with zero quantization loss.
* **Mathematical Guarantee (L2 Normalization):** All feature vectors are strictly L2-normalized ($\hat{u} = \frac{u}{\Vert{}u\Vert{}_2}$) prior to insertion and querying. This forces FAISS's Inner Product index (`IndexFlatIP`) to mathematically compute true Cosine Similarity ($\langle \hat{u}, \hat{v} \rangle \in [-1, 1]$), ensuring stable, bounded confidence scores regardless of image scaling or pixel brightness variations.

## 2. Trade-Offs Made Under the Time Limit
* **Exact Search vs. Approximate Scaling:** Opted for `IndexFlatIP` over compressed indices (like IVFPQ). While IVFPQ scales better to millions of items, it introduces quantization error that harms fine-grained retrieval accuracy on small catalogues. For 5,000 items, exact search is faster and perfectly accurate.
* **Static Thresholding vs. Dynamic Calibration:** Implemented a fixed out-of-distribution rejection threshold ($\tau = 0.45$) derived empirically from testing foreign objects. While a dynamic threshold based on local neighborhood density would be more adaptive, a static calibrated threshold provides predictable, explainable guardrails under a tight timeline.

## 3. Testing Methodology & Evaluation
* **In-Distribution Verification:** Tested via self-querying catalogue assets, confirming exact self-matches return a top confidence score of `1.0000`.
* **Out-of-Distribution ("Stump-the-Model") Testing:** Evaluated against foreign out-of-distribution objects (e.g., electronic accessories and random non-retail items). The system successfully scored them below the threshold (~0.338) and flagged them with `is_matched: False`, preventing false hallucinations.

## 4. Where the System Breaks (Honest Failure Modes)
* **Severe Occlusion & Cluttered Backgrounds:** If a query image features multiple items or a heavy human hand/wrist obscuring more than 50% of the apparel item, the global average pooling of EfficientNet-B0 dilutes the foreground feature representation with background noise, causing a drop in confidence score.
* **Extreme Motion Blur & Lighting Artifacts:** Severely underexposed or motion-blurred phone captures shift the embedding vector away from the clean studio-shot catalogue distribution, occasionally leading to false rejection of valid items (false negatives).

## 5. What I Would Do Next (With Two More Weeks)
1. **Local Object Detection Integration:** Implement a lightweight YOLO or bounding-box cropping stage prior to feature extraction to automatically isolate the retail item from cluttered backgrounds.
2. **Hard Negative Mining:** Fine-tune the feature extractor embedding space using Triplet Loss with hard-negative pairs (e.g., distinguishing between two visually similar shirts with different patterns).
3. **Approximate Index Scaling:** Migrate to a FAISS `IndexIVFFlat` or HNSW graph structure to benchmark performance scaling past 100,000+ catalogue items while keeping latency under 100ms.
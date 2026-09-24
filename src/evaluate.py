import time
import os
import pandas as pd
from src.matcher import CatalogueMatcher

def run_evaluation(test_csv_path="data/evaluation/test_labels.csv"):
    matcher = CatalogueMatcher()
    if not matcher.load_index():
        print("Please build or load your FAISS index first.")
        return

    if not os.path.exists(test_csv_path):
        print(f"Test labels file not found at {test_csv_path}. Create it with columns: image_path, true_id, failure_condition")
        return

    df = pd.read_csv(test_csv_path)
    total = len(df)
    top1_correct = 0
    top5_correct = 0
    latencies = []
    condition_stats = {}

    print(f"Running evaluation across {total} test images...")

    for _, row in df.iterrows():
        img_path = row["image_path"]
        true_id = str(row["true_id"])
        condition = row["failure_condition"]

        if condition not in condition_stats:
            condition_stats[condition] = {"total": 0, "top1": 0, "top5": 0}

        condition_stats[condition]["total"] += 1

        start_time = time.time()
        result = matcher.query(img_path, top_k=5)
        latency_ms = (time.time() - start_time) * 1000
        latencies.append(latency_ms)

        candidates = result["top_candidates"]
        top1_id = str(candidates[0]["id"]) if len(candidates) > 0 else ""
        top5_ids = [str(c["id"]) for c in candidates]

        condition_stats[condition]["total"]
        if top1_id == true_id:
            top1_correct += 1
            condition_stats[condition]["top1"] += 1
        
        if true_id in top5_ids:
            top5_correct += 1
            condition_stats[condition]["top5"] += 1

    print("\n" + "="*40)
    print("EVALUATION RESULTS SUMMARY")
    print("="*40)
    print(f"Overall Top-1 Accuracy: {(top1_correct/total)*100:.2f}%")
    print(f"Overall Top-5 Accuracy: {(top5_correct/total)*100:.2f}%")
    print(f"Average Single-Image Latency: {sum(latencies)/len(latencies):.2f} ms")
    print("\nBreakdown by Failure Condition:")
    for cond, stats in condition_stats.items():
        acc_1 = (stats["top1"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f" - {cond}: Top-1 Acc = {acc_1:.1f}% ({stats['top1']}/{stats['total']})")
    print("="*40)

if __name__ == "__main__":
    run_evaluation()
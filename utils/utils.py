import os
import json
import csv

def collect_metrics(root_dir: str, out_csv: str = "all_metrics.csv"):
    rows = []
    fieldnames = set()

    # Loop over all subdirs in root_dir
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        metrics_file = os.path.join(subdir_path, "metrics.json")
        if not os.path.exists(metrics_file):
            continue

        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        # Add config name (subdir name) to the metrics
        row = {"config": subdir}
        row.update(metrics)
        rows.append(row)
        fieldnames.update(row.keys())

    # Save CSV
    fieldnames = sorted(fieldnames)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Collected {len(rows)} experiments into {out_csv}")

# Example usage
if __name__ == "__main__":
    collect_metrics("source/evaluation/foundationstereo_kitti12-training")

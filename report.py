import csv

best_f1 = 0
patience = 5
patience_counter = 0

metrics_file = "federated_metrics.csv"

# Write CSV header
with open(metrics_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["round","accuracy","auc","precision","recall","f1"])


def weighted_average(metrics, round_counter):

    global best_f1, patience_counter

    total_examples = sum(num_examples for num_examples, _ in metrics)

    accuracy = sum(num_examples * m["accuracy"] for num_examples, m in metrics) / total_examples
    auc = sum(num_examples * m["auc"] for num_examples, m in metrics) / total_examples
    precision = sum(num_examples * m["precision"] for num_examples, m in metrics) / total_examples
    recall = sum(num_examples * m["recall"] for num_examples, m in metrics) / total_examples
    f1 = sum(num_examples * m["f1"] for num_examples, m in metrics) / total_examples

    print(f"\nROUND {round_counter} GLOBAL METRICS")
    print("Accuracy:", accuracy)
    print("AUC:", auc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

    with open(metrics_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([round_counter, accuracy, auc, precision, recall, f1])

    # Early stopping
    if f1 > best_f1:
        best_f1 = f1
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("\nEarly stopping triggered")
        raise SystemExit("Training stopped")

    return {
        "accuracy": accuracy,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


round_counter = 0

def evaluate_metrics_aggregation_fn(metrics):
    global round_counter
    round_counter += 1
    return weighted_average(metrics, round_counter)
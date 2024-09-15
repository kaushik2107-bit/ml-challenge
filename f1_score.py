import argparse
import pandas as pd

def calculate_f1_score(ground_truth, predictions):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    for gt, out in zip(ground_truth, predictions):
        if out != "" and gt != "" and out == gt:
            true_positives += 1
        elif out != "" and gt != "" and out != gt:
            false_positives += 1
        elif out != "" and gt == "":
            false_positives += 1
        elif out == "" and gt != "":
            false_negatives += 1
        elif out == "" and gt == "":
            true_negatives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0

    return f1_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate F1 score based on predictions and ground truth")

    parser.add_argument("--ground_truth", type=str, required=True, help="Path to the ground truth file")
    parser.add_argument("--predictions", type=str, required=True, help="Path to the predictions file")

    args = parser.parse_args()

    ground_truth_df = pd.read_csv(args.ground_truth)
    predictions_df = pd.read_csv(args.predictions)

    ground_truth = ground_truth_df['entity_value'].fillna("").astype(str).tolist()
    predictions = predictions_df['entity_value'].fillna("").astype(str).tolist()

    f1_score = calculate_f1_score(ground_truth, predictions)
    print(f"F1 Score: {f1_score}")
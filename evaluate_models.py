import pandas as pd
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
from scipy.stats import pearsonr

# Models to evaluate
MODELS = {
    "Qwen 2.5 7B":  "valence_arousal_predictions-qwen.xlsx",
    "Qwen 2.5 7B - ZeroShot":  "valence_arousal_predictions-qwen-zeroshot.xlsx",
    "Qwen 2.5 72B":  "valence_arousal_predictions-qwen72.xlsx",
    "Qwen 2.5 72B - ZeroShot":  "valence_arousal_predictions-qwen72-zeroshot.xlsx",
    "Llama 3 8B":   "valence_arousal_predictions-llama.xlsx",
    "Llama 3 8B - ZeroShot":   "valence_arousal_predictions-llama-zeroshot.xlsx",
    "Llama 3 70B":   "valence_arousal_predictions-llama70.xlsx",
    "Llama 3 70B - ZeroShot":   "valence_arousal_predictions-llama70-zeroshot.xlsx"
}

# Load and combine ground truth
gt = pd.concat([
    pd.read_csv("static_annotations_averaged_songs_1_2000.csv"),
    pd.read_csv("static_annotations_averaged_songs_2000_2058.csv")
])
gt.columns = gt.columns.str.strip()
gt = gt.rename(columns={"song_id": "Id", "valence_mean": "gt_valence", "arousal_mean": "gt_arousal"})

# Assign each song to an emotional quadrant based on Russell circumplex model
def get_quadrant(valence, arousal):
    if valence >= 5 and arousal >= 5: return "HVHA" # Happy/Excited
    if valence <  5 and arousal >= 5: return "LVHA" # Angry/Tense
    if valence <  5 and arousal <  5: return "LVLA" # Sad/Depressed
    if valence >= 5 and arousal <  5: return "HVLA" # Calm/Relaxed

# Evaluate each model
rows = []
for model, path in MODELS.items():
    df = pd.read_excel(path).merge(gt, on="Id").dropna(subset=["predicted_valence", "predicted_arousal"])

    q_pred = df.apply(lambda r: get_quadrant(r["predicted_valence"], r["predicted_arousal"]), axis=1)
    q_gt   = df.apply(lambda r: get_quadrant(r["gt_valence"],        r["gt_arousal"]),        axis=1)

    rows.append({
        "Model":             model,
        "MAE Valence":       round(mean_absolute_error(df["gt_valence"], df["predicted_valence"]), 3),
        "MAE Arousal":       round(mean_absolute_error(df["gt_arousal"], df["predicted_arousal"]), 3),
        "Pearson r Valence": round(pearsonr(df["gt_valence"], df["predicted_valence"])[0], 3),
        "Pearson r Arousal": round(pearsonr(df["gt_arousal"], df["predicted_arousal"])[0], 3),
        "Weighted Kappa":    round(cohen_kappa_score(q_gt, q_pred, weights="linear"), 3),
    })

# Print and save results
results = pd.DataFrame(rows).set_index("Model")
print(results.to_string())

with pd.ExcelWriter("evaluation_results.xlsx") as writer:
    results.to_excel(writer, sheet_name="Metrics")

    rows_dist = []
    gt_q = gt.apply(lambda r: get_quadrant(r["gt_valence"], r["gt_arousal"]), axis=1)
    dist = gt_q.value_counts().sort_index()
    rows_dist.append({"Model": "Ground Truth", **dist.to_dict()})

    for model, path in MODELS.items():
        df = pd.read_excel(path).merge(gt, on="Id").dropna(subset=["predicted_valence", "predicted_arousal"])
        pred_q = df.apply(lambda r: get_quadrant(r["predicted_valence"], r["predicted_arousal"]), axis=1)
        dist = pred_q.value_counts().sort_index()
        rows_dist.append({"Model": model, **dist.to_dict()})

    df_dist = pd.DataFrame(rows_dist).set_index("Model")
    df_dist.to_excel(writer, sheet_name="Quadrant Distribution")

print("Saved to evaluation_results.xlsx")
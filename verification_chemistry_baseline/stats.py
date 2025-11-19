import re
import numpy as np

# Path to your log file
log_path = "/mnt/personal/hlavsja3/experiments2025/verification_chemistry_baseline/10117479/"
log_path += "/log"

# Lists to store parsed values
train_auc, val_auc = [], []
train_tpr, val_tpr = [], []

with open(log_path, "r") as f:
    for line in f:
        match = re.match(
            r"Split \d+: Train AUC=([0-9.]+), TPR@5%FPR=([0-9.]+), Val AUC=([0-9.]+), TPR@5%FPR=([0-9.]+)", line
        )
        if match:
            t_auc, t_tpr, v_auc, v_tpr = map(float, match.groups())
            train_auc.append(t_auc)
            train_tpr.append(t_tpr)
            val_auc.append(v_auc)
            val_tpr.append(v_tpr)

# Convert to numpy arrays
train_auc = np.array(train_auc)
train_tpr = np.array(train_tpr)
val_auc = np.array(val_auc)
val_tpr = np.array(val_tpr)

# Print means and standard deviations
print(f"Val AUC: {val_auc.mean():.3f} ± {val_auc.std():.3f}")
print(f"Val TPR@5%FPR: {val_tpr.mean():.3f} ± {val_tpr.std():.3f}")
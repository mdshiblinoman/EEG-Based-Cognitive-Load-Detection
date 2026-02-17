"""
Data Preprocessing Module
=========================
Loads EEG raw data from Arithmetic and Stroop tasks across 4 stress levels
(natural, lowlevel, midlevel, highlevel) for 15 participants each.
Extracts statistical features and prepares train/test datasets.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# ── Column names for the 25-column raw EEG data ──────────────────────────
RAW_COLUMNS = [
    "EEG_TP9", "EEG_AF7", "EEG_AF8", "EEG_TP10",
    "EEG_AUX_L", "EEG_AUX_R", "EEG_DRL", "EEG_REF",
    "ACC_X", "ACC_Y", "ACC_Z",
    "Battery",
    "Marker",
    "Color_R", "Color_G", "Color_B",
    "Param1", "Param2",
    "Zero1", "Zero2", "Zero3",
    "Timestamp_Unix", "Zero4",
    "Timestamp_Str"
]

# Only the numeric sensor columns we actually use for feature extraction
SENSOR_COLS = [
    "EEG_TP9", "EEG_AF7", "EEG_AF8", "EEG_TP10",
    "EEG_AUX_L", "EEG_AUX_R", "EEG_DRL", "EEG_REF",
    "ACC_X", "ACC_Y", "ACC_Z"
]

STRESS_LEVELS = ["natural", "lowlevel", "midlevel", "highlevel"]
STRESS_LABEL_MAP = {"natural": 0, "lowlevel": 1, "midlevel": 2, "highlevel": 3}
NUM_PARTICIPANTS = 15
TASK_TYPES = ["Arithmetic_Data ", "Stroop_Data"]  # note trailing space on Arithmetic

BASE_DIR = Path(__file__).parent / "raw_data"


def load_single_file(filepath: str) -> pd.DataFrame:
    """Load a single raw EEG text file into a DataFrame."""
    try:
        df = pd.read_csv(
            filepath,
            header=None,
            names=RAW_COLUMNS,
            skipinitialspace=True,
            on_bad_lines="skip",
        )
        # Keep only numeric sensor columns
        df = df[SENSOR_COLS].apply(pd.to_numeric, errors="coerce")
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"  [WARN] Could not load {filepath}: {e}")
        return pd.DataFrame(columns=SENSOR_COLS)


def extract_features(df: pd.DataFrame) -> dict:
    """
    Extract statistical features from a recording session.
    For each sensor column we compute: mean, std, min, max, median, skew, kurtosis
    """
    features = {}
    for col in SENSOR_COLS:
        if col in df.columns and len(df) > 0:
            features[f"{col}_mean"] = df[col].mean()
            features[f"{col}_std"] = df[col].std()
            features[f"{col}_min"] = df[col].min()
            features[f"{col}_max"] = df[col].max()
            features[f"{col}_median"] = df[col].median()
            features[f"{col}_skew"] = df[col].skew()
            features[f"{col}_kurtosis"] = df[col].kurtosis()
        else:
            for stat in ["mean", "std", "min", "max", "median", "skew", "kurtosis"]:
                features[f"{col}_{stat}"] = 0.0
    return features


def build_dataset() -> pd.DataFrame:
    """
    Walk through all task folders, stress levels and participants,
    extract features & return a single DataFrame ready for ML/DL.
    """
    rows = []
    for task in TASK_TYPES:
        task_dir = BASE_DIR / task
        if not task_dir.exists():
            print(f"[SKIP] Directory not found: {task_dir}")
            continue
        task_label = task.strip().replace("_Data", "")  # "Arithmetic" or "Stroop"

        for level in STRESS_LEVELS:
            for pid in range(1, NUM_PARTICIPANTS + 1):
                filename = f"{level}-{pid}.txt"
                filepath = task_dir / filename
                if not filepath.exists():
                    print(f"  [SKIP] {filepath}")
                    continue

                print(f"  Loading {task_label}/{level}-{pid} …")
                df = load_single_file(str(filepath))
                if df.empty:
                    continue

                feats = extract_features(df)
                feats["task"] = task_label
                feats["stress_level"] = STRESS_LABEL_MAP[level]
                feats["stress_label"] = level
                feats["participant"] = pid
                rows.append(feats)

    dataset = pd.DataFrame(rows)
    return dataset


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the list of numeric feature column names."""
    exclude = {"task", "stress_level", "stress_label", "participant"}
    return [c for c in df.columns if c not in exclude]


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Building dataset from raw EEG files …")
    print("=" * 60)
    dataset = build_dataset()

    out_path = Path(__file__).parent / "processed_dataset.csv"
    dataset.to_csv(out_path, index=False)
    print(f"\n✓ Saved processed dataset → {out_path}")
    print(f"  Shape: {dataset.shape}")
    print(f"  Stress distribution:\n{dataset['stress_label'].value_counts()}")
    print(f"  Task distribution:\n{dataset['task'].value_counts()}")

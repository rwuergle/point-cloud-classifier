import laspy
import copy
import numpy as np
from point_cloud_classifier.constants import FEATURE_RANGES, CLASSIFICATION_MAP, PROJECT_CLASSIFIED_MAP
from time import time
import os
from functools import wraps
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, balanced_accuracy_score, cohen_kappa_score
import seaborn as sns
import matplotlib.pyplot as plt
import json
import wandb
import pandas as pd

def getSingleIDperGroup(pc: laspy.LasData):
    mapping = np.arange(42, dtype=pc.classification.dtype)
    mapping[18] = 2
    mapping[31] = 2
    mapping[4] = 3
    mapping[5] = 3
    mapping[41] = 9
    
    pc.classification = mapping[pc.classification]
    return pc


def visualize_point_cloud(pc: laspy.LasData, classification: np.ndarray | None = None, outputName: str = "./visualization/visualization.laz", remove_extra_dims: bool = True):
        if classification is None:
            classification = pc.classification

        header = laspy.LasHeader(point_format=pc.header.point_format, version=pc.header.version)

        header.scales = pc.header.scales
        header.offsets = pc.header.offsets

        pcVis = laspy.LasData(header)
        pcVis.points = pc.points.copy()

        if remove_extra_dims:
            pcVis.remove_extra_dims(list(pcVis.point_format.extra_dimension_names))
        
        pcVis.classification = classification
        pcVis.write(outputName)

def visualize_point_cloud_classification(xyz: np.ndarray, classification: np.ndarray | None = None, outputName: str = "./visualization/visualization.laz"):
        if classification is None:
            classification = np.zeros(len(xyz), dtype=np.uint8)
        else:
            classification = classification.astype(np.uint8)

        output_dir = os.path.dirname(outputName)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        header = laspy.LasHeader(point_format=3, version="1.4")
        header.offsets = np.min(xyz, axis=0)
        header.scales = np.array([0.01, 0.01, 0.01])

        pc = laspy.LasData(header)
        pc.x = xyz[:, 0]
        pc.y = xyz[:, 1]
        pc.z = xyz[:, 2]
        pc.classification = classification
        pc.write(outputName)


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time: float = time()
        result: float = func(*args, **kwargs)
        end_time: float = time()
        print(f"{func.__name__} completed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def get_bounds(feature: str):
    match = [bounds for key, bounds in FEATURE_RANGES.items() if feature.startswith(key)]
    return match[0] if match else None

def get_data_summary(labels: np.ndarray, map: dict[int, str] = CLASSIFICATION_MAP):
    values, counts = np.unique(labels, return_counts=True)
    summary = {map[v]: count for v, count in zip(values, counts)}
    
    print("Summary".center(50, "="))
    for key, val in summary.items():
        print(f"{key:<20}: {val:>10}")
    print("=" * 50)

def get_accuracy(predicted, true):
    return np.mean(np.equal(predicted,true))

def get_precision(predicted, true):
    return precision_score(true, predicted)

def get_recall(predicted, true):
    return recall_score(true, predicted)

def get_f1(predicted, true):
    return f1_score(true, predicted)

def iou_score(preds, targets, eps=1e-6):
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection

    return ((intersection + eps) / (union + eps)).mean()


def get_classification_metrics(gt: np.ndarray, preds: np.ndarray, save_json: bool = True, save_plot: bool = True, save_wandb: bool = False, model_name: str = "default", tile_name: str = "default"):

    balanced_acc = balanced_accuracy_score(gt, preds)
    kappa = cohen_kappa_score(gt, preds)
    
    all_labels = list(PROJECT_CLASSIFIED_MAP.keys())
    target_names = list(PROJECT_CLASSIFIED_MAP.values())

    report = classification_report(gt, preds, labels=all_labels, target_names=target_names, digits=4, output_dict=True)

    if save_json:
        metrics = {"balanced_accuracy": float(balanced_acc),"cohen_kappa": float(kappa),"report_dict": report}
        save_path: str = f"./evaluations/{tile_name}_{model_name}.json"
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=4)

    if save_plot:
        plot_confusion_matrix(gt, preds, tile_name, model_name)
        
    if save_wandb:
        wandb.init(
            project="point-cloud-classifier",
            name=f"{model_name}_{tile_name}",
            group=model_name,
            tags=[model_name],
            config={
                "model": model_name,
                "tile": tile_name,
                "num_points": len(gt),
            }
        )

        wandb.log({
            "balanced_accuracy": balanced_acc,
            "cohen_kappa": kappa,
            "accuracy": report["accuracy"],
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
        })

        for label_name in PROJECT_CLASSIFIED_MAP.values():

            if label_name in report:

                wandb.log({
                    f"{label_name}_precision":
                        report[label_name]["precision"],

                    f"{label_name}_recall":
                        report[label_name]["recall"],

                    f"{label_name}_f1":
                        report[label_name]["f1-score"],
                })

        gt_mapped = np.searchsorted(all_labels, gt)
        preds_mapped = np.searchsorted(all_labels, preds)
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                y_true=gt_mapped,
                preds=preds_mapped,
                class_names=list(PROJECT_CLASSIFIED_MAP.values())
            )
        })

        wandb.finish()

    return metrics

def plot_confusion_matrix(gt: np.ndarray, preds: np.ndarray, tile_name: str = 'default', model_name: str = "default"):
    cm = confusion_matrix(gt, preds)
    all_labels = np.unique(gt)
    tick_labels = [f"{CLASSIFICATION_MAP[i]}" for i in all_labels]
    
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_true = np.divide(
        cm.astype('float'), row_sums, 
        out=np.zeros_like(cm, dtype=float), where=row_sums != 0
    )
    
    col_sums = cm.sum(axis=0)[np.newaxis, :]
    cm_pred = np.divide(
        cm.astype('float'), col_sums, 
        out=np.zeros_like(cm, dtype=float), where=col_sums != 0
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(
        cm_true, annot=True, fmt=".2f", cmap="Blues", 
        xticklabels=tick_labels, yticklabels=tick_labels, ax=ax1
    )
    ax1.set_title("Normalized by True Label\n(Per-Class Recall)")
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("True Label")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)

    sns.heatmap(
        cm_pred, annot=True, fmt=".2f", cmap="Greens",
        xticklabels=tick_labels, yticklabels=tick_labels, ax=ax2
    )
    ax2.set_title("Normalized by Predicted Label\n(Per-Class Precision)")
    ax2.set_xlabel("Predicted Label")
    ax2.set_ylabel("True Label")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    save_path = f"./evaluations/plots/{tile_name}_{model_name}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)

    return fig

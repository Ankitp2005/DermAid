import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, accuracy_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

import config

def full_evaluation(model, test_loader, device):
    """
    Computes all standard and critical evaluation metrics.
    """
    model.eval()
    
    y_true_cond = []
    y_score_cond = []
    y_pred_cond = []
    
    y_true_sev = []
    y_pred_sev = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch[0].to(device)
            cond_true = batch[1].to(device)
            sev_true = batch[2].to(device)
            
            cond_logits, sev_logits, _ = model(images)
            
            cond_probs = torch.softmax(cond_logits, dim=1)
            cond_preds = torch.argmax(cond_probs, dim=1)
            
            sev_preds = torch.argmax(torch.softmax(sev_logits, dim=1), dim=1)
            
            y_true_cond.extend(cond_true.cpu().numpy())
            y_score_cond.extend(cond_probs.cpu().numpy())
            y_pred_cond.extend(cond_preds.cpu().numpy())
            
            y_true_sev.extend(sev_true.cpu().numpy())
            y_pred_sev.extend(sev_preds.cpu().numpy())
            
    y_true_cond = np.array(y_true_cond)
    y_score_cond = np.array(y_score_cond)
    y_pred_cond = np.array(y_pred_cond)
    
    y_true_sev = np.array(y_true_sev)
    y_pred_sev = np.array(y_pred_sev)
    
    n_classes = len(config.CLASS_NAMES)
    
    # AUC metrics
    macro_auc = roc_auc_score(y_true_cond, y_score_cond, multi_class='ovr', average='macro')
    
    y_true_bin = label_binarize(y_true_cond, classes=range(n_classes))
    per_class_auc = {}
    for i, name in enumerate(config.CLASS_NAMES):
        try:
            per_class_auc[name] = float(roc_auc_score(y_true_bin[:, i], y_score_cond[:, i]))
        except ValueError:
            per_class_auc[name] = 0.5  # Edge case where class isn't in test batch
            
    # F1 metrics
    macro_f1 = float(f1_score(y_true_cond, y_pred_cond, average='macro', zero_division=0))
    weighted_f1 = float(f1_score(y_true_cond, y_pred_cond, average='weighted', zero_division=0))
    
    # Recall metrics (Critical safety endpoints)
    recalls = recall_score(y_true_cond, y_pred_cond, average=None, labels=range(n_classes), zero_division=0)
    mel_idx = config.CLASS_NAMES.index('mel')
    bcc_idx = config.CLASS_NAMES.index('bcc')
    mel_recall = float(recalls[mel_idx])
    bcc_recall = float(recalls[bcc_idx])
    
    # Accuracies
    severity_accuracy = float(accuracy_score(y_true_sev, y_pred_sev))
    overall_accuracy = float(accuracy_score(y_true_cond, y_pred_cond))
    
    # Text report
    report_str = classification_report(y_true_cond, y_pred_cond, target_names=config.CLASS_NAMES, zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true_cond, y_pred_cond, labels=range(n_classes))
    
    return {
        'macro_auc': float(macro_auc),
        'per_class_auc': per_class_auc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'mel_recall': mel_recall,
        'bcc_recall': bcc_recall,
        'severity_accuracy': severity_accuracy,
        'overall_accuracy': overall_accuracy,
        'classification_report_str': report_str,
        'confusion_matrix_array': cm.tolist(),
        'y_true_cond': y_true_cond.tolist(),
        'y_score_cond': y_score_cond.tolist()
    }


def plot_confusion_matrix(cm, class_names, save_path='results/confusion_matrix.png'):
    cm_arr = np.array(cm)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm_arr.astype('float') / cm_arr.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title('DermAid — Condition Classification Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curves(true_labels, pred_probs, class_names, save_path='results/roc_curves.png'):
    true_labels = np.array(true_labels)
    pred_probs = np.array(pred_probs)
    n_classes = len(class_names)
    
    y_bin = label_binarize(true_labels, classes=range(n_classes))
    
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        
        # Highlight safety-critical classes
        if class_name == 'mel':
            plt.plot(fpr, tpr, color='red', lw=2.5, zorder=5,
                     label=f'ROC {class_name} (AUC = {roc_auc:.2f}) [SAFETY CRITICAL]')
        elif class_name == 'bcc':
            plt.plot(fpr, tpr, color='darkorange', lw=2.5, zorder=4,
                     label=f'ROC {class_name} (AUC = {roc_auc:.2f}) [SAFETY CRITICAL]')
        else:
            plt.plot(fpr, tpr, lw=1.5, alpha=0.6, 
                     label=f'ROC {class_name} (AUC = {roc_auc:.2f})')
                     
    macro_auc = roc_auc_score(true_labels, pred_probs, multi_class='ovr', average='macro')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.title(f'Multi-Class OVR ROC Curves\nMacro Average AUC: {macro_auc:.3f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def print_contest_scorecard(results_dict):
    targets = {
        'macro_auc': 0.91,
        'mel_recall': 0.90,
        'bcc_recall': 0.85,
        'severity_acc': 0.93
    }
    
    def status(val, tgt):
        return "PASS" if val >= tgt else "FAIL"
        
    print("\n" + "="*60)
    print("                DERMAID COMPETITION SCORECARD             ")
    print("="*60)
    print("Metric                 | Score   | Target  | Status")
    print("-" * 60)
    
    m_auc = results_dict['macro_auc']
    m_rec = results_dict['mel_recall']
    b_rec = results_dict['bcc_recall']
    s_acc = results_dict['severity_accuracy']
    
    print(f"Macro OVR AUC          | {m_auc:.4f}  | {targets['macro_auc']:.2f}    | {status(m_auc, targets['macro_auc'])}")
    print(f"Melanoma Recall        | {m_rec:.4f}  | {targets['mel_recall']:.2f}    | {status(m_rec, targets['mel_recall'])}")
    print(f"BCC Recall             | {b_rec:.4f}  | {targets['bcc_recall']:.2f}    | {status(b_rec, targets['bcc_recall'])}")
    print(f"Severity Accuracy      | {s_acc:.4f}  | {targets['severity_acc']:.2f}    | {status(s_acc, targets['severity_acc'])}")
    print("="*60 + "\n")


def save_results(results_dict, path='results/evaluation_results.json'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Omit the raw arrays from the JSON storage
    clean_dict = {
        k: v for k, v in results_dict.items() 
        if k not in ['y_true_cond', 'y_score_cond']
    }
    
    with open(path, 'w') as f:
        json.dump(clean_dict, f, indent=4)

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def predict_with_uncertainty(model, image_tensor, n_passes=20, device='cuda'):
    """
    Performs Monte Carlo (MC) Dropout inference to capture model epistemic uncertainty.
    """
    # model.train() ensures dropout layers remain active during inference
    model.train() 
    image_tensor = image_tensor.to(device)
    
    all_probs = []
    
    with torch.no_grad():
        for _ in range(n_passes):
            cond_logits, _, _ = model(image_tensor)
            # Batch size is assumed to be 1 for single image prediction
            probs = torch.softmax(cond_logits, dim=1).cpu().numpy()[0]
            all_probs.append(probs)
            
    all_probs = np.array(all_probs)
    
    # Axis 0 corresponds to the multiple stochastic forward passes
    mean_probs = np.mean(all_probs, axis=0)
    uncertainty = np.std(all_probs, axis=0)
    
    max_uncertainty = uncertainty.max()
    predicted_class = int(np.argmax(mean_probs))
    
    auto_escalated = max_uncertainty > 0.15
    escalation_reason = None
    if auto_escalated:
        escalation_reason = f"High variance in repeated model predictions (Max std: {max_uncertainty:.3f})"
    
    return {
        'mean_probs': [float(x) for x in mean_probs],
        'predicted_class': predicted_class,
        'max_uncertainty': float(max_uncertainty),
        'per_class_uncertainty': [float(x) for x in uncertainty],
        'auto_escalated': auto_escalated,
        'escalation_reason': escalation_reason
    }


def uncertainty_to_severity_override(original_severity, max_uncertainty):
    """
    Translates model uncertainty into clinical safety overrides.
    """
    # Extremely uncertain cases are dangerous and require specialist review
    if max_uncertainty > 0.25:
        return 'Refer Immediately'
        
    # Moderately uncertain cases originally deemed low risk need short-term review
    if max_uncertainty > 0.15 and original_severity == 'Low Risk':
        return 'Refer Soon'
        
    return original_severity


def calibration_plot(model, val_loader, device, save_path='results/calibration.png'):
    """
    Computes Expected Calibration Error (ECE) and plots a reliability diagram.
    """
    model.eval()
    confidences = []
    accuracies = []
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch[0].to(device)
            cond_labels = batch[1].to(device)
            
            cond_logits, _, _ = model(images)
            probs = torch.softmax(cond_logits, dim=1)
            
            # Use top probability prediction for standard ECE estimation
            max_probs, preds = torch.max(probs, dim=1)
            
            confidences.extend(max_probs.cpu().numpy())
            accuracies.extend((preds == cond_labels).cpu().numpy().astype(float))
            
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    
    # Binning the predictions into 10 equally spaced intervals [0.0, 1.0]
    n_bins = 10
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    
    bin_accs = np.zeros(n_bins)
    bin_confs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        # Determine elements falling into the current bin
        in_bin = (confidences > bins[i]) & (confidences <= bins[i+1])
        if i == 0:
            in_bin = (confidences >= bins[0]) & (confidences <= bins[1])
            
        count = in_bin.sum()
        if count > 0:
            bin_accs[i] = accuracies[in_bin].mean()
            bin_confs[i] = confidences[in_bin].mean()
            bin_counts[i] = count
            
    # Calculate Expected Calibration Error
    ece = 0.0
    total_samples = len(confidences)
    for i in range(n_bins):
        if bin_counts[i] > 0:
            weight = bin_counts[i] / total_samples
            ece += weight * np.abs(bin_accs[i] - bin_confs[i])
            
    # Plotting routine
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', zorder=1, label='Perfectly Calibrated')
    
    valid_bins = bin_counts > 0
    plt.plot(
        bin_confs[valid_bins], 
        bin_accs[valid_bins], 
        'bo-', 
        linewidth=2,
        markersize=8,
        zorder=2,
        label=f'DermAid Calibration\\nECE = {ece:.4f}'
    )
    
    plt.ylabel('Fraction of Positives (Accuracy)')
    plt.xlabel('Mean Predicted Probability (Confidence)')
    plt.title('Reliability Diagram (Calibration)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Calibration plot generated at {save_path}. Overall ECE: {ece:.4f}")

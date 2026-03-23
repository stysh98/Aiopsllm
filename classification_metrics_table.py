#!/usr/bin/env python3

def print_detailed_metrics():
    tp = 25
    fp = 25
    tn = 0
    fn = 0
    
    total = tp + fp + tn + fn
    
    print("="*80)
    print("HDFS ANOMALY DETECTION - DETAILED CLASSIFICATION METRICS")
    print("="*80)
    
    print(f"\nRAW COUNTS:")
    print(f"True Positives (TP):   {tp:3d} - Anomalies correctly identified as anomalies")
    print(f"False Positives (FP):  {fp:3d} - Normal sequences incorrectly identified as anomalies")
    print(f"True Negatives (TN):   {tn:3d} - Normal sequences correctly identified as normal")
    print(f"False Negatives (FN):  {fn:3d} - Anomalies incorrectly identified as normal")
    print(f"Total Samples:         {total:3d}")
    
    print(f"\nCONFUSION MATRIX:")
    print(f"                      PREDICTED")
    print(f"                   Normal  Anomaly  Total")
    print(f"ACTUAL   Normal       {tn:3d}     {fp:3d}    {tn+fp:3d}")
    print(f"         Anomaly      {fn:3d}     {tp:3d}    {fn+tp:3d}")
    print(f"         Total        {tn+fn:3d}     {fp+tp:3d}    {total:3d}")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0
    
    print(f"\nPERFORMANCE METRICS:")
    print(f"Precision:             {precision:.3f} ({precision*100:.1f}%)")
    print(f"Recall (Sensitivity):  {recall:.3f} ({recall*100:.1f}%)")
    print(f"Specificity:           {specificity:.3f} ({specificity*100:.1f}%)")
    print(f"F1-Score:              {f1_score:.3f} ({f1_score*100:.1f}%)")
    print(f"Accuracy:              {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    print(f"\nPROBLEM ANALYSIS:")
    print(f"• The model predicts ALL {total} sequences as anomalies")
    print(f"• Only {tp} out of {total} are actually anomalies")
    print(f"• {fp} normal sequences are misclassified (100% false positive rate)")
    print(f"• {tn} normal sequences are correctly identified (0% true negative rate)")
    print(f"• Precision of {precision:.1%} means half of all predictions are wrong")
    print(f"• The model has learned to always predict 'anomaly'")
    
    print(f"\nWHY THIS IS A CRITICAL PROBLEM:")
    print(f"• In production, this would generate {fp} false alarms for every {tp} real issues")
    print(f"• Operations teams would be overwhelmed with false alerts")
    print(f"• Real anomalies would be lost in the noise")
    print(f"• The system is essentially unusable for anomaly detection")
    
    print("="*80)

if __name__ == "__main__":
    print_detailed_metrics()
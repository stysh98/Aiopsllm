#!/usr/bin/env python3

import json
from typing import Dict, List, Tuple

def load_results(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_predictions_and_truth(data: Dict) -> Tuple[List[bool], List[bool]]:
    predictions = []
    ground_truth = []
    
    for pred in data['anomaly_detection']['predictions']:
        predictions.append(pred['predicted_anomaly'])
    
    for truth in data['anomaly_detection']['ground_truth']:
        ground_truth.append(truth['actual_anomaly'])
    
    return predictions, ground_truth

def calculate_metrics(predictions: List[bool], ground_truth: List[bool]) -> Dict:
    tp = fp = tn = fn = 0
    
    for pred, actual in zip(predictions, ground_truth):
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and actual:
            fn += 1
        else:
            tn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    
    return {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy
    }

def analyze_misclassifications(data: Dict, predictions: List[bool], ground_truth: List[bool]) -> Dict:
    misclassifications = {
        'false_positives': [],
        'false_negatives': []
    }
    
    predictions_data = data['anomaly_detection']['predictions']
    ground_truth_data = data['anomaly_detection']['ground_truth']
    
    for i, (pred, actual) in enumerate(zip(predictions, ground_truth)):
        if pred != actual:
            block_id = predictions_data[i]['block_id']
            confidence = predictions_data[i]['confidence']
            analysis = predictions_data[i]['analysis']
            
            if pred and not actual:
                misclassifications['false_positives'].append({
                    'block_id': block_id,
                    'confidence': confidence,
                    'analysis_snippet': analysis[:500] + "..." if len(analysis) > 500 else analysis
                })
            elif not pred and actual:
                misclassifications['false_negatives'].append({
                    'block_id': block_id,
                    'confidence': confidence,
                    'analysis_snippet': analysis[:500] + "..." if len(analysis) > 500 else analysis
                })
    
    return misclassifications

def print_metrics_table(metrics: Dict):
    print("\n" + "="*60)
    print("HDFS ANOMALY DETECTION - CLASSIFICATION METRICS")
    print("="*60)
    
    print(f"\nCONFUSION MATRIX:")
    print(f"                    Predicted")
    print(f"                 Normal  Anomaly")
    print(f"Actual  Normal     {metrics['tn']:3d}     {metrics['fp']:3d}")
    print(f"        Anomaly    {metrics['fn']:3d}     {metrics['tp']:3d}")
    
    print(f"\nCLASSIFICATION METRICS:")
    print(f"True Positives (TP):   {metrics['tp']:3d}")
    print(f"False Positives (FP):  {metrics['fp']:3d}")
    print(f"True Negatives (TN):   {metrics['tn']:3d}")
    print(f"False Negatives (FN):  {metrics['fn']:3d}")
    print(f"")
    print(f"Precision:             {metrics['precision']:.3f}")
    print(f"Recall:                {metrics['recall']:.3f}")
    print(f"F1-Score:              {metrics['f1_score']:.3f}")
    print(f"Accuracy:              {metrics['accuracy']:.3f}")

def analyze_false_positive_patterns(false_positives: List[Dict]):
    print(f"\n" + "="*60)
    print("FALSE POSITIVE ANALYSIS - Why Normal HDFS Behavior is Misclassified")
    print("="*60)
    
    print(f"\nTotal False Positives: {len(false_positives)}")
    
    common_keywords = {}
    replication_mentions = 0
    network_mentions = 0
    multiple_sources_mentions = 0
    
    for fp in false_positives:
        analysis = fp['analysis_snippet'].lower()
        
        if 'replication' in analysis:
            replication_mentions += 1
        
        if 'network' in analysis:
            network_mentions += 1
            
        if 'multiple sources' in analysis or 'different sources' in analysis:
            multiple_sources_mentions += 1
    
    print(f"\nCOMMON MISCLASSIFICATION PATTERNS:")
    print(f"- Replication mentioned: {replication_mentions}/{len(false_positives)} ({replication_mentions/len(false_positives)*100:.1f}%)")
    print(f"- Network issues mentioned: {network_mentions}/{len(false_positives)} ({network_mentions/len(false_positives)*100:.1f}%)")
    print(f"- Multiple sources mentioned: {multiple_sources_mentions}/{len(false_positives)} ({multiple_sources_mentions/len(false_positives)*100:.1f}%)")
    
    print(f"\nSAMPLE FALSE POSITIVES (Normal blocks classified as anomalies):")
    for i, fp in enumerate(false_positives[:5]):
        print(f"\n{i+1}. Block: {fp['block_id']}")
        print(f"   Confidence: {fp['confidence']}")
        print(f"   Analysis: {fp['analysis_snippet'][:200]}...")

def main():
    data = load_results('results/hdfs_rcaeval_integration_20260310_155309.json')
    
    predictions, ground_truth = extract_predictions_and_truth(data)
    
    print(f"Loaded {len(predictions)} predictions and {len(ground_truth)} ground truth labels")
    
    metrics = calculate_metrics(predictions, ground_truth)
    
    print_metrics_table(metrics)
    
    misclassifications = analyze_misclassifications(data, predictions, ground_truth)
    
    analyze_false_positive_patterns(misclassifications['false_positives'])
    
    print(f"\n" + "="*60)
    print("PROBLEM SUMMARY")
    print("="*60)
    print(f"The model is predicting {len([p for p in predictions if p])}/{len(predictions)} sequences as anomalies")
    print(f"But only {len([t for t in ground_truth if t])}/{len(ground_truth)} are actually anomalies")
    print(f"This means {metrics['fp']} normal sequences are being misclassified as anomalies")
    print(f"The precision is only {metrics['precision']:.1%}, meaning half of all anomaly predictions are wrong")

if __name__ == "__main__":
    main()
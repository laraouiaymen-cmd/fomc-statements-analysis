"""
Model Selection Module

Implements lexicographic model selection:
1. Filter by robustness (mandatory)
2. Compare performance among robust candidates
3. Select ONE model for inference

This ensures we select models that are stable and reliable, not just
those with the highest raw performance metrics.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path


def filter_robust_models(
    sensitivity_data: List[Dict],
    cv_threshold: float = 5.0,
    max_range_threshold: float = 0.10
) -> List[Dict]:
    """
    Step 1: Filter by robustness.
    
    A model is robust if EITHER:
    1. Low CV (< cv_threshold%) - relative stability
    2. Low range (< max_range_threshold) - absolute stability
    
    Args:
        sensitivity_data: List of model sensitivity metrics
        cv_threshold: Maximum acceptable CV% (default: 5%)
        max_range_threshold: Maximum acceptable range
    
    Returns:
        List of robust model configurations
    """
    robust_models = []
    
    for item in sensitivity_data:
        # A model is robust if EITHER condition is met:
        # 1. CV is below threshold (relative stability)
        # OR
        # 2. Absolute range is acceptable (absolute stability)
        is_robust = (
            item['cv_variation'] < cv_threshold or
            item['range'] < max_range_threshold
        )
        
        if is_robust:
            robust_models.append(item)
    
    return robust_models


def select_best_robust_model(
    robust_models: List[Dict],
    all_results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    metric: str = 'roc_auc',
    higher_is_better: bool = True
) -> Optional[Dict]:
    """
    Step 2 & 3: Among robust candidates, select best performing model.
    
    Args:
        robust_models: Robust configurations from filter_robust_models()
        all_results: {model: {config: {embedding: metrics}}}
        metric: Performance metric for selection
        higher_is_better: Whether higher values are better
    
    Returns:
        Selected model dict with 'model', 'embedding', 'config', 'performance'
    """
    if not robust_models:
        return None
    
    # For each robust model, get its mean performance on the selection metric
    candidates = []
    
    for item in robust_models:
        model_name = item['model']
        embedding_name = item['embedding']
        
        # Find the performance across all configs for this (model, embedding) pair
        performances = []
        for config_name, emb_results in all_results[model_name].items():
            if embedding_name in emb_results:
                perf = emb_results[embedding_name].get(metric)
                if perf is not None:
                    performances.append(perf)
        
        if performances:
            mean_performance = np.mean(performances)
            candidates.append({
                **item,
                'performance': mean_performance
            })
    
    if not candidates:
        return None
    
    # Select the best performing robust model
    if higher_is_better:
        best_model = max(candidates, key=lambda x: x['performance'])
    else:
        best_model = min(candidates, key=lambda x: x['performance'])
    
    return best_model


def print_selection_summary(
    all_sensitivity_data: List[Dict],
    robust_models: List[Dict],
    selected_model: Optional[Dict],
    task: str = 'classification',
    metric: str = 'roc_auc'
) -> None:
    """
    Display the model selection process and final choice.
    
    Args:
        all_sensitivity_data: All models before filtering
        robust_models: Models that passed robustness filter
        selected_model: The final selected model
        task: 'classification' or 'regression'
        metric: The metric used for selection
    """
    print("\n" + "="*70)
    print(f"📊 {task.upper()} MODEL SELECTION")
    print("="*70)
    
    # Step 1: Robustness filtering
    print("\n🔹 Step 1: Filter by Robustness")
    print("-" * 70)
    
    filtered_out = [x for x in all_sensitivity_data if x not in robust_models]
    
    print(f"  Total candidates: {len(all_sensitivity_data)}")
    print(f"  ✅ Robust: {len(robust_models)}")
    print(f"  ⚠️  Filtered out: {len(filtered_out)} (fragile/sensitive)")
    
    if filtered_out and len(filtered_out) <= 10:
        print("\n  Filtered out models:")
        for item in filtered_out:
            print(f"    • {item['model']:20s} + {item['embedding']:10s}  "
                  f"(CV: {item['cv_variation']:.1f}%, Range: {item['range']:.4f})")
    
    # Step 2: Performance comparison
    print("\n🔹 Step 2: Compare Performance Among Robust Candidates")
    print("-" * 70)
    
    if not robust_models:
        print("  ❌ No robust models found!")
        return
    
    # Show top 5 robust models by mean performance
    robust_with_perf = sorted(robust_models, key=lambda x: x.get('mean', 0), reverse=True)[:5]
    print(f"\n  Top robust candidates (by mean {metric}):")
    for i, item in enumerate(robust_with_perf, 1):
        marker = "🏆" if selected_model and item == selected_model else "  "
        print(f"    {marker} {i}. {item['model']:20s} + {item['embedding']:10s}  "
              f"Mean: {item['mean']:.4f}, CV: {item['cv_variation']:.1f}%")
    
    # Step 3: Final selection
    print("\n🔹 Step 3: Final Selection for Inference")
    print("-" * 70)
    
    if selected_model:
        print(f"\n  🎯 SELECTED MODEL: {selected_model['model']} + {selected_model['embedding']}")
        # Always use 'performance' field which contains the ROC-AUC mean
        # If metric is generic 'roc_auc', label it 'Mean ROC-AUC'
        label = "Mean ROC-AUC" if metric == "roc_auc" else f"Mean {metric}"
        print(f"     {label}: {selected_model['performance']:.4f}")
        print(f"     Std Dev: {selected_model['std']:.4f}")
        print(f"     Coefficient of Variation: {selected_model['cv_variation']:.1f}%")
        print(f"     Robustness: {selected_model.get('sensitivity', 'Robust')}")
        print(f"\n  ✅ This is the reference specification for {task} directional inference.")
    else:
        print("  ❌ No model could be selected!")


def select_grand_champion(
    clf_selected: Optional[Dict],
    reg_selected: Optional[Dict],
    classification_results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    regression_results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    y_test_class: Optional[np.ndarray] = None
) -> None:
    """
    Final Step: Identify the single "Grand Champion" across both tasks.
    
    Args:
        clf_selected: Best robust classification model
        reg_selected: Best robust regression model
        classification_results: Full classification results
        regression_results: Full regression results
        y_test_class: True test labels to calculate baseline accuracy
    """
    print("\n" + "🏆" * 35)
    print("🏆 FINAL PROJECT \"GRAND CHAMPION\" SPECIFICATION")
    print("🏆" * 35)
    
    if not clf_selected and not reg_selected:
        print("\n  ❌ No robust models found to select a champion!")
        return

    # Determine winner
    clf_auc = clf_selected['performance'] if clf_selected else -1
    reg_auc = reg_selected['performance'] if reg_selected else -1
    
    if clf_auc >= reg_auc:
        winner = clf_selected
        winner_type = "Classification"
        results = classification_results
        context_metric_name = "Accuracy"
        context_metric_key = 'accuracy'
    else:
        winner = reg_selected
        winner_type = "Regression"
        results = regression_results
        context_metric_name = "Directional Accuracy"
        context_metric_key = 'directional_accuracy'
        
    print(f"\n  The most robust directional signal was found using:")
    print(f"  🎯 {winner_type.upper()} + {winner['model']} + {winner['embedding']}")
    print("-" * 70)
    print(f"  Primary Metric (Selection):")
    print(f"  • Mean ROC-AUC: {winner['performance']:.4f}")
    
    # Calculate Mean Contextual Metric
    context_values = []
    for config_name, emb_results in results[winner['model']].items():
        if winner['embedding'] in emb_results:
            val = emb_results[winner['embedding']].get(context_metric_key)
            if val is not None:
                context_values.append(val)
    
    if context_values:
        mean_context = np.mean(context_values) * 100  # Convert to percentage
        print(f"\n  Contextual Metric (Interpretability):")
        print(f"  • Mean {context_metric_name}: {mean_context:.1f}%")
    
    # Baseline comparison (optional but recommended)
    if y_test_class is not None and len(y_test_class) > 0:
        up_ratio = np.sum(y_test_class) / len(y_test_class)
        baseline = max(up_ratio, 1 - up_ratio) * 100
        majority_class = "Up" if up_ratio > 0.5 else "Down"
        print(f"  • Naive Baseline (Always Predict '{majority_class}'): {baseline:.1f}%")

    print(f"\n  Robustness Profile:")
    print(f"  • CV Variation: {winner['cv_variation']:.1f}%")
    print(f"  • Stability: {winner.get('sensitivity', 'Robust')}")
    
    print("-" * 70)
    print(f"  ✅ This is the final reference specification for the study.")
    print("=" * 70 + "\n")


def select_models_for_inference(
    classification_results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    regression_results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    classification_sensitivity: List[Dict],
    regression_sensitivity: List[Dict],
    y_test_class: Optional[np.ndarray] = None
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Main function to select one model for classification and one for regression.
    
    Implements the complete 3-step lexicographic selection process for both tasks,
    followed by Grand Champion selection.
    
    Args:
        classification_results: Classification experiment results
        regression_results: Regression experiment results
        classification_sensitivity: Sensitivity analysis for classification
        regression_sensitivity: Sensitivity analysis for regression
        y_test_class: True test labels for baseline accuracy calculation
    
    Returns:
        Tuple of (selected_classification_model, selected_regression_model)
    """
    print("\n" + "="*70)
    print("🎯 MODEL SELECTION FOR DIRECTIONAL INFERENCE")
    print("="*70)
    print("\nUsing lexicographic selection:")
    print("  1. Filter by robustness (mandatory)")
    print("  2. Compare performance among robust candidates")
    print("  3. Fix ONE model for inference")
    
    # CLASSIFICATION
    clf_robust = filter_robust_models(
        classification_sensitivity,
        cv_threshold=5.0,
        max_range_threshold=0.10
    )
    
    clf_selected = select_best_robust_model(
        clf_robust,
        classification_results,
        metric='roc_auc',
        higher_is_better=True
    )
    
    print_selection_summary(
        classification_sensitivity,
        clf_robust,
        clf_selected,
        task='classification',
        metric='ROC-AUC'
    )
    
    # REGRESSION
    reg_robust = filter_robust_models(
        regression_sensitivity,
        cv_threshold=5.0,
        max_range_threshold=0.10
    )
    
    reg_selected = select_best_robust_model(
        reg_robust,
        regression_results,
        metric='roc_auc',
        higher_is_better=True
    )
    
    print_selection_summary(
        regression_sensitivity,
        reg_robust,
        reg_selected,
        task='regression',
        metric='ROC-AUC'
    )
    
    # GRAND CHAMPION
    select_grand_champion(
        clf_selected,
        reg_selected,
        classification_results,
        regression_results,
        y_test_class
    )
    
    return clf_selected, reg_selected

"""
Evaluation utilities for FOMC statement analysis.

This module provides functions to evaluate model performance on both
classification (binary up/down) and regression (continuous % return) tasks.
All functions work with the model wrappers defined in models.py.
"""

from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import numpy as np

# Type aliases for clarity
Model = Any  # sklearn-compatible model with fit/predict
EmbeddingDict = Dict[str, Tuple[np.ndarray, np.ndarray]]
MetricsDict = Dict[str, float]
ResultsDict = Dict[str, Dict[str, Dict[str, MetricsDict]]]

from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

def save_table_figure(
    df: pd.DataFrame,
    title: str,
    filename: str,
    save_dir: Path
) -> None:
    """
    Render a DataFrame as a table figure and save to PNG.
    """
    plt.figure(figsize=(12, len(df) * 0.5 + 2))
    ax = plt.gca()
    
    # Hide axes
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    
    # Create table
    table = plt.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#f2f2f2'] * len(df.columns)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title(title, pad=20)
    
    save_path = save_dir / filename
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"   Saved figure: {save_path}")



# ============================================================================
# CLASSIFICATION EVALUATION - For binary up/down predictions
# ============================================================================

def evaluate_classifier(
    model: Model,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> MetricsDict:
    """
    Evaluate a classification model on test data.
    
    Returns ROC-AUC (for selection) and Accuracy (for interpretability).
    Model must already be trained.
    
    Args:
        model: Trained classifier (from models.py)
        X_test: Test features (embeddings)
        y_test: True labels (0=down, 1=up)
        
    Returns:
        Dictionary containing evaluation metrics:
            - accuracy: Accuracy score (0-1), for interpretability
            - roc_auc: ROC-AUC score (0-1), for selection
    """
    # Get hard predictions for accuracy
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred)
    }
    
    # Add ROC-AUC if model supports predict_proba
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        except:
            metrics['roc_auc'] = 0.5 # Fallback
    else:
        # Fallback for models without predict_proba
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred)
    
    return metrics


# ============================================================================
# REGRESSION EVALUATION - For continuous % return predictions
# ============================================================================

def evaluate_regressor(
    model: Model,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> MetricsDict:
    """
    Evaluate a regression model on test data.
    
    Computes ROC-AUC for directional prediction by treating sign(y_test) as binary labels.
    
    Args:
        model: Trained regressor
        X_test: Test features
        y_test: True returns
        
    Returns:
        Dict with 'roc_auc' and 'directional_accuracy'
    """
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Convert to binary direction for ROC-AUC
    # True labels: 0 if return < 0 (down), 1 if return >= 0 (up)
    y_true_direction = (y_test >= 0).astype(int)
    
    # Predicted scores: use predicted return as confidence
    # Higher predicted return = higher confidence in "up"
    y_pred_scores = y_pred
    
    # Calculate ROC-AUC for directional prediction
    try:
        roc_auc = roc_auc_score(y_true_direction, y_pred_scores)
    except ValueError:
        # If only one class present in y_true, ROC-AUC is undefined
        roc_auc = 0.5
    
    # Calculate directional accuracy (for interpretability)
    pred_direction = (y_pred >= 0).astype(int)
    directional_acc = accuracy_score(y_true_direction, pred_direction)
    
    return {
        'roc_auc': roc_auc,
        'directional_accuracy': directional_acc
    }





# ============================================================================
# COMPARISON UTILITIES
# ============================================================================


def compare_embeddings(
    embeddings: EmbeddingDict,
    model: Model,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task: str = 'classification'
) -> Dict[str, MetricsDict]:
    """
    Compare different embeddings using the same model configuration.
    
    TRAINS model on X_train and EVALUATES on X_test.
    
    Args:
        embeddings: {embedding_name: (X_train, X_test)} tuples
        model: Untrained model instance with fixed hyperparameters
        y_train: Training labels or returns
        y_test: Test labels or returns
        task: Either 'classification' or 'regression'
        
    Returns:
        {embedding_name: metrics_dict}
    """
    results = {}
    
    for emb_name, (X_train, X_test) in embeddings.items():
        # Create fresh model instance using stored init params
        fresh_model = model.__class__(**model._init_params)
        
        if task == 'classification':
            # Train final model on full training set
            fresh_model.fit(X_train, y_train)
            
            # Evaluate on test set
            metrics = evaluate_classifier(fresh_model, X_test, y_test)
            
            results[emb_name] = metrics
            
        elif task == 'regression':
            # Train on this embedding
            fresh_model.fit(X_train, y_train)
            
            # Evaluate
            results[emb_name] = evaluate_regressor(fresh_model, X_test, y_test)
        else:
            raise ValueError(f"Unknown task: {task}")
    
    return results







def save_results_to_csv(
    classification_results: ResultsDict,
    regression_results: ResultsDict,
    filename: str = "experiment_results.csv",
    save_dir: Path = Path("results")
) -> None:
    """
    Flatten results and save to CSV.
    """
    rows = []
    
    # Process Classification Results
    for model, configs in classification_results.items():
        for config, embeddings in configs.items():
            for emb, metrics in embeddings.items():
                row = {
                    'Task': 'Classification',
                    'Model': model,
                    'Configuration': config,
                    'Embedding': emb,
                    'ROC_AUC': metrics.get('roc_auc'),
                    'Accuracy': metrics.get('accuracy'),
                    'Directional_Accuracy': None
                }
                rows.append(row)
                
    # Process Regression Results
    for model, configs in regression_results.items():
        for config, embeddings in configs.items():
            for emb, metrics in embeddings.items():
                row = {
                    'Task': 'Regression',
                    'Model': model,
                    'Configuration': config,
                    'Embedding': emb,
                    'ROC_AUC': metrics.get('roc_auc'),
                    'Accuracy': None,
                    'Directional_Accuracy': metrics.get('directional_accuracy')
                }
                rows.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    save_path = save_dir / filename
    df.to_csv(save_path, index=False)
    print(f"\n✅ Detailed results saved to: {save_path}")


def analyze_sensitivity(
    classification_results: ResultsDict,
    regression_results: ResultsDict,
    save_dir: Path = Path("results")
) -> Tuple[List[Dict], List[Dict]]:
    """
    Analyze sensitivity of models to hyperparameter changes.
    
    Uses ROC-AUC for both classification and regression (converted to direction).
    Classifies models as:
    - Robust: <5% variation
    - Moderately Sensitive: 5-15% variation
    - Highly Sensitive: >15% variation
    
    Returns:
        Tuple of (classification_sensitivity_data, regression_sensitivity_data)
    """
    print("\n" + "="*70)
    print("📊 SENSITIVITY ANALYSIS")
    print("="*70)
    
    all_sensitivity_data = []
    
    # Helper to calculate sensitivity
    def calculate_sensitivity(
        results: ResultsDict, 
        metric: str, 
        lower_is_better: bool = False, 
        min_abs_variation: float = 0.0
    ) -> List[Dict[str, Any]]:
        sensitivity_info = []
        
        for model, configs in results.items():
            for emb in list(configs.values())[0].keys():  # Get embedding names
                values = []
                for config, emb_results in configs.items():
                    val = emb_results[emb].get(metric)
                    if val is not None:
                        values.append(val)
                
                if len(values) < 2:
                    continue
                
                mean_val = np.mean(values)
                min_val = np.min(values)
                max_val = np.max(values)
                range_val = max_val - min_val
                std_val = np.std(values)
                
                # Calculate Coefficient of Variation (CV) in percent
                # Avoid division by zero
                if abs(mean_val) > 1e-9:
                    cv_variation = (std_val / abs(mean_val)) * 100
                else:
                    cv_variation = 0
                
                # Classify sensitivity
                # Robust if:
                # 1. Coefficient of Variation is less than 5%
                # OR
                # 2. Absolute range is effectively zero (below min threshold)
                if cv_variation < 5 or range_val < min_abs_variation:
                    sensitivity = "Robust"
                elif cv_variation < 15:
                    sensitivity = "Moderately Sensitive"
                else:
                    sensitivity = "Highly Sensitive"
                
                sensitivity_info.append({
                    'model': model,
                    'embedding': emb,
                    'mean': mean_val,
                    'std': std_val,
                    'range': range_val,
                    'cv_variation': cv_variation,
                    'sensitivity': sensitivity
                })
        
        return sensitivity_info
    
    # CLASSIFICATION SENSITIVITY
    # Use ROC-AUC (same as regression) for unified comparison
    clf_sensitivity = calculate_sensitivity(classification_results, 'roc_auc', min_abs_variation=0.10)
    
    # Add to detailed data
    for item in clf_sensitivity:
        all_sensitivity_data.append({
            'Task': 'Classification',
            'Metric': 'ROC-AUC',
            **item
        })
    
    # REGRESSION SENSITIVITY
    # Use ROC-AUC (same as classification) for unified comparison
    reg_sensitivity = calculate_sensitivity(regression_results, 'roc_auc', lower_is_better=False, min_abs_variation=0.10)
    
    # Add to detailed data
    for item in reg_sensitivity:
        all_sensitivity_data.append({
            'Task': 'Regression',
            'Metric': 'ROC-AUC',
            **item
        })
    
    print("\n" + "="*70)
    
    # Return sensitivity data for model selection
    return clf_sensitivity, reg_sensitivity


def save_diagnostic_composition(
    detailed_breakdown: List[Dict],
    save_dir: Path = Path("results")
) -> Path:
    """
    Save diagnostic composition details to CSV.
    
    For each diagnostic view (task × by_embedding/by_model), tracks which
    specific items (models or embeddings) contribute to each category count.
    
    Args:
        detailed_breakdown: List of dicts from create_diagnostic_summary
        save_dir: Directory to save composition file
        
    Returns:
        Path to the saved composition CSV file
    """
    composition_rows = []
    
    # Separate data by task
    clf_data = [row for row in detailed_breakdown if row['Task'] == 'Classification']
    reg_data = [row for row in detailed_breakdown if row['Task'] == 'Regression']
    
    # Define fixed order for deterministic output
    embedding_order = ['TF-IDF', 'BERT', 'FinBERT', 'GTE']
    model_order = ['Logistic Regression', 'Random Forest', 'XGBoost', 'ElasticNet']
    
    # Helper to build composition for a group
    def build_composition(data, group_by, item_col):
        """Build composition lists for a diagnostic view."""
        groups = {}
        
        for row in data:
            group_name = row[group_by]
            item_name = row[item_col]
            category = row['Category']
            
            if group_name not in groups:
                groups[group_name] = {
                    'denominator': set(),
                    'Uninformative': [],
                    'Dangerous': [],
                    'Reliable': []
                }
            
            groups[group_name]['denominator'].add(item_name)
            groups[group_name][category].append(item_name)
        
        # Sort lists deterministically
        for group_name in groups:
            groups[group_name]['denominator'] = sorted(groups[group_name]['denominator'])
            for category in ['Uninformative', 'Dangerous', 'Reliable']:
                groups[group_name][category] = sorted(groups[group_name][category])
        
        return groups
    
    # Build all four diagnostic views
    
    # 1. Classification by Embedding (items = models)
    clf_by_emb = build_composition(clf_data, 'Embedding', 'Model')
    for emb in embedding_order:
        if emb in clf_by_emb:
            composition_rows.append({
                'task': 'Classification',
                'view': 'by_embedding',
                'group': emb,
                'denominator': ','.join(clf_by_emb[emb]['denominator']),
                'uninformative_items': ','.join(clf_by_emb[emb]['Uninformative']),
                'dangerous_items': ','.join(clf_by_emb[emb]['Dangerous']),
                'reliable_items': ','.join(clf_by_emb[emb]['Reliable'])
            })
    
    # 2. Classification by Model (items = embeddings)
    clf_by_model = build_composition(clf_data, 'Model', 'Embedding')
    for model in model_order:
        if model in clf_by_model:
            composition_rows.append({
                'task': 'Classification',
                'view': 'by_model',
                'group': model,
                'denominator': ','.join(clf_by_model[model]['denominator']),
                'uninformative_items': ','.join(clf_by_model[model]['Uninformative']),
                'dangerous_items': ','.join(clf_by_model[model]['Dangerous']),
                'reliable_items': ','.join(clf_by_model[model]['Reliable'])
            })
    
    # 3. Regression by Embedding (items = models)
    reg_by_emb = build_composition(reg_data, 'Embedding', 'Model')
    for emb in embedding_order:
        if emb in reg_by_emb:
            composition_rows.append({
                'task': 'Regression',
                'view': 'by_embedding',
                'group': emb,
                'denominator': ','.join(reg_by_emb[emb]['denominator']),
                'uninformative_items': ','.join(reg_by_emb[emb]['Uninformative']),
                'dangerous_items': ','.join(reg_by_emb[emb]['Dangerous']),
                'reliable_items': ','.join(reg_by_emb[emb]['Reliable'])
            })
    
    # 4. Regression by Model (items = embeddings)
    reg_by_model = build_composition(reg_data, 'Model', 'Embedding')
    for model in model_order:
        if model in reg_by_model:
            composition_rows.append({
                'task': 'Regression',
                'view': 'by_model',
                'group': model,
                'denominator': ','.join(reg_by_model[model]['denominator']),
                'uninformative_items': ','.join(reg_by_model[model]['Uninformative']),
                'dangerous_items': ','.join(reg_by_model[model]['Dangerous']),
                'reliable_items': ','.join(reg_by_model[model]['Reliable'])
            })
    
    # Save to CSV
    df_composition = pd.DataFrame(composition_rows)
    composition_path = save_dir / "diagnostic_composition.csv"
    df_composition.to_csv(composition_path, index=False)
    
    return composition_path


def create_diagnostic_summary(
    classification_sensitivity: List[Dict],
    regression_sensitivity: List[Dict],
    auc_threshold: float = 0.02,
    save_dir: Path = Path("results")
) -> None:
    """
    Create diagnostic summary tables categorizing embeddings and models.
    
    Categories:
    - Uninformative: |mean AUC - 0.5| < threshold (close to random)
    - Dangerous: |mean AUC - 0.5| >= threshold AND not robust (signal but fragile)
    - Reliable: |mean AUC - 0.5| >= threshold AND robust (signal + stable)
    
    Args:
        classification_sensitivity: Classification sensitivity data
        regression_sensitivity: Regression sensitivity data
        auc_threshold: Minimum AUC distance from random (0.5) to be informative
        save_dir: Directory to save diagnostic breakdown CSV
    """
    from collections import defaultdict
    import pandas as pd
    
    # Combine both tasks
    all_data = []
    for item in classification_sensitivity:
        all_data.append({
            'task': 'Classification',
            'embedding': item['embedding'],
            'model': item['model'],
            'mean_auc': item['mean'],
            'robust': item['sensitivity'] == 'Robust',
            'item': item
        })
    
    for item in regression_sensitivity:
        all_data.append({
            'task': 'Regression',
            'embedding': item['embedding'],
            'model': item['model'],
            'mean_auc': item['mean'],
            'robust': item['sensitivity'] == 'Robust',
            'item': item
        })
    
    # Categorize each pair
    detailed_breakdown = []
    for data in all_data:
        distance_from_random = abs(data['mean_auc'] - 0.5)
        
        if distance_from_random < auc_threshold:
            category = 'Uninformative'
        elif data['robust']:
            category = 'Reliable'
        else:
            category = 'Dangerous'
        
        detailed_breakdown.append({
            'Task': data['task'],
            'Embedding': data['embedding'],
            'Model': data['model'],
            'Mean_AUC': data['mean_auc'],
            'Distance_from_Random': distance_from_random,
            'CV_Variation': data['item']['cv_variation'],
            'Range': data['item']['range'],
            'Robust': data['robust'],
            'Category': category
        })
    
    # Save detailed breakdown
    df_detail = pd.DataFrame(detailed_breakdown)
    detail_path = save_dir / "diagnostic_breakdown.csv"
    df_detail.to_csv(detail_path, index=False)
    
    print(f"\n📂 Detailed sensitivity & diagnostic data saved to: {detail_path}")
    print("   (Contains mean ROC-AUC, CV Variation, and Reliability Category for every configuration)")
    
    # Separate data by task
    clf_data = [row for row in detailed_breakdown if row['Task'] == 'Classification']
    reg_data = [row for row in detailed_breakdown if row['Task'] == 'Regression']
    
    # Aggregate by embedding FOR EACH TASK
    clf_embedding_counts = defaultdict(lambda: {'Uninformative': 0, 'Dangerous': 0, 'Reliable': 0})
    for row in clf_data:
        clf_embedding_counts[row['Embedding']][row['Category']] += 1
    
    reg_embedding_counts = defaultdict(lambda: {'Uninformative': 0, 'Dangerous': 0, 'Reliable': 0})
    for row in reg_data:
        reg_embedding_counts[row['Embedding']][row['Category']] += 1
    
    # Aggregate by model FOR EACH TASK
    clf_model_counts = defaultdict(lambda: {'Uninformative': 0, 'Dangerous': 0, 'Reliable': 0})
    for row in clf_data:
        clf_model_counts[row['Model']][row['Category']] += 1
    
    reg_model_counts = defaultdict(lambda: {'Uninformative': 0, 'Dangerous': 0, 'Reliable': 0})
    for row in reg_data:
        reg_model_counts[row['Model']][row['Category']] += 1
    
    # Calculate mean AUC for ranking
    clf_embedding_means = {}
    for emb in clf_embedding_counts.keys():
        aucs = [r['Mean_AUC'] for r in clf_data if r['Embedding'] == emb]
        clf_embedding_means[emb] = np.mean(aucs)
    
    reg_embedding_means = {}
    for emb in reg_embedding_counts.keys():
        aucs = [r['Mean_AUC'] for r in reg_data if r['Embedding'] == emb]
        reg_embedding_means[emb] = np.mean(aucs)
    
    clf_model_means = {}
    for mod in clf_model_counts.keys():
        aucs = [r['Mean_AUC'] for r in clf_data if r['Model'] == mod]
        clf_model_means[mod] = np.mean(aucs)
    
    reg_model_means = {}
    for mod in reg_model_counts.keys():
        aucs = [r['Mean_AUC'] for r in reg_data if r['Model'] == mod]
        reg_model_means[mod] = np.mean(aucs)
    
    # Sort embeddings by # Reliable (desc), then mean AUC (desc)
    clf_sorted_embeddings = sorted(
        clf_embedding_counts.keys(),
        key=lambda x: (clf_embedding_counts[x]['Reliable'], clf_embedding_means[x]),
        reverse=True
    )
    
    reg_sorted_embeddings = sorted(
        reg_embedding_counts.keys(),
        key=lambda x: (reg_embedding_counts[x]['Reliable'], reg_embedding_means[x]),
        reverse=True
    )
    
    # Sort models by # Reliable (desc), then mean AUC (desc)
    clf_sorted_models = sorted(
        clf_model_counts.keys(),
        key=lambda x: (clf_model_counts[x]['Reliable'], clf_model_means[x]),
        reverse=True
    )
    
    reg_sorted_models = sorted(
        reg_model_counts.keys(),
        key=lambda x: (reg_model_counts[x]['Reliable'], reg_model_means[x]),
        reverse=True
    )
    
    # Generate interpretations
    def get_interpretation(counts):
        reliable = counts['Reliable']
        dangerous = counts['Dangerous']
        uninformative = counts['Uninformative']
        total = reliable + dangerous + uninformative
        
        if total == 0:
            return "N/A"
        
        if reliable >= total * 0.6:
            return "Consistently exploitable"
        elif reliable > 0 and dangerous == 0:
            return "Mostly reliable"
        elif dangerous > reliable and dangerous > uninformative:
            return "Unstable signal"
        elif uninformative >= total * 0.6:
            return "No usable signal"
        else:
            return "Mixed signal"
    
    # Print summary tables
    print("\n" + "="*70)
    print("📊 DIAGNOSTIC SUMMARY")
    print("="*70)
    
    print("\nCategory Definitions:")
    print(f"  • Uninformative : |mean ROC-AUC - 0.5| < {auc_threshold}  (Close to random)")
    print(f"  • Dangerous     : |mean ROC-AUC - 0.5| >= {auc_threshold} AND Not Robust (Signal but fragile)")
    print(f"  • Reliable      : |mean ROC-AUC - 0.5| >= {auc_threshold} AND Robust (Signal + Stable)")
    print(f"  • Max AUC       : Best ROC-AUC achieved among Reliable configurations (N/A if none)")
    
    # CLASSIFICATION TABLES
    print("\n🔹 CLASSIFICATION - EMBEDDING DIAGNOSTIC")
    
    # Prepare data for plotting (exclude Interpretation)
    clf_emb_rows = []
    
    print("-" * 90)
    print(f"{'Embedding':<12} | {'Uninformative':^15} | {'Dangerous':^11} | {'Reliable':^10} | {'Max AUC':<10} | {'Interpretation':<20}")
    print("-" * 90)
    for emb in clf_sorted_embeddings:
        counts = clf_embedding_counts[emb]
        interp = get_interpretation(counts)
        # Get max AUC for reliable configs
        reliable_aucs = [r['Mean_AUC'] for r in clf_data if r['Embedding'] == emb and r['Category'] == 'Reliable']
        max_auc = f"{max(reliable_aucs):.3f}" if reliable_aucs else "N/A"
        print(f"{emb:<12} | {counts['Uninformative']:^15} | {counts['Dangerous']:^11} | {counts['Reliable']:^10} | {max_auc:<10} | {interp:<20}")
        
        clf_emb_rows.append({
            'Embedding': emb,
            'Uninformative': counts['Uninformative'],
            'Dangerous': counts['Dangerous'],
            'Reliable': counts['Reliable'],
            'Max AUC': max_auc
        })
        
    # Save figure
    save_table_figure(
        pd.DataFrame(clf_emb_rows),
        "Classification - Embedding Diagnostic",
        "diagnostic_classification_embedding.png",
        save_dir
    )
    
    print("\n🔹 CLASSIFICATION - MODEL DIAGNOSTIC")
    
    # Prepare data for plotting
    clf_model_rows = []
    
    print("-" * 90)
    print(f"{'Model':<22} | {'Uninformative':^15} | {'Dangerous':^11} | {'Reliable':^10} | {'Max AUC':<10} | {'Interpretation':<15}")
    print("-" * 90)
    for mod in clf_sorted_models:
        counts = clf_model_counts[mod]
        interp = get_interpretation(counts)
        # Get max AUC for reliable configs
        reliable_aucs = [r['Mean_AUC'] for r in clf_data if r['Model'] == mod and r['Category'] == 'Reliable']
        max_auc = f"{max(reliable_aucs):.3f}" if reliable_aucs else "N/A"
        print(f"{mod:<22} | {counts['Uninformative']:^15} | {counts['Dangerous']:^11} | {counts['Reliable']:^10} | {max_auc:<10} | {interp:<15}")
        
        clf_model_rows.append({
            'Model': mod,
            'Uninformative': counts['Uninformative'],
            'Dangerous': counts['Dangerous'],
            'Reliable': counts['Reliable'],
            'Max AUC': max_auc
        })

    # Save figure
    save_table_figure(
        pd.DataFrame(clf_model_rows),
        "Classification - Model Diagnostic",
        "diagnostic_classification_model.png",
        save_dir
    )
    
    # REGRESSION TABLES
    print("\n🔹 REGRESSION - EMBEDDING DIAGNOSTIC")
    
    # Prepare data for plotting
    reg_emb_rows = []
    
    print("-" * 90)
    print(f"{'Embedding':<12} | {'Uninformative':^15} | {'Dangerous':^11} | {'Reliable':^10} | {'Max AUC':<10} | {'Interpretation':<20}")
    print("-" * 90)
    for emb in reg_sorted_embeddings:
        counts = reg_embedding_counts[emb]
        interp = get_interpretation(counts)
        # Get max AUC for reliable configs
        reliable_aucs = [r['Mean_AUC'] for r in reg_data if r['Embedding'] == emb and r['Category'] == 'Reliable']
        max_auc = f"{max(reliable_aucs):.3f}" if reliable_aucs else "N/A"
        print(f"{emb:<12} | {counts['Uninformative']:^15} | {counts['Dangerous']:^11} | {counts['Reliable']:^10} | {max_auc:<10} | {interp:<20}")
        
        reg_emb_rows.append({
            'Embedding': emb,
            'Uninformative': counts['Uninformative'],
            'Dangerous': counts['Dangerous'],
            'Reliable': counts['Reliable'],
            'Max AUC': max_auc
        })

    # Save figure
    save_table_figure(
        pd.DataFrame(reg_emb_rows),
        "Regression - Embedding Diagnostic",
        "diagnostic_regression_embedding.png",
        save_dir
    )
    
    print("\n🔹 REGRESSION - MODEL DIAGNOSTIC")
    
    # Prepare data for plotting
    reg_model_rows = []
    
    print("-" * 90)
    print(f"{'Model':<22} | {'Uninformative':^15} | {'Dangerous':^11} | {'Reliable':^10} | {'Max AUC':<10} | {'Interpretation':<15}")
    print("-" * 90)
    for mod in reg_sorted_models:
        counts = reg_model_counts[mod]
        interp = get_interpretation(counts)
        # Get max AUC for reliable configs
        reliable_aucs = [r['Mean_AUC'] for r in reg_data if r['Model'] == mod and r['Category'] == 'Reliable']
        max_auc = f"{max(reliable_aucs):.3f}" if reliable_aucs else "N/A"
        print(f"{mod:<22} | {counts['Uninformative']:^15} | {counts['Dangerous']:^11} | {counts['Reliable']:^10} | {max_auc:<10} | {interp:<15}")

        reg_model_rows.append({
            'Model': mod,
            'Uninformative': counts['Uninformative'],
            'Dangerous': counts['Dangerous'],
            'Reliable': counts['Reliable'],
            'Max AUC': max_auc
        })

    # Save figure
    save_table_figure(
        pd.DataFrame(reg_model_rows),
        "Regression - Model Diagnostic",
        "diagnostic_regression_model.png",
        save_dir
    )

    
    print("\n" + "="*70)
    print(f"💾 Detailed diagnostic breakdown saved to: {detail_path}")
    print("="*70)
    
    # Save composition details to separate file
    composition_path = save_diagnostic_composition(detailed_breakdown, save_dir)
    
    return composition_path


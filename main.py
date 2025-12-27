"""
Complete FOMC Embedding Comparison Experiment

This script compares TF-IDF, BERT, FinBERT, and GTE embeddings for predicting
S&P 500 movements from FOMC statements.

"""

import argparse
from pathlib import Path
import joblib
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data_loader import load_fomc_dataset, chronological_split, get_X_y
from src.embedder_tfidf import build_tfidf
from src.embedder_bert import load_bert, embed_many_bert
from src.embedder_finbert import load_finbert, embed_many_finbert
from src.embedder_gte import load_gte, embed_many_gte
from src.models import (
    LogisticRegression,
    RandomForestClassifier,
    XGBoostClassifier,
    ElasticNetRegressor,
    RandomForestRegressor,
    XGBoostRegressor
)
from src.evaluation import (
    compare_embeddings,
    save_results_to_csv,
    analyze_sensitivity,
    create_diagnostic_summary
)
from src.model_selection import select_models_for_inference
from src.embedding_cache import save_embeddings, load_embeddings
from src.data.update_fomc_statements import update_fomc_statements
from src.data.prepare_processed_data import process_statements


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Complete FOMC Embedding Comparison Experiment")
    parser.add_argument(
        "--update-data", 
        action="store_true", 
        help="Fetch new FOMC statements and rebuild processed dataset before running experiment."
    )
    args = parser.parse_args()

    print("="*70)
    print("FOMC EMBEDDING COMPARISON EXPERIMENT")
    print("="*70)

    # ============================================================================
    # 0. CHECK FOR UPDATES
    # ============================================================================
    if args.update_data:
        print("\n[0/6] Checking for new FOMC statements...")
        update_fomc_statements()
        process_statements()
    else:
        print("\n[0/6] Skipping update (snapshot mode). Use --update-data to enable.")

    # ============================================================================
    # 1. LOAD DATA
    # ============================================================================
    print("\n[1/6] Loading FOMC dataset...")
    csv_path = Path("data/processed/fomc_statements.csv")
    df = load_fomc_dataset(csv_path)
    print(f"✅ Loaded {len(df)} FOMC statements")

    # ============================================================================
    # 2. SPLIT DATA (chronological - no shuffling!)
    # ============================================================================
    print("\n[2/6] Splitting data chronologically...")
    train_df, test_df = chronological_split(df, train_ratio=0.70)
    print(f"✅ Train: {len(train_df)} statements ({train_df['date'].min().year}-{train_df['date'].max().year})")
    print(f"✅ Test:  {len(test_df)} statements ({test_df['date'].min().year}-{test_df['date'].max().year})")

    # ============================================================================
    # 3. EXTRACT TEXT AND LABELS
    # ============================================================================
    print("\n[3/6] Extracting text and labels...")
    X_train, X_test, y_train_reg, y_test_reg, y_train_class, y_test_class = get_X_y(
        train_df, test_df
    )
    print(f"✅ Training samples: {len(X_train)}")
    print(f"✅ Test samples: {len(X_test)}")

    # ============================================================================
    # 4. GENERATE ALL EMBEDDINGS
    # ============================================================================
    print("\n[4/6] Generating embeddings...")

    # TF-IDF
    print("\n  Generating TF-IDF embeddings...")
    X_train_tfidf, X_test_tfidf, _ = build_tfidf(X_train, X_test)
    print(f"  ✅ TF-IDF: {X_train_tfidf.shape[1]} features")

    # BERT (use cache if available)
    cached = load_embeddings('bert', len(df))
    if cached:
        X_train_bert, X_test_bert = cached
    else:
        print("\n  Loading BERT model...")
        bert_tokenizer, bert_model = load_bert()
        print("  Generating BERT embeddings...")
        X_train_bert = embed_many_bert(X_train, bert_tokenizer, bert_model)
        X_test_bert = embed_many_bert(X_test, bert_tokenizer, bert_model)
        save_embeddings(X_train_bert, X_test_bert, 'bert', len(df))
    print(f"  ✅ BERT: {X_train_bert.shape[1]} features")

    # FinBERT (use cache if available)
    cached = load_embeddings('finbert', len(df))
    if cached:
        X_train_finbert, X_test_finbert = cached
    else:
        print("\n  Loading FinBERT model...")
        finbert_tokenizer, finbert_model = load_finbert()
        print("  Generating FinBERT embeddings...")
        X_train_finbert = embed_many_finbert(X_train, finbert_tokenizer, finbert_model)
        X_test_finbert = embed_many_finbert(X_test, finbert_tokenizer, finbert_model)
        save_embeddings(X_train_finbert, X_test_finbert, 'finbert', len(df))
    print(f"  ✅ FinBERT: {X_train_finbert.shape[1]} features")

    # GTE (use cache if available - MOST IMPORTANT since it's slow!)
    cached = load_embeddings('gte', len(df))
    if cached:
        X_train_gte, X_test_gte = cached
    else:
        print("\n  Loading GTE model...")
        gte_tokenizer, gte_model = load_gte()
        print("  Generating GTE embeddings (this may take a while)...")
        X_train_gte = embed_many_gte(X_train, gte_tokenizer, gte_model)
        X_test_gte = embed_many_gte(X_test, gte_tokenizer, gte_model)
        save_embeddings(X_train_gte, X_test_gte, 'gte', len(df))
    print(f"  ✅ GTE: {X_train_gte.shape[1]} features")

    # Combine into dictionary
    embeddings = {
        'TF-IDF': (X_train_tfidf, X_test_tfidf),
        'BERT': (X_train_bert, X_test_bert),
        'FinBERT': (X_train_finbert, X_test_finbert),
        'GTE': (X_train_gte, X_test_gte)
    }

    print(f"\n✅ All embeddings generated!")

    # ============================================================================
    # 5. RUN CLASSIFICATION EXPERIMENTS
    # ============================================================================
    print("\n[5/6] Running experiments...")
    print("  Running classification experiments...")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Define all classification models with 3 configurations each
    classification_models = {
        'Logistic Regression': {
            'conservative': LogisticRegression(C=0.1),
            'moderate': LogisticRegression(C=1.0),
            'aggressive': LogisticRegression(C=10.0)
        },
        'Random Forest': {
            'conservative': RandomForestClassifier(n_estimators=50, max_depth=3),
            'moderate': RandomForestClassifier(n_estimators=100, max_depth=5),
            'aggressive': RandomForestClassifier(n_estimators=200, max_depth=10)
        },
        'XGBoost': {
            'conservative': XGBoostClassifier(n_estimators=50, max_depth=2, learning_rate=0.01),
            'moderate': XGBoostClassifier(n_estimators=100, max_depth=3, learning_rate=0.1),
            'aggressive': XGBoostClassifier(n_estimators=200, max_depth=5, learning_rate=0.3)
        }
    }

    all_results = {}

    for model_name, configs in classification_models.items():
        all_results[model_name] = {}
        
        for config_name, model in configs.items():
            # Compare all embeddings with this configuration
            results = compare_embeddings(
                embeddings,
                model,
                y_train_class,
                y_test_class,
                task='classification'
            )
            
            # Store results
            all_results[model_name][config_name] = results

    print("  ✅ Classification experiments complete")

    # ============================================================================
    # 6. RUN REGRESSION EXPERIMENTS
    # ============================================================================
    print("  Running regression experiments...")

    # Define all regression models with 3 configurations each
    regression_models = {
        'ElasticNet': {
            'conservative': ElasticNetRegressor(alpha=10.0, l1_ratio=0.5),
            'moderate': ElasticNetRegressor(alpha=1.0, l1_ratio=0.5),
            'aggressive': ElasticNetRegressor(alpha=0.1, l1_ratio=0.5)
        },
        'Random Forest': {
            'conservative': RandomForestRegressor(n_estimators=50, max_depth=3),
            'moderate': RandomForestRegressor(n_estimators=100, max_depth=5),
            'aggressive': RandomForestRegressor(n_estimators=200, max_depth=10)
        },
        'XGBoost': {
            'conservative': XGBoostRegressor(n_estimators=50, max_depth=2, learning_rate=0.01),
            'moderate': XGBoostRegressor(n_estimators=100, max_depth=3, learning_rate=0.1),
            'aggressive': XGBoostRegressor(n_estimators=200, max_depth=5, learning_rate=0.3)
        }
    }

    all_reg_results = {}

    for model_name, configs in regression_models.items():
        all_reg_results[model_name] = {}
        
        for config_name, model in configs.items():
            # Compare all embeddings with this configuration
            results = compare_embeddings(
                embeddings,
                model,
                y_train_reg,
                y_test_reg,
                task='regression'
            )
            
            # Store results
            all_reg_results[model_name][config_name] = results

    print("  ✅ Regression experiments complete")

    # ============================================================================
    # 6. ANALYZE SENSITIVITY AND SELECT MODELS
    # ============================================================================
    print("\n[6/6] Analyzing sensitivity and selecting models...")

    # Analyze sensitivity (returns sensitivity data)
    clf_sensitivity, reg_sensitivity = analyze_sensitivity(
        all_results,
        all_reg_results,
        save_dir=results_dir
    )

    # Create diagnostic summary tables
    composition_path = create_diagnostic_summary(
        clf_sensitivity,
        reg_sensitivity,
        auc_threshold=0.02,
        save_dir=results_dir
    )

    # Print composition file location
    print(f"\n📁 Composition details saved to: {composition_path}")

    # Select final models for classification and regression
    clf_selected, reg_selected = select_models_for_inference(
        all_results,
        all_reg_results,
        clf_sensitivity,
        reg_sensitivity,
        y_test_class=y_test_class
    )

    # ============================================================================
    # 7. SAVE RESULTS
    # ============================================================================
    print("\n" + "="*70)
    print("💾 SAVING RESULTS")
    print("="*70)

    # Save results to CSV
    save_results_to_csv(
        classification_results=all_results,
        regression_results=all_reg_results,
        filename="experiment_results.csv",
        save_dir=results_dir
    )

    # ============================================================================
    # 8. FINAL SUMMARY
    # ============================================================================
    print("\n" + "="*70)
    print("✅ EXPERIMENT COMPLETE!")
    print("="*70)

    if clf_selected:
        print(f"\n📊 SELECTED CLASSIFICATION MODEL:")
        print(f"   {clf_selected['model']} + {clf_selected['embedding']}")
        print(f"   ROC-AUC: {clf_selected.get('performance', clf_selected['mean']):.4f}")
        print(f"   Robustness: CV = {clf_selected['cv_variation']:.1f}%")
    else:
        print("\n⚠️  No robust classification model found!")

    if reg_selected:
        print(f"\n📊 SELECTED REGRESSION MODEL:")
        print(f"   {reg_selected['model']} + {reg_selected['embedding']}")
        print(f"   ROC-AUC: {reg_selected.get('performance', reg_selected['mean']):.4f}")
        print(f"   Robustness: CV = {reg_selected['cv_variation']:.1f}%")
    else:
        print("\n⚠️  No robust regression model found!")

    # ============================================================================
    # 9. SAVE GRAND CHAMPION FOR INFERENCE
    # ============================================================================

    if clf_selected or reg_selected:
        print("\n" + "="*70)
        print("💾 SAVING GRAND CHAMPION FOR INFERENCE")
        print("="*70)
        
        # 1. Determine Winner
        clf_score = clf_selected['performance'] if clf_selected else -1
        reg_score = reg_selected['performance'] if reg_selected else -1
        
        if clf_score >= reg_score and clf_selected:
            winner = clf_selected
            winner_task = 'classification'
            y_full = np.concatenate([y_train_class, y_test_class])
            model_map = classification_models
        elif reg_selected:
            winner = reg_selected
            winner_task = 'regression'
            y_full = np.concatenate([y_train_reg, y_test_reg])
            model_map = regression_models
        else:
            winner = None

        if winner:
            print(f"  🏆 Winner: {winner['model']} ({winner_task})")

            # 2. Get Data & Vectorizer
            emb_name = winner['embedding']
            
            # Prepare X_full
            if emb_name == 'TF-IDF':
                print("  Values: Re-fitting TF-IDF on full corpus for inference...")
                full_corpus = X_train + X_test
                
                # Use sklearn TfidfVectorizer (imported at top)
                # Use fixed parameters for production
                tfidf_prod = TfidfVectorizer(max_features=1000, stop_words='english')
                tfidf_prod.fit(full_corpus)
                
                # Save vectorizer
                vectorizer_path = results_dir / 'tfidf_vectorizer.pkl'
                joblib.dump(tfidf_prod, vectorizer_path)
                print(f"  ✅ Saved vectorizer to {vectorizer_path}")
                print(f"     Vocab size: {len(tfidf_prod.vocabulary_)}")
                
                # Transform data to match this vectorizer
                X_full = tfidf_prod.transform(full_corpus).toarray()
                
            elif emb_name == 'BERT':
                X_full = np.vstack((X_train_bert, X_test_bert))
            elif emb_name == 'FinBERT':
                X_full = np.vstack((X_train_finbert, X_test_finbert))
            elif emb_name == 'GTE':
                X_full = np.vstack((X_train_gte, X_test_gte))
            
            # 3. Retrain Model
            # We use the 'moderate' config as the representative implementation
            print(f"  Training final {winner['model']} (Moderate Config) on full dataset...")
            final_model_template = model_map[winner['model']]['moderate']
            # Create fresh instance using stored init params (custom wrappers don't support sklearn clone)
            final_model = final_model_template.__class__(**final_model_template._init_params)
            final_model.fit(X_full, y_full)
            
            # 4. Save Model
            model_path = results_dir / 'grand_champion_model.pkl'
            joblib.dump(final_model, model_path)
            print(f"  ✅ Saved model to {model_path}")
            
            # 5. Save Config
            config = {
                'model_name': winner['model'],
                'embedding': emb_name,
                'task': winner_task,
                'metric_name': 'ROC-AUC',
                'metric_score': winner['performance']
            }
            with open(results_dir / 'grand_champion_config.json', 'w') as f:
                json.dump(config, f, indent=4)
                
            print(f"\n  ✅ Ready for inference! (See prompt below)")

    print(f"\n📁 Results saved in: {results_dir.absolute()}")
    print("\nFiles created:")
    print("  - experiment_results.csv (All Results)")
    print("  - diagnostic_breakdown.csv (Sensitivity Analysis & Reliability)")

    # ============================================================================
    # 10. OPTIONAL PREDICTION
    # ============================================================================
    try:
        from src.inference import run_interactive_prediction, load_prediction_pipeline
        
        print("\n" + "="*70)
        print("🔮 DATA GENERATION COMPLETE")
        print("="*70)
        
        # Only prompt if running interactively or if desired. 
        # For now, we keep the prompt as requested in original code, 
        # but users can Ctrl+C if they just wanted the batch run.
        while True:
            # We use input() which might block if run non-interactively, 
            # but this is the existing behavior the user didn't ask to change (except update step).
            # The update step request specifically said "No user prompts" for the update itself.
            response = input("\nDo you want to run live predictions with the Grand Champion model? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                run_interactive_prediction()
                break
            elif response in ['n', 'no']:
                print("\n👋 Exiting. Results are saved in the 'results' folder.")
                break
            else:
                print("Please enter 'y' or 'n'.")
                
    except ImportError as e:
        print(f"\n⚠️  Could not import prediction module: {e}")
        print("Prediction step skipped.")

    print("\n" + "="*70)

if __name__ == "__main__":
    main()

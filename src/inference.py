"""
Interactive prediction module for FOMC statement analysis.

This module provides functionality to load the best-performing model
and make predictions on new FOMC statement text. It includes an
interactive CLI for real-time predictions using the grand champion model.
"""

import joblib
import json
import numpy as np
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Import embedding modules
from src.embedder_bert import load_bert, embed_many_bert
from src.embedder_finbert import load_finbert, embed_many_finbert
from src.embedder_gte import load_gte, embed_many_gte

def load_prediction_pipeline(results_dir="results"):
    results_path = Path(results_dir)
    config_path = results_path / 'grand_champion_config.json'
    
    if not config_path.exists():
        print("❌ Model configuration not found!")
        return None, None, None
        
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    print(f"\n📂 Loading Model: {config['model_name']} ({config['task']})")
    print(f"🔌 Embedding:   {config['embedding']}")
    
    # Load Model
    model_path = results_path / 'grand_champion_model.pkl'
    if not model_path.exists():
        print(f"❌ Model file missing: {model_path}")
        return None, None, None
    
    model = joblib.load(model_path)
    
    # Load Vectorizer/Embedder Resources
    resources = {}
    if config['embedding'] == 'TF-IDF':
        vec_path = results_path / 'tfidf_vectorizer.pkl'
        if not vec_path.exists():
            print(f"❌ Vectorizer missing: {vec_path}")
            return None, None, None
        resources['vectorizer'] = joblib.load(vec_path)
        
    elif config['embedding'] == 'BERT':
        print("   Loading BERT model (this may take a moment)...")
        tokenizer, bert_model = load_bert()
        resources['tokenizer'] = tokenizer
        resources['model'] = bert_model
        
    elif config['embedding'] == 'FinBERT':
        print("   Loading FinBERT model (this may take a moment)...")
        tokenizer, finbert_model = load_finbert()
        resources['tokenizer'] = tokenizer
        resources['model'] = finbert_model
        
    elif config['embedding'] == 'GTE':
        print("   Loading GTE model (this may take a moment)...")
        tokenizer, gte_model = load_gte()
        resources['tokenizer'] = tokenizer
        resources['model'] = gte_model
        
    return model, config, resources

def predict_text(text, model, config, resources):
    print("\n   Generating embedding...", end="\r")
    
    # Generate Embedding
    if config['embedding'] == 'TF-IDF':
        embedding = resources['vectorizer'].transform([text]).toarray()
        
    elif config['embedding'] == 'BERT':
        embedding = embed_many_bert([text], resources['tokenizer'], resources['model'])
        
    elif config['embedding'] == 'FinBERT':
        embedding = embed_many_finbert([text], resources['tokenizer'], resources['model'])
        
    elif config['embedding'] == 'GTE':
        embedding = embed_many_gte([text], resources['tokenizer'], resources['model'])
    
    print("   Running prediction...    ", end="\r")
    
    # Predict
    prediction = model.predict(embedding)[0]
    
    print(" " * 40, end="\r") # Clear line
    
    # Output Result
    print("-" * 50)
    print(f"📝 Input: \"{text[:60]}...\"")
    print()
    
    if config['task'] == 'classification':
        direction = "UP 🟢" if prediction == 1 else "DOWN 🔴"
        print(f"🔮 Prediction: Market {direction}")
            
    else: # Regression
        direction = "UP 🟢" if prediction > 0 else "DOWN 🔴"
        print(f"🔮 Prediction: Market {direction}")
        print(f"📈 Predicted Return: {prediction:.4f}%")
        
    print("-" * 50)

def run_interactive_prediction():
    """Run the interactive prediction loop."""
    print("\n" + "="*60)
    print("🤖 FOMC PREDICTOR - GRAND CHAMPION MODEL")
    print("="*60)
    
    try:
        model, config, resources = load_prediction_pipeline()
        
        if model is None:
            print("❌ Initialization failed.")
            return

        print("\n✅ System Ready. Type a sentence to predict (or 'exit').")
        
        while True:
            try:
                user_input = input("\n💬 Text > ").strip()
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                if len(user_input) < 5:
                    print("⚠️  Text too short. Please enter a meaningful sentence.")
                    continue
                    
                predict_text(user_input, model, config, resources)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error during prediction: {e}")
                
    except Exception as e:
        print(f"\n❌ Initialization Error: {e}")

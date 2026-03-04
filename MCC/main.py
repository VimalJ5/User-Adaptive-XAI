# main.py
import os
import pandas as pd
import argparse
from datetime import datetime

# Import from our modularized structure
from config import Config
from src.model_loader import ModelManager
from src.explainer import LimeExplainerWrapper
from src.ontology import OntologyManager
from src.generator import ExplanationGenerator

# Mapping based on your notebook
LABEL_TO_CLASS_MAPPING = {
    'LABEL_0': 'Neoplasms', 
    'LABEL_1': 'Digestive system diseases',
    'LABEL_2': 'Nervous system diseases',
    'LABEL_3': 'Cardiovascular diseases',
    'LABEL_4': 'General pathological conditions',
    # Fallbacks depending on how your specific model's config.json is set up
    'Class_0': 'Neoplasms',
    'Class_1': 'Digestive system diseases',
    'Class_2': 'Nervous system diseases',
    'Class_3': 'Cardiovascular diseases',
    'Class_4': 'General pathological conditions'
}

CLASS_NAMES = [
    "Neoplasms", 
    "Digestive system diseases", 
    "Nervous system diseases", 
    "Cardiovascular diseases", 
    "General pathological conditions"
]

def run_experiment(input_file, user_mode, limit=None):
    print(f"🚀 Starting Experiment: Mode={user_mode}")
    
    # Ensure save directories exist
    Config.ensure_dirs()
    
    # 1. Load Data
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {input_file}. Please ensure your dataset exists.")
        return

    if limit:
        df = df.head(limit)
    
    # 2. Initialize Components
    print("⏳ Initializing models and managers (this may take a minute on the first run)...")
    model_manager = ModelManager()
    classifier = model_manager.get_classifier() # We need this to get the top prediction for the LLM
    
    ontology_manager = OntologyManager()
    explainer = LimeExplainerWrapper(class_names=CLASS_NAMES)
    generator = ExplanationGenerator()
    
    results = []
    
    # 3. The Efficiency Loop
    for idx, row in df.iterrows():
        # Adjust 'text' below if your CSV column has a different name
        text = row['text'] 
        print(f"\n[{idx+1}/{len(df)}] Processing text...")
        
        # Step 0: Get the model's actual prediction
        # We run it on the first 512 tokens to match max_length constraints
        prediction_output = classifier(text[:512], top_k=1)[0]
        predicted_id = prediction_output['label']
        predicted_class = LABEL_TO_CLASS_MAPPING.get(predicted_id, predicted_id)
        print(f"  -> Prediction: {predicted_class} (Confidence: {prediction_output['score']:.2f})")
        
        # Step A: Explain (Cached)
        print("  -> Extracting LIME Features...")
        lime_features = explainer.explain(text)
        
        # Step B: Ontology Lookup
        print("  -> Enriching with Ontology...")
        feature_data = ontology_manager.enrich_features(lime_features, user_category=user_mode)
        
        # Step C: Generate
        print("  -> Generating Explanation...")
        explanation = generator.generate(
            predicted_class=predicted_class, 
            feature_data=feature_data, 
            user_category=user_mode
        )
        
        # Store result
        results.append({
            "text_id": idx,
            "original_text": text[:100] + "...", # Save snippet for reference
            "user_mode": user_mode,
            "predicted_class": predicted_class,
            "confidence": round(prediction_output['score'], 3),
            "explanation": explanation
        })

    # 4. Save Results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"{Config.RESULTS_DIR}/exp_{user_mode}_{timestamp}.csv"
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path, index=False)
    
    print(f"\n✅ Experiment Complete!")
    print(f"💾 Results successfully saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the User-Adaptive XAI Pipeline")
    parser.add_argument("--mode", type=str, default="EXPERT", choices=["EXPERT", "BEGINNER"], 
                        help="Target user category for the explanation.")
    parser.add_argument("--limit", type=int, default=5, 
                        help="Number of rows to process from the CSV.")
    parser.add_argument("--input", type=str, default="data/input.csv", 
                        help="Path to the input CSV file containing a 'text' column.")
    
    args = parser.parse_args()
    
    run_experiment(args.input, args.mode, args.limit)
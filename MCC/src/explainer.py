# src/explainer.py
import os
import pickle
import hashlib
from lime.lime_text import LimeTextExplainer
from config import Config
from src.model_loader import ModelManager

class LimeExplainerWrapper:
    def __init__(self, class_names):
        self.explainer = LimeTextExplainer(class_names=class_names)
        self.classifier = ModelManager().get_classifier()
        self.ner = ModelManager().get_ner()
        Config.ensure_dirs()

    def _get_cache_path(self, text):
        # Create a unique filename based on the text hash
        hash_id = hashlib.md5(text.encode()).hexdigest()
        return os.path.join(Config.CACHE_DIR, f"lime_{hash_id}.pkl")

    def explain(self, text, num_features=6):
        cache_path = self._get_cache_path(text)
        
        # 1. Try Loading from Cache
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        
        # 2. Compute if not cached
        # (Insert your entity merging logic here if needed)
        def predictor(texts):
            results = self.classifier(texts, truncation=True, max_length=512)
            # Convert pipeline output to format LIME expects
            # Note: This part depends on your specific model output format
            # You might need to adjust the parsing logic
            probs = []
            for res in results:
                # Assuming binary/multi-class logic here
                score = res['score']
                label_id = int(res['label'].split('_')[-1]) # e.g. LABEL_0 -> 0
                # Construct probability vector (simplification)
                vec = [0.0] * 5 
                vec[label_id] = score
                probs.append(vec) 
            return probs

        exp = self.explainer.explain_instance(
            text, predictor, num_features=num_features, num_samples=Config.LIME_SAMPLES
        )
        features = exp.as_list()
        
        # 3. Save to Cache
        with open(cache_path, "wb") as f:
            pickle.dump(features, f)
            
        return features
# config.py
import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "Models", "my_medical_model")
    ONTOLOGY_PATH = os.path.join(BASE_DIR, "Ontology", "doid.owl")
    RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
    
    # Model Settings
    LLM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
    NER_MODEL_ID = "d4data/biomedical-ner-all"
    DEVICE = "cuda"  # or "cpu"
    
    # Experiment Settings
    LIME_SAMPLES = 500  # Increase for better quality, decrease for speed
    CACHE_DIR = os.path.join(BASE_DIR, "cache")

    @staticmethod
    def ensure_dirs():
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
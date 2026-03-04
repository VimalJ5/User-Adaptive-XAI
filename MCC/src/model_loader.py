# src/model_loader.py
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModelForTokenClassification, 
    AutoModelForCausalLM,
    pipeline
)
from config import Config

class ModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.models = {}
        return cls._instance

    def get_classifier(self):
        if "classifier" not in self.models:
            print("⏳ Loading Classifier...")
            tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_DIR)
            model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_DIR)
            self.models["classifier"] = pipeline(
                "text-classification", model=model, tokenizer=tokenizer, 
                device=0 if torch.cuda.is_available() else -1
            )
        return self.models["classifier"]

    def get_ner(self):
        if "ner" not in self.models:
            print("⏳ Loading NER...")
            tokenizer = AutoTokenizer.from_pretrained(Config.NER_MODEL_ID)
            model = AutoModelForTokenClassification.from_pretrained(Config.NER_MODEL_ID)
            self.models["ner"] = pipeline(
                "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
        return self.models["ner"]

    def get_llm(self):
        if "llm" not in self.models:
            print(f"⏳ Loading LLM ({Config.LLM_MODEL_ID})...")
            self.models["llm_tokenizer"] = AutoTokenizer.from_pretrained(Config.LLM_MODEL_ID)
            self.models["llm"] = AutoModelForCausalLM.from_pretrained(
                Config.LLM_MODEL_ID,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        return self.models["llm"], self.models["llm_tokenizer"]
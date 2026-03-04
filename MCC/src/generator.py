import torch
from config import Config
from src.model_loader import ModelManager

class ExplanationGenerator:
    def __init__(self):
        self.model_manager = ModelManager()
        # System Prompt from your notebook
        self.SYSTEM_PROMPT = """
You are an explanation generator for a biomedical XAI system.

You do NOT classify text.
You do NOT add external medical knowledge.
You do NOT infer physiology, causation, or mechanisms.

Your ONLY role:
Convert the structured explanation plan (LIME tokens + ontology ancestors +
model prediction) into a clear natural-language explanation.

Rules:
1. Use ONLY the provided ontology triples and model prediction.
2. NEVER add biomedical facts that are not explicitly listed.
3. If a concept has no relation to the predicted class, state this neutrally.
4. Beginner → broad/simple language.
5. Expert → technical terms and ontology hierarchy words.
6. DO NOT write with bullet points. Produce one coherent paragraph.
7. Be concise and strictly grounded.
"""

    def _construct_prompt(self, predicted_class, feature_data, user_category):
        """Constructs the User Prompt string."""
        
        # Build structured triple block
        triple_lines = []
        salient_tokens = []
        
        for item in feature_data:
            word = item['feature_word']
            ancestors = item['ancestors']
            salient_tokens.append(word)
            
            if len(ancestors) > 0:
                # Format: (ischemia -> ancestor -> vascular disease)
                chain = " -> ".join(ancestors)
                triple_lines.append(f"({word} → ancestor → {chain})")
            else:
                triple_lines.append(f"({word} → ancestor → NONE)")
        
        triples_block = "\n".join(triple_lines)

        prompt = f"""
### INPUT
text: [REDACTED FOR BREVITY]
Prediction: {predicted_class}

Salient tokens identified by LIME:
{salient_tokens}

Ontology triples (feature → relation → ancestor):
{triples_block}

User type: {user_category}

### OUTPUT REQUIREMENTS
Write a single coherent explanation that:
- States the prediction.
- Uses ONLY the information above.
- Maps each token to its listed ancestors.
- If a token has no biomedical relevance to the prediction, state this without guessing.
- For BEGINNER → use broad/simple language.
- For EXPERT → include hierarchy terms like "anatomical entity", "subclass of", etc.
- Do NOT add medical facts not listed in the triples.
- Do NOT speculate, infer physiology, or make causal claims.

### EXPLANATION:
"""
        return prompt

    def generate(self, predicted_class, feature_data, user_category, max_new_tokens=180):
        """
        Main generation function.
        """
        if not feature_data:
            return f"The model predicted {predicted_class}, but no ontology-based features were available."

        # 1. Prepare Prompts
        task_prompt = self._construct_prompt(predicted_class, feature_data, user_category)
        
        # 2. Get Model & Tokenizer (Singleton)
        llm, tokenizer = self.model_manager.get_llm()
        
        # 3. Apply Chat Template
        full_prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": task_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True
        )

        # 4. Generate
        inputs = tokenizer(full_prompt, return_tensors="pt").to(llm.device)
        
        with torch.no_grad():
            output = llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,  # Low temp for factual consistency
                do_sample=True,
                repetition_penalty=1.1
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)

        # 5. Parse Output (Strip prompt)
        if "### EXPLANATION:" in decoded:
            explanation = decoded.split("### EXPLANATION:")[-1].strip()
        elif "assistant" in decoded:
             # Fallback if the chat template adds 'assistant' label
             explanation = decoded.split("assistant")[-1].strip()
        else:
            explanation = decoded.strip()

        return explanation
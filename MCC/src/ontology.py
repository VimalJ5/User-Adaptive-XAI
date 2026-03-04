import owlready2
from config import Config

class OntologyManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OntologyManager, cls).__new__(cls)
            cls._instance.ontology = None
            cls._instance.load_ontology()
        return cls._instance

    def load_ontology(self):
        """Loads the .owl file specified in Config."""
        if self.ontology is None:
            print(f"📚 Loading Ontology from {Config.ONTOLOGY_PATH}...")
            try:
                self.ontology = owlready2.get_ontology(Config.ONTOLOGY_PATH).load()
                print("✅ Ontology loaded successfully.")
            except Exception as e:
                print(f"❌ Error loading ontology: {e}")
                self.ontology = None

    def find_concept(self, term):
        """
        Searches the ontology for a class matching the term (label or synonym).
        Returns the first matching OwlReady2 Class object or None.
        """
        if not self.ontology:
            return None
        
        term_lower = term.lower().replace("_", " ")
        
        # 1. Direct search by IRI or name
        result = self.ontology.search_one(label=term_lower)
        if result:
            return result
            
        # 2. Fuzzy search over labels and synonyms if exact match fails
        # Note: This is a simplified search. You can expand with specific annotation properties.
        results = self.ontology.search(label=f"*{term_lower}*")
        if results:
            return results[0] # Return best match
            
        return None

    def get_ancestors(self, concept_class):
        """Retrieves a list of ancestor labels for a given concept."""
        if not concept_class:
            return []
        
        # owlready2 .ancestors() returns a set of classes
        # We convert them to string labels and filter out the root 'Thing'
        ancestors = set()
        for anc in concept_class.ancestors():
            if hasattr(anc, "label") and anc.label:
                # anc.label is often a list, take the first one
                label = anc.label[0] if isinstance(anc.label, list) else anc.label
                ancestors.add(label)
        
        # Remove the concept itself and generic roots if desired
        if hasattr(concept_class, "label"):
            self_label = concept_class.label[0] if isinstance(concept_class.label, list) else concept_class.label
            if self_label in ancestors:
                ancestors.remove(self_label)
                
        return list(ancestors)

    def select_ancestors(self, concept_class, user_category):
        """
        Adaptive Ancestor Selection:
        - EXPERT: Returns specific, deep ancestors (Full hierarchy).
        - BEGINNER: Returns only high-level, broad categories (Shallow hierarchy).
        """
        all_ancestors = self.get_ancestors(concept_class)
        
        if not all_ancestors:
            return []

        # Logic: Experts get technical granularity; Beginners get broad categories.
        # Ideally, you'd calculate depth here. For now, we simulate this by 
        # picking specific known high-level concepts for beginners.
        
        if user_category == "EXPERT":
            # Return almost everything (Technical)
            return all_ancestors
        
        elif user_category == "BEGINNER":
            # Filter for broad, recognizable terms
            # In a real DOID implementation, you might filter by depth < 3
            broad_terms = [
                "disease", "syndrome", "disorder", "organ system disease", 
                "cardiovascular system disease", "nervous system disease", 
                "gastrointestinal system disease", "cancer", "symptom"
            ]
            return [anc for anc in all_ancestors if anc.lower() in broad_terms]
            
        return all_ancestors

    def enrich_features(self, lime_features, user_category="EXPERT"):
        """
        Main pipeline function.
        Input: List of (feature_word, score) from LIME.
        Output: Structured list of dicts with ancestors.
        """
        enriched_data = []
        
        # Process top features (e.g., top 6)
        top_features = [f[0] for f in lime_features[:6]]
        
        for feature_word in top_features:
            # 1. Find Concept
            concept = self.find_concept(str(feature_word))
            
            if concept:
                # 2. Adaptive Selection
                ancestors = self.select_ancestors(concept, user_category)
                
                enriched_data.append({
                    'feature_word': str(feature_word),
                    'ancestors': ancestors
                })
            else:
                # Valid to include features even if ontology lookup fails? 
                # Your notebook skips them, so we skip them here too.
                continue
                
        return enriched_data
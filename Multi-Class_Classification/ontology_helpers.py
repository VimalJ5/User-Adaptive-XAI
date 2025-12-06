import math
from typing import List, Tuple, Optional
from owlready2 import get_ontology, Thing

# Module-level variables to cache ontology statistics for scoring
# These are populated when load_ontology() is called
MAX_DEPTH = 12
TOTAL_CLASSES = 0
OBJECT_PROPERTIES = []

def _class_depth_raw(concept) -> int:
    """
    Internal helper: Calculates the raw depth of a concept in the hierarchy.
    """
    try:
        ancestors = [a for a in concept.ancestors() if isinstance(a, type)]
        if Thing in ancestors:
            ancestors = [a for a in ancestors if a is not Thing]
        return max(len(ancestors), 0)
    except:
        return 0

def load_ontology(owl_path: str):
    """
    Loads the ontology from the specified path and pre-computes global statistics
    (Total Classes, Max Depth) required for the adaptive scoring metrics.
    """
    global MAX_DEPTH, TOTAL_CLASSES, OBJECT_PROPERTIES
    
    print(f"Stage: LOADING ONTOLOGY from '{owl_path}'")
    ont = get_ontology(owl_path).load()
    print("Stage: ONTOLOGY loaded")
    
    # PRE-COMPUTE GLOBAL STATS FOR NORMALIZATION
    classes = list(ont.classes())
    TOTAL_CLASSES = len(classes)
    OBJECT_PROPERTIES = list(ont.object_properties())
    
    # Estimate Max Depth (Heuristic scan of first 500 classes for performance)
    current_max = 0
    scan_limit = 500
    for c in classes[:scan_limit]:
        d = _class_depth_raw(c)
        if d > current_max: current_max = d
    
    MAX_DEPTH = current_max if current_max > 0 else 12 # Default fallback
    
    # print(f"Stage: Stats - Total Classes: {TOTAL_CLASSES}, Est. Max Depth: {MAX_DEPTH}")
    return ont

def find_concept(ontology, label_or_name: str):
    """
    Robust search for a concept by its label (preferred) or internal name.
    """
    # 1. Try exact label search
    candidates = list(ontology.search(label=label_or_name))
    
    # 2. Try manual iteration if search fails (sometimes needed for specific OWL formats)
    if not candidates:
        candidates = [
            c for c in ontology.classes() 
            if any(str(lbl).lower() == label_or_name.lower() 
                   for lbl in getattr(c, 'label', []))
        ]
    
    if candidates:
        return candidates[0]
        
    # 3. Fallback to direct lookup by ID/Name
    try:
        return ontology[label_or_name]
    except Exception:
        return None

def calculate_suitability_score(concept, user_category: str) -> float:
    """
    Calculates a suitability score (0.0 to 1.0) for a concept based on the User Category.
    
    Metrics:
    - Specificity: Normalized depth in the tree.
    - Popularity: Normalized number of incoming object properties.
    
    Strategies:
    - BEGINNER: Prefers High Popularity and Low Specificity.
    - EXPERT: Prefers High Specificity (Popularity is ignored).
    - INTERMEDIATE: Prefers a balance (Bell curve around 0.5 specificity).
    """
    # 1. Calculate Specificity (S) 
    raw_depth = _class_depth_raw(concept)
    specificity = raw_depth / MAX_DEPTH if MAX_DEPTH > 0 else 0
    
    # 2. Calculate Popularity (P) 
    incoming_links = 0
    for prop in OBJECT_PROPERTIES:
        # Check if this concept is the target (Range) of a property
        if concept in prop.range:
            incoming_links += 1
            
    popularity = incoming_links / TOTAL_CLASSES if TOTAL_CLASSES > 0 else 0
    
    # 3. User-Adaptive Scoring Logic
    cat = user_category.upper()
    score = 0.0
    
    if cat == 'BEGINNER':
        # Weights: 60% Popularity, 40% Generality (1 - Specificity)
        score = (0.6 * popularity) + (0.4 * (1.0 - specificity))
        
    elif cat == 'EXPERT':
        # Experts want the most specific, technical term possible.
        score = specificity
        
    elif cat == 'INTERMEDIATE':
        # Intermediates need the "middle" ground.
        score = 1.0 - abs(specificity - 0.5)
        
    return score

def get_ancestors(concept, include_self=False):
    """Returns a sorted list of ancestors (by depth)."""
    ancestors = [a for a in concept.ancestors() if isinstance(a, type) and a is not Thing]
    ancestors_sorted = sorted(ancestors, key=lambda x: _class_depth_raw(x))
    return [concept] + ancestors_sorted if include_self else ancestors_sorted

# def select_ancestors(entity_concept, user_category: str) -> List[str]:
#     """
#     Main logic function: Selects the top-k most suitable ancestors 
#     for a given entity and user category.
#     """
#     if entity_concept is None:
#         return []

#     # 1. Get all candidates
#     ancestors = get_ancestors(entity_concept)
    
#     # 2. Score candidates based on user profile
#     scored_candidates = []
#     for anc in ancestors:
#         score = calculate_suitability_score(anc, user_category)
#         scored_candidates.append((anc, score))
        
#     # 3. Sort by Score (Descending) - Best fit first
#     scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
#     # 4. Selection Heuristics (How many to show?)
#     if user_category.upper() == 'EXPERT':
#         top_k = 4
#     elif user_category.upper() == 'BEGINNER':
#         top_k = 3
#     else: # INTERMEDIATE
#         top_k = 3
    
#     selected_tuples = scored_candidates[:top_k]
    
#     # 5. Extract Labels and re-sort by hierarchy depth for logical reading order
#     def label(c):
#         return str(getattr(c, 'label', [c.name])[0])

#     selected_objs = [obj for obj, score in selected_tuples]
#     # Sort by depth so the output explanation reads: General -> Specific
#     selected_objs.sort(key=lambda x: _class_depth_raw(x))
    
#     final_labels = [label(o) for o in selected_objs]

#     return final_labels

def select_ancestors(entity_concept, user_category: str, ablation_mode: str = "normal"):
    if entity_concept is None:
        return []

    # 1. Get all ancestors (raw list)
    ancestors = get_ancestors(entity_concept)

    # ------- 🔥 ABLATION MODES HERE -------
    if ablation_mode == "full":  
        # Return ALL ancestors in natural hierarchy order
        return [str(getattr(a, 'label', [a.name])[0]) for a in ancestors]

    if ablation_mode == "one_parent":  
        # Only the immediate parent (superclass)
        if len(ancestors) >= 1:
            a = ancestors[0]
            return [str(getattr(a, 'label', [a.name])[0])]
        else:
            return []

    if ablation_mode == "no_ontology":
        # Return empty list (control)
        return []

    # ------- NORMAL USER-ADAPTIVE MODE -------
    scored_candidates = []
    for anc in ancestors:
        score = calculate_suitability_score(anc, user_category)
        scored_candidates.append((anc, score))
        
    # Sort by suitability score
    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    # Pick top-k based on user category
    if user_category.upper() == "EXPERT":
        top_k = 4
    elif user_category.upper() == "BEGINNER":
        top_k = 3
    else:
        top_k = 3

    selected_objs = [obj for obj, _ in scored_candidates[:top_k]]

    # Sort final list by depth (general → specific)
    selected_objs.sort(key=lambda x: _class_depth_raw(x))

    # Convert to string labels
    def label(c):
        return str(getattr(c, 'label', [c.name])[0])

    return [label(o) for o in selected_objs]

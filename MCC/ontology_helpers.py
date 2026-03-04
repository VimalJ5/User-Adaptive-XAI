"""
ontology_helpers.py
===================
Ontology loading, concept lookup, difficulty scoring, and ancestor selection.

Public API
----------
    load_ontology(owl_path)                          → owlready2 ontology
    find_concept(ontology, label_or_name)            → concept | None
    select_ancestors(concept, user_category, mode)  → List[str]
    calculate_suitability_score(concept, user_cat)  → float
    get_ancestors(concept, include_self)             → List[concept]
"""

from __future__ import annotations

from typing import Optional
from owlready2 import get_ontology, Thing

# ── Module-level cache (populated by load_ontology) ──────────────────────────
_MAX_DEPTH: int = 12
_TOTAL_CLASSES: int = 0
_OBJECT_PROPERTIES: list = []


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _class_depth(concept) -> int:
    """Return the depth of *concept* in the class hierarchy (0-based)."""
    try:
        ancestors = [a for a in concept.ancestors() if isinstance(a, type) and a is not Thing]
        return len(ancestors)
    except Exception:
        return 0


def _label(concept) -> str:
    """Return the preferred string label for *concept*."""
    return str(getattr(concept, "label", [concept.name])[0])


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_ontology(owl_path: str):
    """
    Load the OWL ontology and pre-compute global statistics needed for
    adaptive scoring (total classes, max depth, object properties).

    Parameters
    ----------
    owl_path : str
        Local file path or IRI of the OWL file.

    Returns
    -------
    owlready2 ontology object
    """
    global _MAX_DEPTH, _TOTAL_CLASSES, _OBJECT_PROPERTIES

    print(f"[Ontology] Loading from '{owl_path}' …")
    ont = get_ontology(owl_path).load()
    print("[Ontology] Loaded successfully.")

    classes = list(ont.classes())
    _TOTAL_CLASSES = len(classes)
    _OBJECT_PROPERTIES = list(ont.object_properties())

    # Estimate max depth from the first 500 classes (performance heuristic)
    scan_limit = min(500, len(classes))
    max_depth = max((_class_depth(c) for c in classes[:scan_limit]), default=0)
    _MAX_DEPTH = max_depth if max_depth > 0 else 12  # fallback default

    print(f"[Ontology] Stats → classes: {_TOTAL_CLASSES}, est. max depth: {_MAX_DEPTH}")
    return ont


def find_concept(ontology, label_or_name: str):
    """
    Search for a concept by label (case-insensitive) or internal name.

    Lookup order:
        1. owlready2 label search
        2. Manual label iteration (handles some OWL format quirks)
        3. Direct name/ID fallback

    Returns
    -------
    concept or None
    """
    # 1. Fast owlready2 label search
    candidates = list(ontology.search(label=label_or_name))

    # 2. Case-insensitive manual scan
    if not candidates:
        lower = label_or_name.lower()
        candidates = [
            c for c in ontology.classes()
            if any(str(lbl).lower() == lower for lbl in getattr(c, "label", []))
        ]

    if candidates:
        return candidates[0]

    # 3. Direct ID / name lookup
    try:
        return ontology[label_or_name]
    except Exception:
        return None


def get_ancestors(concept, include_self: bool = False) -> list:
    """
    Return ancestors of *concept* sorted from most-general to most-specific
    (ascending depth order), excluding owl:Thing.

    Parameters
    ----------
    include_self : bool
        Prepend *concept* itself to the returned list.
    """
    ancestors = [
        a for a in concept.ancestors()
        if isinstance(a, type) and a is not Thing
    ]
    ancestors.sort(key=_class_depth)
    return ([concept] + ancestors) if include_self else ancestors


def calculate_suitability_score(concept, user_category: str) -> float:
    """
    Compute a suitability score in [0, 1] for *concept* given *user_category*.

    Metrics
    -------
    - Specificity  S(c) = depth(c) / max_depth          (higher → more technical)
    - Popularity   P(c) = incoming_props / total_classes (higher → more common)

    Scoring
    -------
    - BEGINNER:     0.6 * P  +  0.4 * (1 - S)   — prefers popular, general concepts
    - EXPERT:       S                             — prefers deep, specific concepts
    - INTERMEDIATE: 1 - |S - 0.5|               — prefers mid-level concepts
    """
    # Specificity
    specificity = _class_depth(concept) / _MAX_DEPTH if _MAX_DEPTH > 0 else 0.0

    # Popularity (number of object properties whose range includes this concept)
    incoming = sum(1 for prop in _OBJECT_PROPERTIES if concept in prop.range)
    popularity = incoming / _TOTAL_CLASSES if _TOTAL_CLASSES > 0 else 0.0

    cat = user_category.upper()
    if cat == "BEGINNER":
        return 0.6 * popularity + 0.4 * (1.0 - specificity)
    elif cat == "EXPERT":
        return specificity
    else:  # INTERMEDIATE
        return 1.0 - abs(specificity - 0.5)


def select_ancestors(
    entity_concept,
    user_category: str,
    ablation_mode: str = "normal",
) -> list[str]:
    """
    Select the most suitable ancestor labels for *entity_concept* based on
    *user_category* and *ablation_mode*.

    Parameters
    ----------
    entity_concept : owlready2 class or None
    user_category  : "BEGINNER" | "INTERMEDIATE" | "EXPERT"
    ablation_mode  : "normal"      — user-adaptive top-k selection  (default)
                     "full"        — all ancestors in hierarchy order
                     "one_parent"  — immediate parent only
                     "no_ontology" — empty list (control condition)

    Returns
    -------
    List of string labels, ordered general → specific.
    """
    if entity_concept is None:
        return []

    ancestors = get_ancestors(entity_concept)

    # ── Ablation modes ────────────────────────────────────────────────────────
    if ablation_mode == "full":
        return [_label(a) for a in ancestors]

    if ablation_mode == "one_parent":
        return [_label(ancestors[0])] if ancestors else []

    if ablation_mode == "no_ontology":
        return []

    # ── Normal: user-adaptive top-k ──────────────────────────────────────────
    scored = [
        (a, calculate_suitability_score(a, user_category))
        for a in ancestors
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    top_k = 4 if user_category.upper() == "EXPERT" else 3
    selected = [concept for concept, _ in scored[:top_k]]

    # Re-sort selected ancestors general → specific for readable output
    selected.sort(key=_class_depth)
    return [_label(a) for a in selected]

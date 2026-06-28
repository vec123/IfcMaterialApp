# src/query/typology_filters.py

import operator
from typing import Dict, Any, List

# Map string operators to native Python operators safely
OPERATOR_MAP = {
    "==": operator.eq,
    ">":  operator.gt,
    "<":  operator.lt,
    ">=": operator.ge,
    "<=": operator.le
}

def filter_by_typology_id(typologies: Dict[str, Any], typology_ids: List[str], **kwargs) -> Dict[str, Any]:
    if not typology_ids:
        return typologies
    return {k: v for k, v in typologies.items() if k in typology_ids}


def filter_by_element_class(typologies: Dict[str, Any], ifc_class: str, **kwargs) -> Dict[str, Any]:
    if not ifc_class:
        return typologies
    target = ifc_class.lower()
    # Scans the inherited structural assignments stored inside metadata attributes
    return {
        k: v for k, v in typologies.items() 
        if str(v.get("ifc_element_class", "")).lower() == target or target in str(v.get("associated_classes", [])).lower()
    }


def filter_by_layer_count(typologies: Dict[str, Any], layer_count: int = None, operator_str: str = "==", **kwargs) -> Dict[str, Any]:

    clean_op = kwargs.get("operator", operator_str).strip()

    min_layers = kwargs.get("min_layers")
    max_layers = kwargs.get("max_layers")
    
    filtered = {}
    for k, v in typologies.items():
        layers = v.get("layers", [])
        actual_count = len(layers)
        
        # Scenario A: Range match handling ("between X and Y")
        if min_layers is not None or max_layers is not None:
            # If min_layers is provided, actual must be >= min
            satisfies_min = (actual_count >= int(min_layers)) if min_layers is not None else True
            # If max_layers is provided, actual must be <= max
            satisfies_max = (actual_count <= int(max_layers)) if max_layers is not None else True
            
            if satisfies_min and satisfies_max:
                filtered[k] = v
                
        # Scenario B: Standard single operator matching fallback
        elif layer_count is not None:
            op_func = OPERATOR_MAP.get(clean_op, operator.eq)
            if op_func(actual_count, int(layer_count)):
                filtered[k] = v
                
    return filtered

def filter_by_material_mix(typologies: Dict[str, Any], materials: List[str], matching_mode: str = "AND", **kwargs) -> Dict[str, Any]:
    if not materials:
        return typologies
    
    target_materials = [m.lower().strip() for m in materials]
    filtered = {}
    
    for k, v in typologies.items():
        # Harvest all unique material token tags embedded inside the typology definition layer track
        embedded_materials = {str(layer.get("material_id", "")).lower() for layer in v.get("layers", [])}
        
        if matching_mode == "AND":
            # Check if every single target material is accounted for
            if all(any(target in emb for emb in embedded_materials) for target in target_materials):
                filtered[k] = v
        else: # OR Match
            if any(any(target in emb for emb in embedded_materials) for target in target_materials):
                filtered[k] = v
                
    return filtered


def filter_by_total_thickness(typologies: Dict[str, Any], thickness_m: float = None, operator_str: str = "==", tolerance: float = 0.005, **kwargs) -> Dict[str, Any]:
    clean_op = kwargs.get("operator", operator_str).strip()
    min_thick = kwargs.get("min_thickness")
    max_thick = kwargs.get("max_thickness")
    
    filtered = {}
    for k, v in typologies.items():
        total_depth = v.get("total_thickness_m")
        if total_depth is None:
            total_depth = sum(float(layer.get("thickness_m", 0.0)) for layer in v.get("layers", []))
            
        # Range handling
        if min_thick is not None or max_thick is not None:
            satisfies_min = (total_depth >= float(min_thick)) if min_thick is not None else True
            satisfies_max = (total_depth <= float(max_thick)) if max_thick is not None else True
            if satisfies_min and satisfies_max:
                filtered[k] = v
        
        # Single value matching
        elif thickness_m is not None:
            if clean_op == "==":
                if abs(total_depth - thickness_m) <= tolerance:
                    filtered[k] = v
                    continue
            
            op_func = OPERATOR_MAP.get(clean_op, operator.eq)
            if op_func(total_depth, thickness_m):
                filtered[k] = v
                
    return filtered
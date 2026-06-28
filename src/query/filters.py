import math
import json
from typing import List, Dict, Any, Union, Callable


def _is_nan_value(value: Any) -> bool:
    """Checks if a value is effectively 'Not a Number' or None."""
    return (
        value is None or 
        (isinstance(value, float) and math.isnan(value)) or 
        str(value).lower() == 'nan'
    )


def get_constructions_by_materials(
    material_names: List[str], 
    constructions_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Returns constructions matching specified materials, handling multi-material requests
    using strict intersection (AND logic) by evaluating independent search sub-sets.
    """
    if not material_names:
        return constructions_data

    # 1. Normalize strings: Split by comma if the inputs were compressed into a single string element
    normalized_targets: List[str] = []
    for item in material_names:
        if isinstance(item, str) and "," in item:
            normalized_targets.extend([m.strip() for m in item.split(",") if m.strip()])
        elif isinstance(item, str) and item.strip():
            normalized_targets.append(item.strip())

    if not normalized_targets:
        return constructions_data

    # Helper lambda to run a clean search for a single independent text string
    def search_single_material(target_text: str, dataset: Dict[str, Any]) -> Dict[str, Any]:
        target_lower = target_text.lower()
        sub_matches = {}
        for const_id, info in dataset.items():
            layers = info.get("layers", [])
            # Substring matching ensures 'mineral wool' flags 'lana mineral' or similar tokens safely
            found_materials = [str(layer.get("material")).strip().lower() for layer in layers if layer.get("material")]
            
            if any(target_lower in mat or mat in target_lower for mat in found_materials):
                sub_matches[const_id] = info
        return sub_matches

    # 2. Extract matches for the first required item to establish our baseline subset
    final_matches = search_single_material(normalized_targets[0], constructions_data)

    # 3. Iteratively intersect subsequent requirements against the shrinking dataset
    for remaining_target in normalized_targets[1:]:
        # If any prior step resulted in an empty result pool, we break early
        if not final_matches:
            break
            
        # Get matching items for the next item in isolation
        next_set_matches = search_single_material(remaining_target, constructions_data)
        
        # Intersect via the module's set operation tool
        final_matches = perform_set_operation(
            set_a=final_matches, 
            set_b=next_set_matches, 
            operation="intersection"
        )

    return final_matches

def filter_constructions_by_thickness(
    constructions: Dict[str, Any], 
    target_thickness: float, 
    mode: str = "total", 
    tolerance: float = 0.005, 
    include_eat: bool = True, 
    include_nan: bool = False
) -> Dict[str, Any]:
    """
    Filters constructions based on thickness, either per individual layer or by the total sum.
    
    Args:
        constructions: Dictionary of constructions to filter.
        target_thickness: The numerical value (usually in meters) to match.
        mode: "total" to check sum of all layers, "individual" to see if any one layer matches.
        tolerance: Allowed margin of error for the match.
        include_eat: If True, treats "eAT" (estimated Air Thickness) as a valid match.
        include_nan: If True, includes layers with undefined thicknesses.
    """
    valid_matches = {}

    def matches_target(val: Union[float, str, None]) -> bool:
        if val == "eAT": return include_eat
        if _is_nan_value(val): return include_nan
        try:
            return abs(float(val) - target_thickness) <= tolerance
        except (ValueError, TypeError):
            return False

    for const_id, data in constructions.items():
        layers = data.get("layers", [])

        if mode == "individual":
            if any(matches_target(l.get("thickness")) for l in layers):
                valid_matches[const_id] = data
        else:
            # Mode: Total Thickness Calculation
            total_sum = 0.0
            has_eat, has_nan = False, False
            
            for layer in layers:
                t = layer.get("thickness")
                if t == "eAT": has_eat = True
                elif _is_nan_value(t): has_nan = True
                else:
                    try: total_sum += float(t)
                    except: has_nan = True
            
            # Logic gate for air/missing values
            if not include_eat and has_eat: continue
            if not include_nan and has_nan: continue
            
            # Check if total matches or if it's an eAT-based match
            if abs(total_sum - target_thickness) <= tolerance or (include_eat and has_eat and total_sum < target_thickness):
                valid_matches[const_id] = data
                
    return valid_matches

def filter_by_material_thickness(
    constructions: Dict[str, Any],
    target_material: str,
    target_thickness: float,
    tolerance: float = 0.005,
    include_eat: bool = False,
    include_nan: bool = False
) -> Dict[str, Any]:
    """
    Advanced filter that looks for a specific material and thickness paired in the SAME layer.
    
    Example: Find all walls that have exactly 0.04m of "lana mineral".
    """
    valid_matches = {}
    search_material = target_material.strip().lower()

    for const_id, data in constructions.items():
        layers = data.get("layers", [])

        def layer_matches(layer: Dict[str, Any]) -> bool:
            # Check Material
            mat_name = str(layer.get("material", "")).strip().lower()
            if search_material not in mat_name:
                return False
            
            # Check Thickness
            thickness = layer.get("thickness")
            if thickness == "eAT": return include_eat
            if _is_nan_value(thickness): return include_nan
            
            try:
                return abs(float(thickness) - target_thickness) <= tolerance
            except (ValueError, TypeError):
                return False

        if any(layer_matches(l) for l in layers):
            valid_matches[const_id] = data

    return valid_matches

def filter_constructions_by_layer_count(
    constructions: Dict[str, Any], 
    count: int, 
    operator: str = "=="
) -> Dict[str, Any]:
    """
    Filters constructions based on the complexity (number of layers).
    
    Args:
        constructions: Dictionary of constructions.
        count: The number of layers to compare against.
        operator: Comparison string (e.g., ">", "<=", "==").
    """
    operators: Dict[str, Callable[[int, int], bool]] = {
        "==": lambda x, y: x == y,
        ">":  lambda x, y: x > y,
        "<":  lambda x, y: x < y,
        ">=": lambda x, y: x >= y,
        "<=": lambda x, y: x <= y,
    }
    
    if operator not in operators:
        raise ValueError(f"Operator '{operator}' not supported.")

    valid_matches = {}
    compare_func = operators[operator]

    for const_id, data in constructions.items():
        layer_count = len(data.get("layers", []))
        if compare_func(layer_count, count):
            valid_matches[const_id] = data
            
    return valid_matches

def find_by_id(
    constructions: Dict[str, Any],
    ids: List[str],
) -> Dict[str, Any]:
    """Returns constructions whose IDs match any in the given list (case-insensitive)."""
    search = {i.strip().upper() for i in ids if i}
    return {k: v for k, v in constructions.items() if k.strip().upper() in search}

def find_by_category(
    constructions: Dict[str, Any],
    category: str,
) -> Dict[str, Any]:
    """Returns all constructions belonging to the given category (case-insensitive)."""
    target = category.strip().lower()
    return {k: v for k, v in constructions.items() if v.get("category", "").lower() == target}

def perform_set_operation(
    set_a: Dict[str, Any], 
    set_b: Dict[str, Any], 
    operation: str = "intersection"
) -> Dict[str, Any]:
    """
    Merges or compares two sets of constructions using standard set theory.
    
    Operations:
        'intersection': Items in both sets.
        'union': All unique items from both sets.
        'difference': Items in set_a but NOT in set_b.
        'symmetric_difference': Items in either A or B, but not both.
    """
    keys_a = set(set_a.keys())
    keys_b = set(set_b.keys())
    
    # Use built-in set methods for speed
    ops = {
        "intersection": keys_a.intersection,
        "union": keys_a.union,
        "difference": keys_a.difference,
        "symmetric_difference": keys_a.symmetric_difference
    }
    
    if operation not in ops:
        raise ValueError(f"Operation '{operation}' not recognized.")

    result_keys = ops[operation](keys_b)

    # Combine sources to retrieve data for any selected key
    combined_source = {**set_a, **set_b}
    return {k: combined_source[k] for k in result_keys if k in combined_source}
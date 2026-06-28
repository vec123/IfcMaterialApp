"""
IFC data extractor — produces 4 JSON files per IFC element type:
  1. element_to_typology.json      — element name → typology (layer set) name
  2. typologies_as_layers.json     — typology name → ordered list of material layers
  3. unique_materials.json         — unique material names
  4. elements_without_typology.json — elements with no material layer set

Run:
    python extract_ifc_data.py <ifc_path> [--output-dir extracted_ifc_data]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from src.ifc.element_types import ELEMENT_TYPES
import ifcopenshell



# ---------------------------------------------------------------------------
# Low-level IFC helpers
# ---------------------------------------------------------------------------

def _element_label(element) -> str:
    """Return a unique, human-readable key for an element."""
    name = element.Name if element.Name else "Unnamed"
    return f"{name}_{element.id()}"


def _extract_typology(element) -> Tuple[Optional[str], Optional[List[Dict]]]:
    """
    Return (typology_name, layers) for an element.

    typology_name — the LayerSetName from IfcMaterialLayerSet(Usage), or the
                    material name for single-material elements.
    layers        — list of {"material": str, "thickness_m": float|None,
                             "layer_name": str|None} ordered from exterior to
                    interior (as stored in IFC).

    Returns (None, None) when the element has no material association.
    """
    if not hasattr(element, "HasAssociations"):
        return None, None

    for rel in element.HasAssociations:
        if not rel.is_a("IfcRelAssociatesMaterial"):
            continue

        mat = rel.RelatingMaterial

        # -------------------------------------------------------------------
        if mat.is_a("IfcMaterialLayerSetUsage"):
            layer_set = mat.ForLayerSet
            name = (layer_set.LayerSetName or "").strip() or f"LayerSet_{layer_set.id()}"
            layers = [
                {
                    "material": (getattr(lyr.Material, "Name", None) if lyr.Material else None),
                    "thickness_m": getattr(lyr, "LayerThickness", None),
                    "layer_name": getattr(lyr, "Name", None),
                }
                for lyr in layer_set.MaterialLayers
            ]
            return name, layers

        # -------------------------------------------------------------------
        if mat.is_a("IfcMaterialLayerSet"):
            name = (mat.LayerSetName or "").strip() or f"LayerSet_{mat.id()}"
            layers = [
                {
                    "material": (getattr(lyr.Material, "Name", None) if lyr.Material else None),
                    "thickness_m": getattr(lyr, "LayerThickness", None),
                    "layer_name": getattr(lyr, "Name", None),
                }
                for lyr in mat.MaterialLayers
            ]
            return name, layers

        # -------------------------------------------------------------------
        if mat.is_a("IfcMaterialLayer"):
            mat_name = getattr(mat.Material, "Name", None) if mat.Material else None
            layer_name = getattr(mat, "Name", None)
            name = (layer_name or mat_name or f"Layer_{mat.id()}").strip()
            layers = [
                {
                    "material": mat_name,
                    "thickness_m": getattr(mat, "LayerThickness", None),
                    "layer_name": layer_name,
                }
            ]
            return name, layers

        # -------------------------------------------------------------------
        if mat.is_a("IfcMaterial"):
            return mat.Name, [{"material": mat.Name, "thickness_m": None, "layer_name": None}]

        # -------------------------------------------------------------------
        if mat.is_a("IfcMaterialList"):
            names = [m.Name for m in mat.Materials if m.Name]
            combined = " | ".join(names) or f"MaterialList_{mat.id()}"
            layers = [{"material": n, "thickness_m": None, "layer_name": None} for n in names]
            return combined, layers

    return None, None


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_ifc_data(cfg) -> Dict:
    """
    Load *ifc_path*, extract typology data and write 4 JSON files to *output_dir*.
    Returns a summary dict with counts.
    """
    ifc_path = cfg["paths"]["ifc_extracted_file"]
    output_dir = cfg["paths"]["ifc_extracted_dir"]
    ifc = ifcopenshell.open(ifc_path)

    # Containers keyed by IFC element type
    element_to_typology: Dict[str, Dict[str, str]] = {}
    typologies_as_layers: Dict[str, Dict[str, List]] = {}
    unique_materials: Dict[str, List[str]] = {}
    no_typology: Dict[str, List[str]] = {}

    for ifc_type in ELEMENT_TYPES:
        elements = list(ifc.by_type(ifc_type))
        if not elements:
            continue

        elem_map: Dict[str, str] = {}
        typo_map: Dict[str, List] = {}
        type_mats: Set[str] = set()
        no_typo: List[str] = []

        for element in elements:
            label = _element_label(element)
            try:
                typo_name, layers = _extract_typology(element)
            except Exception:
                typo_name, layers = None, None

            if typo_name is None:
                no_typo.append(label)
            else:
                elem_map[label] = typo_name
                if layers:
                    typo_map[typo_name] = layers
                    for lyr in layers:
                        if lyr["material"]:
                            type_mats.add(lyr["material"])

        element_to_typology[ifc_type] = elem_map
        typologies_as_layers[ifc_type] = typo_map
        unique_materials[ifc_type] = sorted(type_mats)
        no_typology[ifc_type] = no_typo

    # Write JSONs
    os.makedirs(output_dir, exist_ok=True)

    def _save(data, filename):
            path = filename
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Saved: {path}")

    _save(element_to_typology, cfg["paths"]["ifc_extracted_element_to_typology"])
    _save(typologies_as_layers, cfg["paths"]["ifc_extracted_typology_layers"])
    _save(unique_materials, cfg["paths"]["ifc_extracted_unique_materials"])
    _save(no_typology, cfg["paths"]["ifc_extracted_element_missing_typology"])

    summary = {
        "ifc_types_found": list(element_to_typology.keys()),
        "elements_with_typology": sum(len(v) for v in element_to_typology.values()),
        "elements_without_typology": sum(len(v) for v in no_typology.values()),
        "unique_typologies": sum(len(v) for v in typologies_as_layers.values()),
        "unique_materials_total": len({m for mats in unique_materials.values() for m in mats}),
    }
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract IFC typology/material data to JSON files."
    )
    parser.add_argument("ifc_path", help="Path to the .ifc file")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "extracted_ifc_data"),
        help="Directory where the 4 JSON files will be written (default: ./extracted_ifc_data)",
    )
    args = parser.parse_args()

    ifc_path = args.ifc_path
    if not Path(ifc_path).exists():
        print(f"Error: IFC file not found: {ifc_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading: {ifc_path}")
    summary = extract_ifc_data(ifc_path, args.output_dir)

    print("\nExtraction complete:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

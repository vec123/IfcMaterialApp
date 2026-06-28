import os
import sys
import json
import ifcopenshell
from src.utils.load_conf import build_cfg  # Import your unified config compiler

cfg = build_cfg()

def update_ifc_with_new_materials(input_ifc_path, output_ifc_path, material_mappings):
    model = ifcopenshell.open(input_ifc_path)
    project = model.by_type("IfcProject")[0]
    
    # 1. Get all materials in the model
    materials = model.by_type("IfcMaterial")
    
    for mat in materials:
        # Check if the material exists in the new mapping structure
        if mat.Name in material_mappings:
            data = material_mappings[mat.Name]
            
            # Check if there are matches and grab the first one
            if "matches" in data and len(data["matches"]) > 0:
                best_match = data["matches"][0]
                props = best_match["metadata"]
                
                # 2. Add properties to the material
                pset = model.createIfcPropertySet(
                    ifcopenshell.guid.new(),
                    project.OwnerHistory,
                    "Pset_MaterialThermalProperties",
                    None,
                    [
                        model.createIfcPropertySingleValue("ThermalConductivity", None, 
                            model.createIfcThermalConductivityMeasure(props["thermal_conductivity"]), None),
                        model.createIfcPropertySingleValue("Density", None, 
                            model.createIfcMassDensityMeasure(props["density"]), None),
                        model.createIfcPropertySingleValue("SpecificHeatCapacity", None, 
                            model.createIfcSpecificHeatCapacityMeasure(props["specific_heat"]), None)
                    ]
                )
                
                # Link the PropertySet to the Material
                model.createIfcRelDefinesByProperties(
                    ifcopenshell.guid.new(),
                    project.OwnerHistory,
                    None,
                    None,
                    [mat],
                    pset
                )
                print(f"Updated material '{mat.Name}' with match: '{best_match['canonical_material_id']}'")
            else:
                print(f"No matches found for material: {mat.Name}")

    model.write(output_ifc_path)
    print(f"Saved new IFC to: {output_ifc_path}")

def load_json(file_path: str):
    """
    Safely loads a JSON file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON configuration file not found at: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_ifc(file_path: str):
    """
    Verifies and returns the path to the IFC file. 
    Note: ifcopenshell.open() itself handles the loading, 
    so we return the validated path here.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"IFC file not found at: {file_path}")
    
    return file_path

if __name__ == "__main__":
    # Example Input Data (Add your full IFC typologies here)
    ifc_file = load_ifc(cfg["paths"]["ifc_extracted_file"])
    material_mappings = load_json(cfg["paths"]["tmp_match_matrix"])

    element_to_typology = load_json(cfg["paths"]["ifc_extracted_element_to_typology"])
    typologies_to_matieral_layers = load_json(cfg["paths"]["typologies_as_layers"])
    # Run update
    new_data = update_ifc_with_new_materials(ifc_file, "test.ifc", material_mappings)

    # Output result
    print(json.dumps(new_data, indent=2, ensure_ascii=False))
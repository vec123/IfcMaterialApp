import ifcopenshell
import json
from src.utils.load_conf import build_cfg
from src.ifc.element_types import ELEMENT_TYPES

def get_metadata(mapping_data):
    """
    Handles both JSON formats:
    1. With 'matches' list: uses the first match.
    2. Direct mapping: returns the dict directly.
    """
    if isinstance(mapping_data, dict) and "matches" in mapping_data:
        if mapping_data["matches"] and isinstance(mapping_data["matches"], list):
            return mapping_data["matches"][0].get("metadata", {})
    elif isinstance(mapping_data, dict) and "metadata" in mapping_data:
        return mapping_data.get("metadata", {})
    return {}

def update_ifc_with_assembly_properties(input_ifc_path, output_ifc_path, material_mappings, element_to_typology, typologies_layers):
    model = ifcopenshell.open(input_ifc_path)
    project = model.by_type("IfcProject")[0]
    
    ifc_types = ELEMENT_TYPES
    updated_count = 0

    for ifc_type in ifc_types:
        elements = model.by_type(ifc_type)
        for elem in elements:
            name = elem.Name if elem.Name else "Unnamed"
            elem_label = f"{name}_{elem.id()}"
            
            typology_name = element_to_typology.get(ifc_type, {}).get(elem_label)
            if not typology_name: continue
                
            layers = typologies_layers.get(ifc_type, {}).get(typology_name, [])
            if not layers: continue

            pset_name = f"Pset_ThermalAssembly_{typology_name.replace(':', '_')}"
            properties = []

            for layer in layers:
                mat_id = layer.get("material")
                mapping_data = material_mappings.get(mat_id)
                
                # Use helper to resolve the metadata regardless of JSON format
                props = get_metadata(mapping_data)
                
                if props:
                    cond = props.get("thermal_conductivity", 0)
                    heat = props.get("specific_heat", 0)
                    dens = props.get("density", 0)
                    
                    if cond > 0:
                        properties.append(model.createIfcPropertySingleValue(
                            f"{mat_id}_Conductivity", None, model.createIfcThermalConductivityMeasure(cond), None))
                    if heat > 0:
                        properties.append(model.createIfcPropertySingleValue(
                            f"{mat_id}_SpecificHeat", None, model.createIfcSpecificHeatCapacityMeasure(heat), None))
                    if dens > 0:
                        properties.append(model.createIfcPropertySingleValue(
                            f"{mat_id}_Density", None, model.createIfcMassDensityMeasure(dens), None))

            if properties:
                pset = model.createIfcPropertySet(
                    ifcopenshell.guid.new(), project.OwnerHistory, pset_name, None, properties
                )
                model.createIfcRelDefinesByProperties(
                    ifcopenshell.guid.new(), project.OwnerHistory, None, None, [elem], pset
                )
                updated_count += 1
                print(f"Success: Updated {elem_label} ({len(properties)} props)")

    model.write(output_ifc_path)
    print(f"\nProcessing complete. Updated {updated_count} elements. Saved to: {output_ifc_path}")

if __name__ == "__main__":
    cfg = build_cfg()
    update_ifc_with_assembly_properties(
        cfg["paths"]["ifc_extracted_file"], 
        "test_walls.ifc", 
        json.load(open(cfg["paths"]["match_matrix"], "r", encoding="utf-8")),
        json.load(open(cfg["paths"]["ifc_extracted_element_to_typology"], "r", encoding="utf-8")),
        json.load(open(cfg["paths"]["ifc_extracted_typology_layers"], "r", encoding="utf-8"))
    )
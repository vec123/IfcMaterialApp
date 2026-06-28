import src.query.filters as cl

class ExecutionAgent:
    def __init__(self, database: dict):
        self.database = database
        # Align keys with your IntentAgent values
        self.tool_map = {
            "find_by_id":                cl.find_by_id,
            "find_by_category":          cl.find_by_category,
            "find_by_layer_type":        cl.get_constructions_by_materials,
            "filter_by_thickness":       cl.filter_constructions_by_thickness,
            "filter_by_number_of_layers": cl.filter_constructions_by_layer_count,
            "filter_by_specific_layer":  cl.filter_by_material_thickness,
        }

    def execute(self, intent: str, mapped_params: dict) -> list:
        if intent not in self.tool_map:
            print(f"⚠️ No execution tool for intent: {intent}")
            return []

        try:
            if intent == "find_by_id":
                return self.tool_map[intent](
                    constructions=self.database,
                    ids=mapped_params.get("ids", []),
                )

            elif intent == "find_by_category":
                return self.tool_map[intent](
                    constructions=self.database,
                    category=mapped_params.get("category", ""),
                )

            elif intent == "find_by_layer_type":
                return self.tool_map[intent](
                    material_names=mapped_params.get("materials", []),
                    constructions_data=self.database
                )

            elif intent == "filter_by_thickness":
                return self.tool_map[intent](
                    constructions=self.database,
                    target_thickness=mapped_params.get("thickness", 0.0),
                    mode=mapped_params.get("mode", "total"),
                    tolerance=mapped_params.get("tolerance", 0.005)
                )

            elif intent == "filter_by_specific_layer":
                # Note: SpecificLayerInputAgent uses 'material' (singular)
                # while map_params_materials likely converts it to a list/mapped string
                mat = mapped_params.get("material") 
                if isinstance(mat, list): mat = mat[0] if mat else ""
                
                return self.tool_map[intent](
                    constructions=self.database,
                    target_material=mat,
                    target_thickness=mapped_params.get("target_thickness", 0.0),
                    tolerance=mapped_params.get("tolerance", 0.005)
                )

            elif intent == "filter_by_number_of_layers":
                return self.tool_map[intent](
                    constructions=self.database,
                    count=mapped_params.get("layer_count", 0),
                    operator=mapped_params.get("operator", "==")
                )

        except Exception as e:
            print(f"❌ Execution Error for {intent}: {e}")
            return []
        
        return []

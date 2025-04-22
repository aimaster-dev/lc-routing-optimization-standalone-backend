import json
from collections import defaultdict
from bson import ObjectId
from datetime import datetime
import numpy as np

def read_json_file(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def smart_aggregate_by_obj(docs):
    grouped = defaultdict(list)

    # Group documents by OBJ.$oid
    for doc in docs:
        obj_info = doc.get("OBJ")
        if not obj_info or "$oid" not in obj_info:
            raise ValueError("Missing OBJ.$oid in document.")
        obj_id = obj_info["$oid"]
        grouped[obj_id].append(doc)

    final_results = {}

    for obj_id, group_docs in grouped.items():
        merged_obj = {}
        field_values = defaultdict(list)

        # Collect all values for each field
        for doc in group_docs:
            for key, value in doc.items():
                field_values[key].append(value)

        # Analyze fields
        for field, values in field_values.items():
            # If all values are the same, keep as scalar
            unique_values = set(json.dumps(v, default=str) for v in values)
            if len(unique_values) == 1:
                # Use the first value as scalar
                merged_obj[field] = values[0]
            else:
                # If values are numbers, make it a numpy array
                if all(isinstance(v, (int, float)) for v in values if v is not None):
                    merged_obj[field] = np.array(values)
                else:
                    # Otherwise, keep as list
                    merged_obj[field] = values

        # Reconvert fields if needed (e.g., ObjectId, datetime)
        for key, value in merged_obj.items():
            if isinstance(value, dict) and "$oid" in value:
                merged_obj[key] = ObjectId(value["$oid"])
            elif isinstance(value, dict) and "$date" in value:
                merged_obj[key] = datetime.fromisoformat(value["$date"])

            # Handle nested fields inside dict (like nonce)
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict) and "$oid" in subvalue:
                        value[subkey] = ObjectId(subvalue["$oid"])
                    if isinstance(subvalue, dict) and "$date" in subvalue:
                        value[subkey] = datetime.fromisoformat(subvalue["$date"])

        final_results[obj_id] = merged_obj

    return final_results

# Example usage:
input_json_path = "your_input_file.json"  # ← replace with your input
documents = read_json_file(input_json_path)

# Aggregate
result = smart_aggregate_by_obj(documents)

# Display result
for obj_id, merged_data in result.items():
    print(f"\nOBJ {obj_id}:")
    for k, v in merged_data.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: array of length {len(v)}")
        else:
            print(f"  {k}: {v}")

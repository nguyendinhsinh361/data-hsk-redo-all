import json
import os

def fill_missing_ids(input_file, output_file=None):
    """
    Read a JSON file and fill objects with missing fields using values from the previous object.
    Fields that will be filled: 'id', 'value', 'kind', 'check_admin'
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str, optional): Path to save the output JSON file. If None, will use the input filename with '_filled' suffix.
    
    Returns:
        str: Path to the output file
    """
    # Set default output file if not provided
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_filled.json"
    
    # Read the JSON file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return None
    
    # Check if data is a list
    if not isinstance(data, list):
        print("Error: JSON data is not a list of objects")
        return None
    
    # Fields to fill
    fields_to_fill = ['id', 'value', 'kind', 'check_admin']
    
    # Current values for each field
    current_values = {field: None for field in fields_to_fill}
    filled_counts = {field: 0 for field in fields_to_fill}
    
    # Fill missing fields
    for item in data:
        for field in fields_to_fill:
            if field in item and item[field]:
                # Update current value when we find a valid one
                current_values[field] = item[field]
            elif current_values[field] is not None:
                # Fill missing field with the current value
                item[field] = current_values[field]
                filled_counts[field] += 1
    
    # Write the updated data to the output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        # Print summary
        print("Successfully filled missing fields:")
        for field, count in filled_counts.items():
            print(f"  - {field}: {count} items")
        print(f"Output saved to: {output_file}")
        return output_file
    except Exception as e:
        print(f"Error writing output file: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    input_file = "extract_n8n.json"
    fill_missing_ids(input_file) 
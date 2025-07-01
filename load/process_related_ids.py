import json
import os
import sys

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def remove_largest_id_and_flatten_arrays(input_file, output_file=None):
    """
    Read a JSON file with nested arrays, remove the element with the largest ID
    from each nested array, and then flatten the arrays for each kind/key.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str, optional): Path to save the output JSON file. 
                                    If None, will use input_file with '_modified' suffix
    
    Returns:
        dict: The modified and flattened data
    """
    try:
        # Read the JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Process each kind/key in the data
        result_data = {}
        for kind, arrays in data.items():
            # First, remove largest ID from each nested array
            processed_arrays = []
            for nested_array in arrays:
                # Check if this is an array with at least one element
                if isinstance(nested_array, list) and len(nested_array) > 1:
                    # Find the element with the largest ID
                    max_id = max(nested_array)
                    # Remove the element with the largest ID
                    processed_arrays.append([id_val for id_val in nested_array if id_val != max_id])
                else:
                    processed_arrays.append(nested_array)
            
            # Now flatten the arrays for this kind
            flattened_array = []
            for nested_array in processed_arrays:
                flattened_array.extend(nested_array)
            
            # Store the flattened array for this kind
            result_data[kind] = flattened_array
            
        
        result_flattened = []
        for kind, array in result_data.items():
            result_flattened.extend(array)
        
        # Determine output file path if not provided
        if output_file is None:
            base_name, ext = os.path.splitext(input_file)
            output_file = f"{base_name}_modified{ext}"
        
        # Write the modified and flattened data to the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)
            
        with open('related_ids_modified_flattened.json', 'w', encoding='utf-8') as f:
            json.dump(result_flattened, f, indent=4, ensure_ascii=False)
        
        print(f"Successfully processed {input_file} and saved to {output_file}")
        return result_data
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return None
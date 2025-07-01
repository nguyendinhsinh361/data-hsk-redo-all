#!/usr/bin/env python3
from process_related_ids import remove_largest_id_and_flatten_arrays
import os

def main():
    # Get the absolute path to the related_ids_analysis.json file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    input_file = os.path.join(project_root, "transform", "related_ids_analysis.json")
    output_file = os.path.join(project_root, "load", "related_ids_analysis_modified.json")
    
    print(f"Processing file: {input_file}")
    print(f"Output will be saved to: {output_file}")
    
    # Process the file
    result = remove_largest_id_and_flatten_arrays(input_file, output_file)
    
    if result:
        print("Processing completed successfully.")
    else:
        print("Processing failed.")

if __name__ == "__main__":
    main() 
from fill_missing_ids import fill_missing_ids
import os

def main():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the input JSON file
    input_file = os.path.join(current_dir, "extract_n8n.json")
    
    # Call the function to fill missing IDs
    output_file = fill_missing_ids(input_file)
    
    if output_file:
        print(f"Processing complete. Output saved to: {output_file}")
    else:
        print("Error occurred during processing.")

if __name__ == "__main__":
    main() 
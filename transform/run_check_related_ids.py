import os
from check_related_ids import analyze_related_ids

def main():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the input JSON file (extract_n8n_filled.json)
    input_file = os.path.join(os.path.dirname(current_dir), "extract", "extract_n8n_filled.json")
    
    # Path to the output JSON file
    output_dir = os.path.join(os.path.dirname(current_dir), "transform")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "related_ids_analysis.json")
    
    # Run the analysis
    result_file = analyze_related_ids(input_file, output_file)
    
    if result_file:
        print(f"Analysis complete. Results saved to: {result_file}")
    else:
        print("Error occurred during analysis.")

if __name__ == "__main__":
    main() 
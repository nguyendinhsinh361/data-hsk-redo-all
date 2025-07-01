import json
import os
import sys
from collections import defaultdict

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.common import get_raw_data, save_data_to_json

def find_related_ids_by_kind(data):
    """
    For each kind, find objects where either 'id' or 'Giống với id' matches
    with any other object's 'id' or 'Giống với id' in the same kind.
    
    Args:
        data (list): List of objects from the JSON file
        
    Returns:
        dict: Dictionary with kind as keys and lists of related ID groups as values
    """
    # Group data by kind
    data_by_kind = defaultdict(list)
    for item in data:
        if 'kind' in item:
            data_by_kind[item['kind']].append(item)
    
    results = {}
    
    # Process each kind separately
    for kind, items in data_by_kind.items():
        print(f"Processing kind: {kind} with {len(items)} items")
        
        # Create a map of all IDs (both 'id' and 'Giống với id') to their objects
        id_map = {}
        for item in items:
            if 'id' in item and item['id']:
                if item['id'] not in id_map:
                    id_map[item['id']] = []
                id_map[item['id']].append(item)
            
            if 'Giống với id' in item and item['Giống với id']:
                if item['Giống với id'] not in id_map:
                    id_map[item['Giống với id']] = []
                id_map[item['Giống với id']].append(item)
        
        # Find related ID groups
        processed_ids = set()
        related_groups = []
        
        for id_val, items in id_map.items():
            if id_val in processed_ids:
                continue
            
            # Start a new related group
            current_group = set()
            queue = [id_val]
            
            # Process all connected IDs
            while queue:
                current_id = queue.pop(0)
                if current_id in processed_ids:
                    continue
                
                processed_ids.add(current_id)
                current_group.add(current_id)
                
                # Add all objects with this ID
                for item in id_map.get(current_id, []):
                    # Add both 'id' and 'Giống với id' to the queue
                    if 'id' in item and item['id'] and item['id'] not in processed_ids:
                        queue.append(item['id'])
                    
                    if 'Giống với id' in item and item['Giống với id'] and item['Giống với id'] not in processed_ids:
                        queue.append(item['Giống với id'])
            
            # Only add groups with more than one ID
            if len(current_group) > 1:
                related_groups.append(list(current_group))
        
        # Store results for this kind
        results[kind] = related_groups
    
    return results

def analyze_related_ids(input_file, output_file=None):
    """
    Analyze the input JSON file to find related IDs by kind and save the results.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str, optional): Path to save the output JSON file. If None, will use the input filename with '_related_ids' suffix.
    
    Returns:
        str: Path to the output file
    """
    # Set default output file if not provided
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_related_ids.json"
    
    # Read the JSON file
    try:
        data = get_raw_data(input_file)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return None
    
    # Find related IDs by kind
    results = find_related_ids_by_kind(data)
    
    # Generate summary statistics
    summary = {
        "total_kinds": len(results),
        "kinds_with_related_ids": sum(1 for groups in results.values() if groups),
        "total_related_groups": sum(len(groups) for groups in results.values()),
        "kind_statistics": {
            kind: {
                "related_groups": len(groups),
                "total_related_ids": sum(len(group) for group in groups)
            } for kind, groups in results.items()
        }
    }
    
    # Combine results and summary
    # output_data = {
    #     "summary": summary,
    #     "related_ids_by_kind": results
    # }
    output_data = results
    
    # Write the results to the output file
    try:
        save_data_to_json(output_data, output_file)
        print(f"Analysis complete. Results saved to: {output_file}")
        
        # Print summary
        print("\nSummary:")
        print(f"Total kinds analyzed: {summary['total_kinds']}")
        print(f"Kinds with related IDs: {summary['kinds_with_related_ids']}")
        print(f"Total related ID groups: {summary['total_related_groups']}")
        
        # Print kinds with the most related groups
        if summary['kinds_with_related_ids'] > 0:
            print("\nTop kinds with most related ID groups:")
            top_kinds = sorted(
                [(kind, stats["related_groups"]) for kind, stats in summary["kind_statistics"].items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            for kind, count in top_kinds:
                if count > 0:
                    print(f"  - Kind {kind}: {count} related groups")
        
        return output_file
    except Exception as e:
        print(f"Error writing output file: {e}")
        return None

if __name__ == "__main__":
    # Get the path to the input file
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # Default to the filled extract file
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        input_file = os.path.join(current_dir, "extract", "extract_n8n_filled.json")
    
    # Run the analysis
    analyze_related_ids(input_file) 
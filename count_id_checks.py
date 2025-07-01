import pandas as pd
import os
from pathlib import Path

def count_id_checks_by_sheet(excel_file_path):
    """
    Count the number of ID checks in each sheet of the Excel file.
    
    Args:
        excel_file_path (str): Path to the Excel file
        
    Returns:
        dict: Dictionary with sheet names as keys and ID counts as values
    """
    # Check if file exists
    if not os.path.exists(excel_file_path):
        print(f"File not found: {excel_file_path}")
        return None, None
    
    try:
        # Read all sheets in the Excel file
        excel_file = pd.ExcelFile(excel_file_path)
        sheet_names = excel_file.sheet_names
        
        # Initialize result dictionary
        results = {}
        details = {}
        
        # Process each sheet
        for sheet_name in sheet_names:
            # Read the sheet
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            # Look for ID columns (common names in Vietnamese and English)
            possible_id_columns = ['ID', 'id', 'Mã', 'mã', 'Ma', 'ma', 
                                  'Mã câu hỏi', 'ID câu hỏi', 'Mã CH', 
                                  'Question ID', 'QuestionID', 'Mã cau hoi']
            
            id_column = None
            # Try to find ID column from common names
            for col_name in possible_id_columns:
                if col_name in df.columns:
                    id_column = col_name
                    break
            
            # If no direct match, look for columns containing 'id' or 'mã'
            if id_column is None:
                for col in df.columns:
                    col_lower = str(col).lower()
                    if 'id' in col_lower or 'ma' in col_lower or 'mã' in col_lower:
                        id_column = col
                        break
            
            # Count IDs
            if id_column is not None:
                # Count non-null values in the ID column
                id_count = df[id_column].count()
                # Get list of unique IDs for reference
                unique_ids = df[id_column].dropna().unique()
                id_column_name = id_column
            else:
                # If still no ID column found, use the first column as a fallback
                id_count = df.iloc[:, 0].count() if not df.empty else 0
                unique_ids = df.iloc[:, 0].dropna().unique() if not df.empty else []
                id_column_name = df.columns[0] if not df.empty else "Unknown"
            
            # Store results
            results[sheet_name] = id_count
            details[sheet_name] = {
                'column_name': id_column_name,
                'total_ids': id_count,
                'unique_ids': len(unique_ids),
                'sample_ids': list(unique_ids[:5]) if len(unique_ids) > 0 else []
            }
        
        return results, details
    
    except Exception as e:
        print(f"Error processing Excel file: {e}")
        return None, None

if __name__ == "__main__":
    # Try multiple possible paths for the Excel file
    possible_paths = [
        # Direct path
        "excel/Check Dữ Liệu Câu Hỏi Trùng Lặp Các Dạng HSK.xlsx",
        # Absolute path based on current script
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                    "excel", "Check Dữ Liệu Câu Hỏi Trùng Lặp Các Dạng HSK.xlsx"),
        # Fixed absolute path from user_info
        "/Users/nguyendinhsinh/Documents/data/data-hsk-redo-all/excel/Check Dữ Liệu Câu Hỏi Trùng Lặp Các Dạng HSK.xlsx"
    ]
    
    # Try each path
    excel_file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            excel_file_path = path
            break
    
    if excel_file_path is None:
        print("Excel file not found in any of the expected locations.")
        print("Please provide the full path to the Excel file:")
        excel_file_path = input("> ")
    
    # Count ID checks
    print(f"Using Excel file: {excel_file_path}")
    sheet_counts, details = count_id_checks_by_sheet(excel_file_path)
    
    # Display results
    if sheet_counts:
        print("\n===== NUMBER OF ID CHECKS IN EACH SHEET =====")
        for sheet_name, count in sheet_counts.items():
            print(f"{sheet_name}: {count} IDs")
        
        print("\n===== DETAILED INFORMATION =====")
        for sheet_name, info in details.items():
            print(f"\nSheet: {sheet_name}")
            print(f"  ID Column: {info['column_name']}")
            print(f"  Total IDs: {info['total_ids']}")
            print(f"  Unique IDs: {info['unique_ids']}")
            print(f"  Sample IDs: {', '.join(str(id) for id in info['sample_ids'][:3])}" 
                  if info['sample_ids'] else "  Sample IDs: None")
    else:
        print("Failed to count ID checks.") 
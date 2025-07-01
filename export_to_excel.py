import os
import json
import pandas as pd
from tqdm import tqdm
import argparse
import numpy as np

def json_to_excel(input_folder, output_file=None, sort_order=None):
    """
    Convert all JSON files in the input folder to an Excel file with each JSON file as a separate sheet.
    
    Args:
        input_folder (str): Path to the folder containing JSON files
        output_file (str, optional): Path to the output Excel file. If None, it will use the folder name.
        sort_order (list, optional): List defining the order of sheets. If None, alphabetical order is used.
    """
    # If output_file is not specified, use the folder name
    if output_file is None:
        folder_name = os.path.basename(input_folder)
        output_file = f"excel/Check Dữ Liệu Câu Hỏi Trùng Lặp Các Dạng HSK.xlsx"
    
    # Ensure the excel directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Define the standard sort order for kind values if not provided
    if sort_order is None:
        sort_order = [
            "110001", "110002", "110003", "110004", "120001", "120002", "120003", "120004", 
            "210001", "210002", "210003", "210004", "220001", "220002", "220003", "220004", 
            "310001", "310002", "310003", "310004", "320001", "320002", "320003", "330001", 
            "330002", "410001", "410002", "410003_1", "410003_2", "420001", "420002", 
            "420003_1", "420003_2", "430001", "430002", "510001", "510002_1", "510002_2", 
            "520001", "520002", "520003", "530001", "530002", "530003", "610001", "610002", 
            "610003", "620001", "620002", "620003", "620004", "630001"
        ]
    
    # Get all JSON files in the input folder
    all_json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    
    # Sort the JSON files based on the predefined order
    def get_sort_key(filename):
        basename = os.path.splitext(filename)[0]
        try:
            # Find the position in the sort_order list
            return sort_order.index(basename)
        except ValueError:
            # If not in the list, put at the end
            return len(sort_order) + all_json_files.index(filename)
    
    # Sort JSON files based on the custom order
    json_files = sorted(all_json_files, key=get_sort_key)
    
    print(f"Converting {len(json_files)} JSON files to Excel sheets...")
    print(f"Sheet order: {', '.join([os.path.splitext(f)[0] for f in json_files])}")
    
    # Create Excel writer
    writer = pd.ExcelWriter(
        output_file, 
        engine='xlsxwriter'
    )
    
    # Process each JSON file
    for json_file in tqdm(json_files):
        try:
            # Read JSON file
            with open(os.path.join(input_folder, json_file), 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Skip empty files (files with just empty array or 2 bytes)
            if not data or (isinstance(data, str) and len(data) <= 2):
                print(f"Skipping empty file: {json_file}")
                continue
            
            # Group data by items to merge rows with same base data
            grouped_data = {}
            
            for item in data:
                # Copy only the main item fields (excluding 'similarities')
                base_item = {k: v for k, v in item.items() if k != 'similarities'}
                
                # Create a unique key from the base item
                # Assuming the first field is the ID, or otherwise a unique identifier
                item_key = str(list(base_item.values())[0]) if base_item else str(item.get('id', ''))
                
                # If this is the first time we see this item, initialize it
                if item_key not in grouped_data:
                    grouped_data[item_key] = {
                        'base_item': base_item,
                        'similarities': []
                    }
                
                # Add all similarities for this item
                if 'similarities' in item and item['similarities']:
                    for sim_id, sim_data in item['similarities'].items():
                        grouped_data[item_key]['similarities'].append({
                            'id': sim_data.get('id', ''),
                            'value': sim_data.get('value', ''),
                            'score': f"{sim_data.get('score', '')}%",
                            'check_admin': sim_data.get('check_admin', '')
                        })
            
            # Convert the grouped data to a format suitable for Excel
            flattened_data = []
            merge_ranges = []  # Store the cell ranges to merge
            current_row = 1  # Start from row 1 (after header)
            
            for item_key, item_data in grouped_data.items():
                # Get the base item
                base_item = item_data['base_item']
                similarities = item_data['similarities']
                base_columns = list(base_item.keys())
                
                if not similarities:
                    # If no similarities, add a single row with empty similarity fields
                    row = {**base_item}
                    row['Giống với id'] = ""
                    row['So sánh value với id'] = ""
                    row['Tỉ lệ giống với id'] = ""
                    row['Kiểm tra admin'] = ""
                    row['Confirm'] = ""  # Thêm cột Confirm
                    flattened_data.append(row)
                    current_row += 1
                else:
                    # For the first similarity, include the base item data
                    first_sim = similarities[0]
                    first_row = {**base_item}
                    first_row['Giống với id'] = first_sim['id']
                    first_row['So sánh value với id'] = first_sim['value']
                    first_row['Tỉ lệ giống với id'] = first_sim['score']
                    first_row['Kiểm tra admin'] = first_sim['check_admin']
                    first_row['Confirm'] = ""  # Thêm cột Confirm
                    flattened_data.append(first_row)
                    
                    start_row = current_row
                    current_row += 1
                    
                    # For remaining similarities, only include empty cells for base item
                    for sim in similarities[1:]:
                        # Create a row with empty values for base fields
                        empty_base = {k: "" for k in base_item.keys()}
                        row = {**empty_base}
                        row['Giống với id'] = sim['id']
                        row['So sánh value với id'] = sim['value']
                        row['Tỉ lệ giống với id'] = sim['score']
                        row['Kiểm tra admin'] = sim['check_admin']
                        row['Confirm'] = ""  # Thêm cột Confirm
                        flattened_data.append(row)
                        current_row += 1
                    
                    # If there are multiple similarities, we need to merge cells
                    if len(similarities) > 1:
                        end_row = current_row - 1
                        # Store merge ranges for each base column
                        for col_idx, col_name in enumerate(base_columns):
                            merge_ranges.append((col_idx, start_row, col_idx, end_row))
            
            # Create DataFrame
            df = pd.DataFrame(flattened_data)
            
            # Replace NaN/Inf values with empty strings
            df = df.replace([np.inf, -np.inf, np.nan], '')
            
            # Write to Excel sheet (use filename without extension as sheet name)
            sheet_name = os.path.splitext(json_file)[0]
            # Excel sheet names can't exceed 31 characters
            if len(sheet_name) > 31:
                sheet_name = sheet_name[:31]
            
            # Write DataFrame to Excel with custom header format
            df.to_excel(
                writer, 
                sheet_name=sheet_name, 
                index=False,
                header=False  # Don't write the header with to_excel
            )
            
            # Get workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
            
            # Create a header format with blue background and bold text
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4F81BD',  # Blue background
                'font_color': 'white',   # White text
                'border': 1,
                'border_color': '#000000',
                'text_wrap': True  # Enable text wrapping for headers
            })
            
            # Create cell format with text wrapping
            cell_format = workbook.add_format({
                'text_wrap': True,  # Enable text wrapping
                'valign': 'top',    # Align text to top of cell
                'align': 'center'   # Center text horizontally
            })
            
            # Format with right border and text wrapping
            right_border_format = workbook.add_format({
                'right': 2,         # Medium border
                'right_color': '#000000',  # Black color
                'text_wrap': True,  # Enable text wrapping
                'valign': 'top',    # Align text to top of cell
                'align': 'center'   # Center text horizontally
            })
            
            # Get total number of columns
            total_columns = len(df.columns)
            
            # Apply column widths and group borders for all columns
            for col_num in range(total_columns):
                # Xử lý các cột trong nhóm gốc (0, 1, 2, 3)
                if col_num < total_columns - 1:  # Nếu không phải cột cuối
                    position = col_num % 4
                    
                    # Set width based on position in group
                    if position == 0:  # First column in group
                        width_pixels = 200
                    elif position == 1:  # Second column in group
                        width_pixels = 400
                    elif position == 2:  # Third column in group
                        width_pixels = 200
                    else:  # Fourth column in group
                        width_pixels = 200
                    
                    char_width = width_pixels / 7
                    
                    # Apply format with or without right border
                    if position == 3:  # Fourth column gets right border
                        worksheet.set_column(col_num, col_num, char_width, right_border_format)
                    else:
                        worksheet.set_column(col_num, col_num, char_width, cell_format)
                else:
                    # Cột Confirm ở cuối
                    width_pixels = 150
                    char_width = width_pixels / 7
                    worksheet.set_column(col_num, col_num, char_width, right_border_format)
            
            # Write the header row manually with the custom format
            for col_num, value in enumerate(df.columns):
                worksheet.write(0, col_num, value, header_format)
            
            # Freeze the top row
            worksheet.freeze_panes(1, 0)
            
            # Write data rows (starting from row 1)
            for row_num, row_data in enumerate(df.values):
                for col_num, cell_value in enumerate(row_data):
                    # Handle NaN/Inf values by converting them to empty strings
                    if isinstance(cell_value, float) and (pd.isna(cell_value) or np.isinf(cell_value)):
                        cell_value = ''
                    
                    # Apply appropriate format based on column position
                    if col_num < total_columns - 1:  # Nếu không phải cột cuối
                        position = col_num % 4
                        format_to_use = right_border_format if position == 3 else cell_format
                    else:
                        # Cột Confirm luôn có đường viền phải
                        format_to_use = right_border_format
                    
                    # Write cell with appropriate format
                    worksheet.write(row_num + 1, col_num, cell_value, format_to_use)
            
            # Merge cells for the same items
            merged_format = workbook.add_format({
                'text_wrap': True,
                'valign': 'top',
                'align': 'center',
                'border': 1
            })
            
            # Apply the merges
            for col, start_row, col, end_row in merge_ranges:
                worksheet.merge_range(start_row, col, end_row, col, df.iloc[start_row-1, col], merged_format)
                    
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    writer.close()
    print(f"Excel file created successfully: {output_file}")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Convert JSON files to Excel sheets')
    parser.add_argument('-i', '--input', type=str, default="output/data_question_check_admin_all", 
                        help='Input folder containing JSON files')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output Excel file name (default: folder_name.xlsx)')
    parser.add_argument('--sort', action='store_true',
                        help='Sort sheets according to predefined order')
    
    args = parser.parse_args()
    
    # Define the standard sort order for kind values
    standard_sort_order = [
        "110001", "110002", "110003", "110004", "120001", "120002", "120003", "120004", 
        "210001", "210002", "210003", "210004", "220001", "220002", "220003", "220004", 
        "310001", "310002", "310003", "310004", "320001", "320002", "320003", "330001", 
        "330002", "410001", "410002", "410003_1", "410003_2", "420001", "420002", 
        "420003_1", "420003_2", "430001", "430002", "510001", "510002_1", "510002_2", 
        "520001", "520002", "520003", "530001", "530002", "530003", "610001", "610002", 
        "610003", "620001", "620002", "620003", "620004", "630001"
    ]
    
    # Convert JSON files to Excel with optional sorting
    sort_order = standard_sort_order if args.sort else None
    json_to_excel(args.input, args.output, sort_order) 
import os
import json
import re

def extract_test_number(title_text):
    """Extract test number from title text like 'Test 1' or 'Bài kiểm tra 1'"""
    if not title_text:
        return None
    match = re.search(r'(\d+)$', title_text.strip())
    if match:
        return int(match.group(1))
    # Special case for Korean format like "1 번 시험지"
    match = re.search(r'^(\d+)\s+번', title_text.strip())
    if match:
        return int(match.group(1))
    return None

def get_skill_suffix(skill_name, lang):
    """Get the suffix for a skill name in different languages"""
    skill_suffixes = {
        "Reading": {
            "vi": "Đọc",
            "en": "Reading",
            "ko": "읽기",
            "ja": "読解",
            "fr": "Lecture",
            "ru": "Чтение"
        },
        "Listening": {
            "vi": "Nghe",
            "en": "Listening",
            "ko": "듣기",
            "ja": "聞き取り",
            "fr": "Écoute",
            "ru": "Аудирование"
        },
        "Writing": {
            "vi": "Viết",
            "en": "Writing",
            "ko": "쓰기",
            "ja": "書き取り",
            "fr": "Écriture",
            "ru": "Письмо"
        }
    }
    
    if skill_name in skill_suffixes and lang in skill_suffixes[skill_name]:
        return skill_suffixes[skill_name][lang]
    return skill_name  # Return the original name if not found

def read_json_files_and_combine():
    """
    Read all JSON files from data_exam_redo/input directory,
    combine them into a single array, and save to data_exam_redo/output/combined_data.json
    """
    # Define input and output paths
    input_dir = os.path.join('data_exam_redo', 'input')
    output_dir = os.path.join('data_exam_redo', 'output')
    output_file = os.path.join(output_dir, 'exam_redo_full.json')
    output_skill_file = os.path.join(output_dir, 'exam_redo_skill.json')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the combined data arrays
    combined_data = []
    skill_data = []
    
    # Get all JSON files in the input directory
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    # Sort the file names by level and test number
    def extract_level_and_test(filename):
        match = re.match(r'Level_(\d+)_Test_(\d+)\.json', filename)
        if match:
            level = int(match.group(1))
            test = int(match.group(2))
            return (level, test)
        return (float('inf'), float('inf'))  # For any files that don't match the pattern
    
    json_files.sort(key=extract_level_and_test)
    
    # Read each JSON file and add its content to the combined data array
    for file_name in json_files:
        file_path = os.path.join(input_dir, file_name)
        try:
            # Extract level from filename
            match = re.match(r'Level_(\d+)_Test_(\d+)\.json', file_name)
            if not match:
                print(f"Skipping {file_name}: Could not extract level and test number")
                continue
                
            level = int(match.group(1))
            test_num = int(match.group(2))
            
            # Determine the title increment based on level
            title_increment = 40 if level <= 2 else 15
            
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                if isinstance(data, list):
                    # Process each item in the list
                    for item in data:
                        if isinstance(item, dict):
                            # Create a deep copy of the item for the regular exam
                            regular_item = json.loads(json.dumps(item))
                            regular_item["active"] = 0
                            regular_item["type"] = 4
                            
                            # Update title for regular item
                            if 'title' in regular_item:
                                original_test_num = extract_test_number(regular_item['title'])
                                if original_test_num is not None:
                                    regular_item['title'] = f"Test {original_test_num + title_increment}"
                            
                            # Update title_lang for regular item if it exists
                            if 'title_lang' in regular_item:
                                for lang in regular_item['title_lang']:
                                    title_text = regular_item['title_lang'][lang]
                                    if lang == 'ko':
                                        # Special handling for Korean format
                                        match = re.search(r'^(\d+)\s+번', title_text)
                                        if match:
                                            original_num = int(match.group(1))
                                            new_num = original_num + title_increment
                                            regular_item['title_lang'][lang] = f"{new_num} 번 시험지"
                                    else:
                                        original_test_num = extract_test_number(title_text)
                                        if original_test_num is not None:
                                            regular_item['title_lang'][lang] = title_text.replace(
                                                str(original_test_num), 
                                                str(original_test_num + title_increment)
                                            )
                            
                            # Add regular item to combined data
                            combined_data.append(regular_item)
                            
                            # Extract skill-based tests from parts
                            if 'parts' in item and isinstance(item['parts'], list):
                                original_test_num = extract_test_number(item.get('title', ''))
                                if original_test_num is not None:
                                    new_test_num = original_test_num + title_increment
                                    
                                    # Create separate tests for each skill in parts
                                    for part in item['parts']:
                                        if 'name' in part and part['name'] in ['Listening', 'Reading', 'Writing']:
                                            skill_name = part['name']
                                            skill_test = {
                                                "id": 0,
                                                "title": f"Test {new_test_num} - {skill_name}",
                                                "parts": [part],
                                                "level": level,
                                                "groups": [],
                                                "score": 100,
                                                "active": 0,
                                                "time": part["time"],
                                                "sequence": 0,
                                                "type": 5,
                                                "title_lang": {},
                                            }
                                            
                                            # Create title_lang entries for each language
                                            if 'title_lang' in item:
                                                for lang in item['title_lang']:
                                                    title_text = item['title_lang'][lang]
                                                    original_lang_test_num = extract_test_number(title_text)
                                                    if original_lang_test_num is not None:
                                                        new_lang_test_num = original_lang_test_num + title_increment
                                                        skill_suffix = get_skill_suffix(skill_name, lang)
                                                        
                                                        if lang == 'ko':
                                                            # Special handling for Korean format
                                                            match = re.search(r'^(\d+)\s+번', title_text)
                                                            if match:
                                                                skill_test['title_lang'][lang] = f"{new_lang_test_num} 번 시험지 - {skill_suffix}"
                                                            else:
                                                                skill_test['title_lang'][lang] = f"{title_text.replace(str(original_lang_test_num), str(new_lang_test_num))} - {skill_suffix}"
                                                        else:
                                                            skill_test['title_lang'][lang] = f"{title_text.replace(str(original_lang_test_num), str(new_lang_test_num))} - {skill_suffix}"
                                            
                                            skill_data.append(skill_test)
                    
                elif isinstance(data, dict):
                    # Process a single dictionary item
                    # Create a deep copy of the item for the regular exam
                    regular_item = json.loads(json.dumps(data))
                    regular_item["active"] = 0
                    regular_item["type"] = 4
                    
                    # Update title for regular item
                    if 'title' in regular_item:
                        original_test_num = extract_test_number(regular_item['title'])
                        if original_test_num is not None:
                            regular_item['title'] = f"Test {original_test_num + title_increment}"
                    
                    # Update title_lang for regular item if it exists
                    if 'title_lang' in regular_item:
                        for lang in regular_item['title_lang']:
                            title_text = regular_item['title_lang'][lang]
                            if lang == 'ko':
                                # Special handling for Korean format
                                match = re.search(r'^(\d+)\s+번', title_text)
                                if match:
                                    original_num = int(match.group(1))
                                    new_num = original_num + title_increment
                                    regular_item['title_lang'][lang] = f"{new_num} 번 시험지"
                            else:
                                original_test_num = extract_test_number(title_text)
                                if original_test_num is not None:
                                    regular_item['title_lang'][lang] = title_text.replace(
                                        str(original_test_num), 
                                        str(original_test_num + title_increment)
                                    )
                    
                    # Add regular item to combined data
                    combined_data.append(regular_item)
                    
                    # Extract skill-based tests from parts
                    if 'parts' in data and isinstance(data['parts'], list):
                        original_test_num = extract_test_number(data.get('title', ''))
                        if original_test_num is not None:
                            new_test_num = original_test_num + title_increment
                            
                            # Create separate tests for each skill in parts
                            for part in data['parts']:
                                if 'name' in part and part['name'] in ['Listening', 'Reading', 'Writing']:
                                    skill_name = part['name']
                                    skill_test = {
                                        "id": 0,
                                        "title": f"Test {new_test_num} - {skill_name}",
                                        "parts": [part],
                                        "level": level,
                                        "groups": [],
                                        "score": 100,
                                        "active": 0,
                                        "time": part["time"],
                                        "sequence": 0,
                                        "type": 5,
                                        "title_lang": {},
                                    }
                                    
                                    # Create title_lang entries for each language
                                    if 'title_lang' in data:
                                        for lang in data['title_lang']:
                                            title_text = data['title_lang'][lang]
                                            original_lang_test_num = extract_test_number(title_text)
                                            if original_lang_test_num is not None:
                                                new_lang_test_num = original_lang_test_num + title_increment
                                                skill_suffix = get_skill_suffix(skill_name, lang)
                                                
                                                if lang == 'ko':
                                                    # Special handling for Korean format
                                                    match = re.search(r'^(\d+)\s+번', title_text)
                                                    if match:
                                                        skill_test['title_lang'][lang] = f"{new_lang_test_num} 번 시험지 - {skill_suffix}"
                                                    else:
                                                        skill_test['title_lang'][lang] = f"{title_text.replace(str(original_lang_test_num), str(new_lang_test_num))} - {skill_suffix}"
                                                else:
                                                    skill_test['title_lang'][lang] = f"{title_text.replace(str(original_lang_test_num), str(new_lang_test_num))} - {skill_suffix}"
                                    
                                    skill_data.append(skill_test)
                
            print(f"Processed: {file_name}")
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
    
    # Assign IDs starting from 2699
    NEWEST_ID_COUNT = 2699
    starting_id = NEWEST_ID_COUNT + 1
    
    # Assign IDs to regular tests
    for i, item in enumerate(combined_data):
        item['id'] = starting_id + i
    
    # Assign IDs to skill tests (continuing from where regular tests ended)
    skill_starting_id = starting_id + len(combined_data)
    for i, item in enumerate(skill_data):
        item['id'] = skill_starting_id + i
    
    # Save the combined data to the output files
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(combined_data, outfile, ensure_ascii=False, indent=2)
    
    with open(output_skill_file, 'w', encoding='utf-8') as outfile:
        json.dump(skill_data, outfile, ensure_ascii=False, indent=2)
    
    print(f"Combined {len(json_files)} files into {output_file}")
    print(f"Total regular tests: {len(combined_data)}")
    print(f"Total skill tests: {len(skill_data)}")
    print(f"Files were processed in order from smallest to largest test number within each level")
    print(f"Regular test IDs assigned from {starting_id} to {starting_id + len(combined_data) - 1}")
    print(f"Skill test IDs assigned from {skill_starting_id} to {skill_starting_id + len(skill_data) - 1}")
    print(f"Title numbers updated: +40 for levels 1-2, +15 for levels 3-6")
    print(f"Korean title format fixed")
    print(f"Skill tests created for Reading, Listening, and Writing")
    print(f"Regular tests saved to: {output_file}")
    print(f"Skill tests saved to: {output_skill_file}")

if __name__ == "__main__":
    read_json_files_and_combine()

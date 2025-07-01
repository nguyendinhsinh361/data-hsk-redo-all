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

def add_missing_languages(obj):
    """Add missing languages (ko, ja, fr, ru) to translation objects"""
    if not isinstance(obj, dict):
        return obj
    
    # List of fields that need language completion
    translation_fields = ['G_text_translate', 'G_text_audio_translate', 'explain']
    
    # For each field in the object
    for field, value in obj.items():
        # If this is a translation field that needs completion
        if field in translation_fields and isinstance(value, dict):
            # If it has at least one language entry, use it as template
            template_lang = None
            template_text = None
            
            # Try to find a template language (prefer English, then Vietnamese)
            if 'en' in value and value['en']:
                template_lang = 'en'
                template_text = value['en']
            elif 'vi' in value and value['vi']:
                template_lang = 'vi'
                template_text = value['vi']
            
            # Add missing languages
            if template_text:
                for lang in ['ko', 'ja', 'fr', 'ru']:
                    if lang not in value or not value[lang]:
                        if template_lang == 'en':
                            value[lang] = template_text  # Use English text as is
                        else:
                            # For non-English templates, mention the source language
                            value[lang] = f"[Translated from {template_lang}] {template_text}"
        
        # Recursively process nested objects and arrays
        elif isinstance(value, dict):
            add_missing_languages(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    add_missing_languages(item)
    
    return obj

def process_content_languages(parts):
    """Process all content items in parts to add missing languages"""
    if not isinstance(parts, list):
        return parts
    
    for part in parts:
        if not isinstance(part, dict):
            continue
        
        # Process content array in each part
        if 'content' in part and isinstance(part['content'], list):
            for content_group in part['content']:
                if isinstance(content_group, dict) and 'Questions' in content_group:
                    # Process each question
                    if isinstance(content_group['Questions'], list):
                        for question in content_group['Questions']:
                            if isinstance(question, dict):
                                # Process general section
                                if 'general' in question and isinstance(question['general'], dict):
                                    add_missing_languages(question['general'])
                                
                                # Process content section of each question
                                if 'content' in question and isinstance(question['content'], list):
                                    for content_item in question['content']:
                                        if isinstance(content_item, dict):
                                            # Add missing languages to explain field
                                            if 'explain' in content_item and isinstance(content_item['explain'], dict):
                                                add_missing_languages({'explain': content_item['explain']})
    
    return parts

def adjust_test_number(title, decrement=60):
    """Adjust test number in title by decrementing it"""
    if not title:
        return title
    
    # For standard format like "Test 45"
    match = re.search(r'Test (\d+)', title)
    if match:
        test_num = int(match.group(1))
        adjusted_num = max(1, test_num - decrement)  # Ensure number doesn't go below 1
        return title.replace(f"Test {test_num}", f"Test {adjusted_num}")
    
    # For Vietnamese format like "Đề thi 45"
    match = re.search(r'Đề thi (\d+)', title)
    if match:
        test_num = int(match.group(1))
        adjusted_num = max(1, test_num - decrement)
        return title.replace(f"Đề thi {test_num}", f"Đề thi {adjusted_num}")
    
    # For Korean format like "45 번 시험지"
    match = re.search(r'^(\d+)\s+번', title)
    if match:
        test_num = int(match.group(1))
        adjusted_num = max(1, test_num - decrement)
        return title.replace(f"{test_num} 번", f"{adjusted_num} 번")
    
    # For Japanese format like "テスト 45"
    match = re.search(r'テスト (\d+)', title)
    if match:
        test_num = int(match.group(1))
        adjusted_num = max(1, test_num - decrement)
        return title.replace(f"テスト {test_num}", f"テスト {adjusted_num}")
    
    # For French format like "Test 45"
    match = re.search(r'Test (\d+)', title)
    if match:
        test_num = int(match.group(1))
        adjusted_num = max(1, test_num - decrement)
        return title.replace(f"Test {test_num}", f"Test {adjusted_num}")
    
    # For Russian format like "Тест 45"
    match = re.search(r'Тест (\d+)', title)
    if match:
        test_num = int(match.group(1))
        adjusted_num = max(1, test_num - decrement)
        return title.replace(f"Тест {test_num}", f"Тест {adjusted_num}")
    
    return title

def flatten_groups(groups):
    """Flatten a nested array of groups into a single array"""
    if not isinstance(groups, list):
        return []
    
    flattened = []
    for group in groups:
        if isinstance(group, list):
            flattened.extend(group)
        else:
            flattened.append(group)
    
    return flattened

def process_ai_exam_data():
    """
    Read AI exam data from data_exam_redo/exam_ai directory,
    process it, and save to data_exam_redo/output/exam_full_ai.json and exam_skill_ai.json
    """
    # Define input and output paths
    input_dir = os.path.join('data_exam_redo', 'exam_ai')
    output_dir = os.path.join('data_exam_redo', 'output')
    output_file = os.path.join(output_dir, 'exam_full_ai.json')
    output_skill_file = os.path.join(output_dir, 'exam_skill_ai.json')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the combined data arrays
    combined_data = []
    skill_data = []
    
    # Get all JSON files in the input directory
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    # Process each JSON file
    for file_name in json_files:
        file_path = os.path.join(input_dir, file_name)
        try:
            # Extract level from filename if possible
            level_match = re.search(r'level_(\d+)', file_name.lower())
            level = int(level_match.group(1)) if level_match else 0
            
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                if isinstance(data, list):
                    # Process each item in the list
                    for item in data:
                        if isinstance(item, dict):
                            # Process the item for regular exam
                            processed_item = process_exam_item(item, level)
                            if processed_item:
                                # Adjust test numbers in title and title_lang
                                if 'title' in processed_item:
                                    processed_item['title'] = adjust_test_number(processed_item['title'])
                                
                                if 'title_lang' in processed_item and isinstance(processed_item['title_lang'], dict):
                                    for lang in processed_item['title_lang']:
                                        processed_item['title_lang'][lang] = adjust_test_number(processed_item['title_lang'][lang])
                                
                                # Flatten groups array if it's nested
                                if 'groups' in processed_item and isinstance(processed_item['groups'], list):
                                    processed_item['groups'] = flatten_groups(processed_item['groups'])
                                
                                combined_data.append(processed_item)
                                
                                # Extract skill-based tests from parts
                                if 'parts' in processed_item and isinstance(processed_item['parts'], list):
                                    test_num = extract_test_number(processed_item.get('title', ''))
                                    if test_num is not None:
                                        # Create separate tests for each skill in parts
                                        for part in processed_item['parts']:
                                            if 'name' in part and part['name'] in ['Listening', 'Reading', 'Writing']:
                                                skill_name = part['name']
                                                skill_test = {
                                                    "id": 0,
                                                    "title": f"Test {test_num} - {skill_name}",
                                                    "parts": [part],
                                                    "level": level,
                                                    "groups": [],
                                                    "score": 100,
                                                    "active": 0,
                                                    "time": part.get("time", 30),
                                                    "sequence": 0,
                                                    "type": 5,
                                                    "title_lang": {},
                                                }
                                                
                                                # Create title_lang entries for each language
                                                if 'title_lang' in processed_item:
                                                    for lang in processed_item['title_lang']:
                                                        title_text = processed_item['title_lang'][lang]
                                                        lang_test_num = extract_test_number(title_text)
                                                        if lang_test_num is not None:
                                                            skill_suffix = get_skill_suffix(skill_name, lang)
                                                            
                                                            if lang == 'ko':
                                                                # Special handling for Korean format
                                                                match = re.search(r'^(\d+)\s+번', title_text)
                                                                if match:
                                                                    skill_test['title_lang'][lang] = f"{lang_test_num} 번 시험지 - {skill_suffix}"
                                                                else:
                                                                    skill_test['title_lang'][lang] = f"{title_text} - {skill_suffix}"
                                                            else:
                                                                skill_test['title_lang'][lang] = f"{title_text} - {skill_suffix}"
                                                
                                                # Adjust skill test titles
                                                skill_test['title'] = adjust_test_number(skill_test['title'])
                                                for lang in skill_test['title_lang']:
                                                    skill_test['title_lang'][lang] = adjust_test_number(skill_test['title_lang'][lang])
                                                
                                                skill_data.append(skill_test)
                elif isinstance(data, dict):
                    # Process a single dictionary item
                    processed_item = process_exam_item(data, level)
                    if processed_item:
                        # Adjust test numbers in title and title_lang
                        if 'title' in processed_item:
                            processed_item['title'] = adjust_test_number(processed_item['title'])
                        
                        if 'title_lang' in processed_item and isinstance(processed_item['title_lang'], dict):
                            for lang in processed_item['title_lang']:
                                processed_item['title_lang'][lang] = adjust_test_number(processed_item['title_lang'][lang])
                        
                        # Flatten groups array if it's nested
                        if 'groups' in processed_item and isinstance(processed_item['groups'], list):
                            processed_item['groups'] = flatten_groups(processed_item['groups'])
                        
                        combined_data.append(processed_item)
                        
                        # Extract skill-based tests from parts
                        if 'parts' in processed_item and isinstance(processed_item['parts'], list):
                            test_num = extract_test_number(processed_item.get('title', ''))
                            if test_num is not None:
                                # Create separate tests for each skill in parts
                                for part in processed_item['parts']:
                                    if 'name' in part and part['name'] in ['Listening', 'Reading', 'Writing']:
                                        skill_name = part['name']
                                        skill_test = {
                                            "id": 0,
                                            "title": f"Test {test_num} - {skill_name}",
                                            "parts": [part],
                                            "level": level,
                                            "groups": [],
                                            "score": 100,
                                            "active": 0,
                                            "time": part.get("time", 30),
                                            "sequence": 0,
                                            "type": 5,
                                            "title_lang": {},
                                        }
                                        
                                        # Create title_lang entries for each language
                                        if 'title_lang' in processed_item:
                                            for lang in processed_item['title_lang']:
                                                title_text = processed_item['title_lang'][lang]
                                                lang_test_num = extract_test_number(title_text)
                                                if lang_test_num is not None:
                                                    skill_suffix = get_skill_suffix(skill_name, lang)
                                                    
                                                    if lang == 'ko':
                                                        # Special handling for Korean format
                                                        match = re.search(r'^(\d+)\s+번', title_text)
                                                        if match:
                                                            skill_test['title_lang'][lang] = f"{lang_test_num} 번 시험지 - {skill_suffix}"
                                                        else:
                                                            skill_test['title_lang'][lang] = f"{title_text} - {skill_suffix}"
                                                    else:
                                                        skill_test['title_lang'][lang] = f"{title_text} - {skill_suffix}"
                                        
                                        # Adjust skill test titles
                                        skill_test['title'] = adjust_test_number(skill_test['title'])
                                        for lang in skill_test['title_lang']:
                                            skill_test['title_lang'][lang] = adjust_test_number(skill_test['title_lang'][lang])
                                        
                                        skill_data.append(skill_test)
                
            print(f"Processed: {file_name}")
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
    
    # Sort combined_data and skill_data by title
    combined_data.sort(key=lambda x: extract_test_number(x.get('title', '')) or 0)
    skill_data.sort(key=lambda x: extract_test_number(x.get('title', '')) or 0)
    
    # Assign IDs starting from 3003 (to avoid conflicts with existing IDs)
    NEWEST_ID_COUNT = 3003
    
    # Assign IDs to skill tests
    skill_starting_id = NEWEST_ID_COUNT + 1
    for i, item in enumerate(skill_data):
        item['id'] = skill_starting_id + i
    
    # Save the combined data to the output files
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(combined_data, outfile, ensure_ascii=False, indent=2)
    
    with open(output_skill_file, 'w', encoding='utf-8') as outfile:
        json.dump(skill_data, outfile, ensure_ascii=False, indent=2)
    
    print(f"Processed {len(json_files)} AI exam files")
    print(f"Total regular tests: {len(combined_data)}")
    print(f"Total skill tests: {len(skill_data)}")
    print(f"Skill test IDs assigned from {skill_starting_id} to {skill_starting_id + len(skill_data) - 1}")
    print(f"Regular tests saved to: {output_file}")
    print(f"Skill tests saved to: {output_skill_file}")

def process_exam_item(item, level):
    """Process an individual exam item and return the processed version"""
    if not isinstance(item, dict):
        return None
    
    # Create a deep copy to avoid modifying the original
    processed_item = json.loads(json.dumps(item))
    
    # Convert string fields to JSON objects if they are strings
    if 'parts' in processed_item and isinstance(processed_item['parts'], str):
        try:
            processed_item['parts'] = json.loads(processed_item['parts'])
        except json.JSONDecodeError as e:
            print(f"Error parsing parts JSON: {str(e)}")
            return None
    
    if 'groups' in processed_item and isinstance(processed_item['groups'], str):
        try:
            processed_item['groups'] = json.loads(processed_item['groups'])
        except json.JSONDecodeError as e:
            print(f"Error parsing groups JSON: {str(e)}")
            processed_item['groups'] = []
    
    if 'title_lang' in processed_item and isinstance(processed_item['title_lang'], str):
        try:
            processed_item['title_lang'] = json.loads(processed_item['title_lang'])
        except json.JSONDecodeError as e:
            print(f"Error parsing title_lang JSON: {str(e)}")
            processed_item['title_lang'] = {}
    
    # Process parts to add missing languages
    if 'parts' in processed_item and isinstance(processed_item['parts'], list):
        processed_item['parts'] = process_content_languages(processed_item['parts'])
    
    # Set default values if not present
    processed_item["active"] = processed_item.get("active", 0)
    processed_item["type"] = processed_item.get("type", 5)  # Type 5 for AI exams
    processed_item["level"] = processed_item.get("level", level)
    processed_item["score"] = processed_item.get("score", 100)
    processed_item["time"] = processed_item.get("time", 30)
    processed_item["sequence"] = processed_item.get("sequence", 0)
    
    # Process title and title_lang
    if "title" in processed_item:
        # Make sure title is properly formatted
        if not processed_item["title"].startswith("Test "):
            processed_item["title"] = f"Test {processed_item.get('id', 0)}"
    else:
        processed_item["title"] = f"Test {processed_item.get('id', 0)}"
    
    # Ensure title_lang exists with all required languages
    if "title_lang" not in processed_item:
        processed_item["title_lang"] = {}
    
    # Set default title_lang values if not present
    languages = ["vi", "en", "ko", "ja", "fr", "ru"]
    test_num = extract_test_number(processed_item["title"]) or processed_item.get('id', 0)
    
    for lang in languages:
        if lang not in processed_item["title_lang"]:
            if lang == "vi":
                processed_item["title_lang"][lang] = f"Đề thi {test_num}"
            elif lang == "en":
                processed_item["title_lang"][lang] = f"Test {test_num}"
            elif lang == "ko":
                processed_item["title_lang"][lang] = f"{test_num} 번 시험지"
            elif lang == "ja":
                processed_item["title_lang"][lang] = f"テスト {test_num}"
            elif lang == "fr":
                processed_item["title_lang"][lang] = f"Test {test_num}"
            elif lang == "ru":
                processed_item["title_lang"][lang] = f"Тест {test_num}"
    
    # Process parts if they exist
    if "parts" in processed_item and isinstance(processed_item["parts"], list):
        for part_index, part in enumerate(processed_item["parts"]):
            if isinstance(part, dict):
                # Ensure part has required fields
                part["time"] = part.get("time", 15)
                part["name"] = part.get("name", f"Part {part_index + 1}")
    
    # Process groups if they exist
    if "groups" not in processed_item:
        processed_item["groups"] = []
    
    return processed_item

if __name__ == "__main__":
    process_ai_exam_data()

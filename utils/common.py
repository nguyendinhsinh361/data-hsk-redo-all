from bs4 import BeautifulSoup
import re
import json
import pandas as pd
import os
import csv


def get_raw_data(path):
    data = []
    with open(path, "r", encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data


def save_data_to_json(data, path):
    with open(path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

def is_japanese(word):
    return '\u4e00' <= word <= '\u9fff' or '\u3040' <= word <= '\u30ff'


def is_vietnamese(text):
    return not is_japanese(text)


def is_han(word):
    return all('\u4e00' <= char <= '\u9fff' for char in word)


def is_latin(text):
    latin_character = re.compile(r'^[a-zA-Z]+$')
    return bool(latin_character.match(text))


def contains_latin_or_numbers(text):
    latin_or_numbers_pattern = re.compile(r'[a-zA-Z0-9]+')
    return bool(re.search(latin_or_numbers_pattern, text))


def count_han_words(text):
    han_word_count = 0

    for char in text:
        if is_han(char):
            han_word_count += 1
    return han_word_count


def count_pair_of_parentheses(text):
    text_sparate = text.split("()")
    return len(text_sparate) - 1


def get_number_tag_p(text):
    if (not contains_html(text)):
        return True
    # Parse the HTML
    soup = BeautifulSoup(text, 'html.parser')

    count_opening_tags = len(soup.find_all('p'))
    count_closing_tags = text.count('</p>')
    return count_opening_tags == count_closing_tags

def get_number_tag_p_for_colum_detail(text):
    if (not contains_html(text)):
        return 0
    # Parse the HTML
    soup = BeautifulSoup(text, 'html.parser')

    count_closing_tags = text.count('</p>')
    return count_closing_tags


def get_text_from_html(text):
    if (not contains_html(text)):
        return text
    # Parse the HTML
    soup = BeautifulSoup(text, 'html.parser')

    result = ''.join(soup.find_all(text=True))
    return result


def get_text_from_html_romanji(text):
    if (not contains_html(text)):
        return text
    # Parse the HTML
    soup = BeautifulSoup(text, 'html.parser')

    result = ''.join([tmp.strip() for tmp in soup.find_all(
        text=True) if contains_latin_or_numbers(tmp.strip())])
    if(text.startswith("()<")): result = f"(){result}"
    return result


def contains_html(text):
    return bool(re.search(r'<.*?>', text))


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def check_brackets(col1_explain, condition):
    explain_jp = re.findall(condition, col1_explain)
    return bool(len(explain_jp))


def check_explain_match_type_1(col1, col2, col3, condition):
    try:
        explain_jp = re.findall(condition, col1)
        if not col2 and not explain_jp:
            return True
        if not explain_jp or not col2:
            return False
        answers = [element.strip()
                   for element in col2.split("\n") if element.strip() != ""]
        explain_final = explain_jp[0].split("[な]")[
            0] if "[な]" in explain_jp[0] else explain_jp[0].split("[な")[0]
        if "[" in explain_final:
            explain_final = explain_jp[0].split("[na]")[0]
        return clean_text(answers[int(col3)-1]) == clean_text(explain_final)
    except Exception as e:
        print(e)
        return False


def check_explain_match_type_2(col1, col2, condition):
    explain_jp = re.findall(condition, col1)
    if not col2 and not explain_jp:
        return True
    if not explain_jp or not col2:
        return False
    return clean_text(remove_special_characters(explain_jp[0]).replace(" ", "")) == clean_text(remove_special_characters(col2).replace(" ", ""))


def flatten_recursive(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten_recursive(item))
        else:
            result.append(item)
    return result

def convert_json_to_excel(data, path_excel):
    df = pd.DataFrame(data)
    df.to_excel(path_excel, index=False, engine='openpyxl')


def add_sheet_pandas(filename, sheetname, data):
    if not filename.endswith('.xlsx'):
        filename += '.xlsx'

    try:
        if os.path.exists(filename):
            df = pd.DataFrame(data)
            with pd.ExcelWriter(filename, engine='openpyxl', mode='a') as writer:
                df.to_excel(writer, sheet_name=sheetname, index=False)
        else:
            df = pd.DataFrame(data)
            df.to_excel(filename, sheet_name=sheetname, index=False)
    except Exception as e:
        print(e)


def clean_text(text):
    result = text.strip().strip(",").rstrip("。").rstrip(".").rstrip("?").rstrip("？").upper()
    return result.replace(",", "").replace(" ", "")


def contains_kana(text):
    kana_pattern = r'[\u3040-\u309F\u30A0-\u30FFー]+'
    return bool(re.search(kana_pattern, text))


def contains_romaji(text):
    latin_or_special_pattern = re.compile(
        r'^[a-zA-Z0-9!@#$%^&*()_+{}\[\]:;"\'<>,.?/\\|\-~`\s\n]+$')
    return bool(latin_or_special_pattern.match(text))


def remove_special_characters(text):
    result =  re.sub(r'[^\w\s]', '', text)
    return result.replace(" ", "")

def remove_attributes(html):
    # Regular expression to remove attributes from HTML tags
    pattern = r'<(\w+)(\s+[^>]+)?>'
    
    # Replace the matched pattern with the tag name
    clean_html = re.sub(pattern, r'<\1>', html)
    
    return clean_html

def process_html(text):
    # Find all matches of <tag>content</tag> and replace them
    text = re.sub(r'<div>(.*?)<\/div>', r'\1\n', text)
    pattern = re.compile(r'<[^>]+>')
    text_final = re.sub(pattern, '\n', text)
    return text_final

def csv_to_json(csv_file_path, json_file_path):
    data = []
    
    # Tăng giới hạn kích thước trường CSV nếu cần
    csv.field_size_limit(1000000)  # Đặt giới hạn cao hơn (ví dụ: 1 triệu ký tự)

    # Đọc tệp CSV với mã hóa UTF-8
    with open(csv_file_path, mode='r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            data.append(row)
    
    # Ghi dữ liệu vào tệp JSON với mã hóa UTF-8
    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
    
    return data
        
def list_json_files(directory_path):
    # Liệt kê tất cả các file và chỉ lấy các file có phần mở rộng là .json
    json_files = [f for f in os.listdir(directory_path) if f.endswith('.json') and os.path.isfile(os.path.join(directory_path, f))]
    
    # Hàm để trích xuất số từ tên file, nếu không có số thì trả về 0
    def extract_number(file_name):
        match = re.search(r'\d+', file_name)
        return int(match.group()) if match else 0
    
    # Sắp xếp dựa trên giá trị số học trong tên file
    json_files_sorted = sorted(json_files, key=extract_number)
    return json_files_sorted

def list_any_files(directory_path, extension_path):
    # Liệt kê tất cả các file và chỉ lấy các file có phần mở rộng là .json
    json_files = [f for f in os.listdir(directory_path) if f.endswith(extension_path) and os.path.isfile(os.path.join(directory_path, f))]
    
    # Hàm để trích xuất số từ tên file, nếu không có số thì trả về 0
    def extract_number(file_name):
        match = re.search(r'\d+', file_name)
        return int(match.group()) if match else 0
    
    # Sắp xếp dựa trên giá trị số học trong tên file
    json_files_sorted = sorted(json_files, key=extract_number)
    return json_files_sorted

def find_object_by_word(word, data):
    for obj in data:
        if obj['word'] == word:
            return obj
    return None
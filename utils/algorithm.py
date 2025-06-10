import numpy as np
import torch
from functools import lru_cache
import time
import re
from collections import defaultdict
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
import hashlib
import binascii

# Kiểm tra xem có GPU không và sử dụng nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Biên dịch sẵn các biểu thức chính quy để tăng tốc
HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
WHITESPACE_PATTERN = re.compile(r'\s+')
SPECIAL_CHARS_PATTERN = re.compile(r'[-&;\'\"<>]|&nbsp;|&quot;|&amp;|&apos;|&lt;|&gt;')

# Số lượng cores tối đa để sử dụng (tự động phát hiện)
MAX_CORES = max(1, multiprocessing.cpu_count() - 1)  # Để lại 1 core cho hệ thống

@lru_cache(maxsize=4096)
def levenshtein_distance(str1, str2):
    """
    Tính khoảng cách Levenshtein giữa 2 chuỗi (sử dụng cache để tối ưu)
    Phiên bản tối ưu sử dụng mảng 1 chiều thay vì ma trận 2 chiều
    """
    # Các trường hợp đặc biệt để tránh tính toán không cần thiết
    if str1 == str2:
        return 0
    
    if not str1:
        return len(str2)
    
    if not str2:
        return len(str1)
    
    # Tối ưu hóa cho trường hợp một chuỗi là phần đầu/đuôi của chuỗi kia
    if str1.startswith(str2):
        return len(str1) - len(str2)
    
    if str2.startswith(str1):
        return len(str2) - len(str1)
    
    if str1.endswith(str2):
        return len(str1) - len(str2)
    
    if str2.endswith(str1):
        return len(str2) - len(str1)
    
    # Đảm bảo str1 là chuỗi ngắn hơn để tối ưu hóa
    if len(str1) > len(str2):
        str1, str2 = str2, str1
    
    # Tối ưu hóa bằng cách chỉ sử dụng 2 mảng 1 chiều (previous và current)
    # thay vì ma trận đầy đủ
    len_str1 = len(str1)
    len_str2 = len(str2)
    
    # Chỉ cần giữ hai hàng của ma trận - đảm bảo sử dụng list, không phải range
    prev_row = list(range(len_str1 + 1))
    current_row = [0] * (len_str1 + 1)
    
    for i in range(1, len_str2 + 1):
        current_row[0] = i
        
        for j in range(1, len_str1 + 1):
            # Các phép toán insert, delete, substitution
            insert_cost = current_row[j-1] + 1
            delete_cost = prev_row[j] + 1
            
            if str1[j-1] == str2[i-1]:
                substitute_cost = prev_row[j-1]  # Không thay đổi
            else:
                substitute_cost = prev_row[j-1] + 1
                
            current_row[j] = min(insert_cost, delete_cost, substitute_cost)
            
        # Swap hàng trước và hiện tại
        prev_row, current_row = current_row, prev_row
    
    # Giá trị cuối cùng nằm ở prev_row vì đã swap
    return prev_row[len_str1]

def levenshtein_distance_torch(str1, str2):
    """
    Tính khoảng cách Levenshtein giữa 2 chuỗi sử dụng PyTorch (cho GPU)
    """
    # Các trường hợp đặc biệt để tăng tốc
    if str1 == str2:
        return 0
    
    if not str1:
        return len(str2)
    
    if not str2:
        return len(str1)
    
    # Mã hóa các chuỗi thành các tensor số nguyên (các ký tự)
    # Sử dụng mã ASCII để đơn giản
    chars1 = torch.tensor([ord(c) for c in str1], device=device)
    chars2 = torch.tensor([ord(c) for c in str2], device=device)
    
    len_str1 = len(str1)
    len_str2 = len(str2)
    
    # Khởi tạo ma trận khoảng cách
    distance_matrix = torch.zeros((len_str2 + 1, len_str1 + 1), device=device)
    
    # Khởi tạo hàng và cột đầu tiên
    for i in range(len_str2 + 1):
        distance_matrix[i, 0] = i
    
    for j in range(len_str1 + 1):
        distance_matrix[0, j] = j
    
    # Tính toán ma trận khoảng cách
    for i in range(1, len_str2 + 1):
        for j in range(1, len_str1 + 1):
            if chars2[i-1] == chars1[j-1]:
                substitution_cost = 0
            else:
                substitution_cost = 1
            
            distance_matrix[i, j] = min(
                distance_matrix[i-1, j] + 1,  # Delete
                distance_matrix[i, j-1] + 1,  # Insert
                distance_matrix[i-1, j-1] + substitution_cost  # Substitute
            )
    
    # Giá trị ở góc dưới cùng bên phải là khoảng cách Levenshtein
    result = distance_matrix[-1, -1].item()
    
    return result

def levenshtein_vectorized(batch_a, batch_b):
    """
    Tính khoảng cách Levenshtein cho nhiều cặp chuỗi cùng lúc sử dụng vector hóa PyTorch
    Phiên bản tối ưu với xử lý batch song song
    
    Args:
        batch_a: Batch các chuỗi thứ nhất 
        batch_b: Batch các chuỗi thứ hai
        
    Returns:
        Tensor chứa khoảng cách Levenshtein cho mỗi cặp
    """
    batch_size = len(batch_a)
    if batch_size == 0:
        return torch.zeros(0, device=device)
    
    # Xử lý song song các cặp có cùng độ dài
    # Nhóm các cặp chuỗi theo độ dài để xử lý hiệu quả
    length_groups = {}
    
    for i in range(batch_size):
        a, b = batch_a[i], batch_b[i]
        length_key = (len(a), len(b))
        
        if length_key not in length_groups:
            length_groups[length_key] = {'a': [], 'b': [], 'indices': []}
        
        length_groups[length_key]['a'].append(a)
        length_groups[length_key]['b'].append(b)
        length_groups[length_key]['indices'].append(i)
    
    # Tạo mảng kết quả
    distances = torch.zeros(batch_size, device=device)
    
    # Xử lý từng nhóm độ dài
    for (len_a, len_b), group in length_groups.items():
        group_size = len(group['indices'])
        
        # Trường hợp đặc biệt: một chuỗi rỗng
        if len_a == 0:
            for idx, i in enumerate(group['indices']):
                distances[i] = len_b
            continue
            
        if len_b == 0:
            for idx, i in enumerate(group['indices']):
                distances[i] = len_a
            continue
        
        # Mã hóa các chuỗi thành tensor để vector hóa
        a_tensors = []
        b_tensors = []
        
        for a_str in group['a']:
            a_tensors.append(torch.tensor([ord(c) for c in a_str], dtype=torch.int32, device=device))
            
        for b_str in group['b']:
            b_tensors.append(torch.tensor([ord(c) for c in b_str], dtype=torch.int32, device=device))
        
        # Tạo tensor batch
        a_batch = torch.stack(a_tensors) if len(a_tensors) > 1 else a_tensors[0].unsqueeze(0)
        b_batch = torch.stack(b_tensors) if len(b_tensors) > 1 else b_tensors[0].unsqueeze(0)
        
        # Tạo ma trận DP 3D: [batch_size, len_a+1, len_b+1]
        dp = torch.zeros((group_size, len_a+1, len_b+1), dtype=torch.int32, device=device)
        
        # Khởi tạo
        for i in range(len_a+1):
            dp[:, i, 0] = i
        for j in range(len_b+1):
            dp[:, 0, j] = j
        
        # Tính ma trận DP cho toàn bộ batch
        for i in range(1, len_a+1):
            for j in range(1, len_b+1):
                # Tính cost matrix cho batch
                cost = (a_batch[:, i-1] != b_batch[:, j-1]).to(torch.int32)
                
                # Vector hóa phép toán min
                dp[:, i, j] = torch.min(
                    torch.min(
                        dp[:, i-1, j] + 1,       # Xóa
                        dp[:, i, j-1] + 1        # Chèn
                    ),
                    dp[:, i-1, j-1] + cost       # Thay thế hoặc giữ nguyên
                )
        
        # Lưu kết quả
        for idx, i in enumerate(group['indices']):
            distances[i] = dp[idx, len_a, len_b]
    
    return distances

def levenshtein_distance_batch_optimized(str_list1, str_list2, batch_size=128):
    """
    Phiên bản tối ưu của levenshtein_distance_batch sử dụng batch processing
    với xử lý song song và nhóm theo độ dài
    """
    n1, n2 = len(str_list1), len(str_list2)
    result = np.zeros((n1, n2))
    
    # Xử lý theo batch
    for i in range(0, n1, batch_size):
        i_end = min(i + batch_size, n1)
        batch_a = str_list1[i:i_end]
        
        for j in range(0, n2, batch_size):
            j_end = min(j + batch_size, n2)
            batch_b = str_list2[j:j_end]
            
            # Tạo tất cả các cặp chuỗi trong batch
            pairs_a, pairs_b = [], []
            indices_i, indices_j = [], []
            
            for idx_a, a in enumerate(batch_a):
                for idx_b, b in enumerate(batch_b):
                    # Nếu 2 chuỗi giống nhau, không cần tính
                    if a == b:
                        result[i + idx_a, j + idx_b] = 0
                        continue
                        
                    # Nếu một chuỗi rỗng, distance = độ dài chuỗi kia
                    if len(a) == 0:
                        result[i + idx_a, j + idx_b] = len(b)
                        continue
                        
                    if len(b) == 0:
                        result[i + idx_a, j + idx_b] = len(a)
                        continue
                    
                    # Nếu chuỗi này là phần đầu/đuôi của chuỗi kia
                    if a.startswith(b):
                        result[i + idx_a, j + idx_b] = len(a) - len(b)
                        continue
                        
                    if b.startswith(a):
                        result[i + idx_a, j + idx_b] = len(b) - len(a)
                        continue
                        
                    if a.endswith(b):
                        result[i + idx_a, j + idx_b] = len(a) - len(b)
                        continue
                        
                    if b.endswith(a):
                        result[i + idx_a, j + idx_b] = len(b) - len(a)
                        continue
                    
                    # Thêm vào danh sách cần tính
                    pairs_a.append(a)
                    pairs_b.append(b)
                    indices_i.append(idx_a)
                    indices_j.append(idx_b)
            
            # Tính khoảng cách cho tất cả các cặp còn lại
            if len(pairs_a) > 0:
                distances = levenshtein_vectorized(pairs_a, pairs_b)
                
                # Điền kết quả vào ma trận
                for idx, (idx_a, idx_b) in enumerate(zip(indices_i, indices_j)):
                    result[i + idx_a, j + idx_b] = distances[idx].item()
    
    return result

def levenshtein_distance_numpy(str1, str2):
    """
    Tính khoảng cách Levenshtein giữa 2 chuỗi sử dụng NumPy
    Tối ưu hóa hiệu suất trên CPU
    """
    m, n = len(str1), len(str2)
    
    # Tạo ma trận khoảng cách
    d = np.zeros((m+1, n+1), dtype=np.int32)
    
    # Khởi tạo hàng và cột đầu tiên
    d[0, :] = np.arange(n+1)
    d[:, 0] = np.arange(m+1)
    
    # Tính toán khoảng cách
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if str1[i-1] == str2[j-1] else 1
            d[i, j] = min(
                d[i-1, j] + 1,      # Xóa
                d[i, j-1] + 1,      # Chèn
                d[i-1, j-1] + cost  # Thay thế hoặc giữ nguyên
            )
    
    return d[m, n]

def levenshtein_distance_batch(str_list1, str_list2):
    """
    Tính khoảng cách Levenshtein giữa hai danh sách chuỗi
    """
    # Sử dụng phiên bản nào tùy thuộc vào số lượng phần tử
    use_gpu = torch.cuda.is_available() and len(str_list1) * len(str_list2) > 100
    use_numpy = len(str_list1) * len(str_list2) > 100
    
    if use_gpu and len(str_list1) * len(str_list2) > 1000:
        # Sử dụng phiên bản vector hóa hoàn toàn cho bộ dữ liệu lớn
        return levenshtein_distance_batch_optimized(str_list1, str_list2)
    
    results = []
    for s1 in str_list1:
        row = []
        for s2 in str_list2:
            if use_gpu:
                row.append(levenshtein_distance_torch(s1, s2))
            elif use_numpy:
                row.append(levenshtein_distance_numpy(s1, s2))
            else:
                row.append(levenshtein_distance(s1, s2))
        results.append(row)
    return np.array(results)

def calculate_similarity(str1, str2):
    """
    Tính độ tương đồng từ 0 đến 1 (1 là giống nhau hoàn toàn)
    """
    max_length = max(len(str1), len(str2))
    if max_length == 0:
        return 1.0  # Cả 2 chuỗi đều rỗng
    
    # Sử dụng GPU nếu chuỗi dài và có GPU
    if max_length > 100 and torch.cuda.is_available():
        distance = levenshtein_distance_torch(str1, str2)
    # Sử dụng NumPy nếu chuỗi dài
    elif max_length > 50:
        distance = levenshtein_distance_numpy(str1, str2)
    else:
        distance = levenshtein_distance(str1, str2)
    
    return (max_length - distance) / max_length

def calculate_similarity_batch(batch_1, batch_2=None):
    """
    Tính ma trận độ tương đồng giữa 2 batch chuỗi
    """
    if batch_2 is None:
        batch_2 = batch_1
    
    n = len(batch_1)
    m = len(batch_2)
    
    # Khởi tạo ma trận kết quả
    similarity_matrix = np.zeros((n, m))
    
    # Tính toán độ tương đồng cho từng cặp
    for i in range(n):
        for j in range(m):
            similarity_matrix[i, j] = calculate_similarity(batch_1[i], batch_2[j])
    
    return similarity_matrix

def batch_process_large_arrays(arr, batch_size=100):
    """
    Chia mảng thành các batch nhỏ hơn để xử lý dữ liệu lớn
    """
    for i in range(0, len(arr), batch_size):
        yield arr[i:i + batch_size]

def check_string_similarity(str1, str2):
    """
    Hàm chính để kiểm tra độ giống nhau của 2 chuỗi
    """
    similarity = calculate_similarity(str1, str2)
    distance = levenshtein_distance(str1, str2)
    percentage = round(similarity * 100, 2)
    
    result = {
        'similarity': similarity,
        'percentage': f"{percentage}%",
        'distance': distance,
        'details': {
            'string1': str1,
            'string2': str2,
            'length1': len(str1),
            'length2': len(str2)
        }
    }
    
    return result

# Phiên bản đơn giản hơn nếu chỉ cần độ tương đồng
def simple_similarity(str1, str2):
    """
    Hàm đơn giản chỉ trả về phần trăm tương đồng
    """
    return round(calculate_similarity(str1, str2) * 100, 2)

def compare_array_elements(arr):
    """
    So sánh từng phần tử trong mảng với tất cả các phần tử còn lại
    
    Args:
        arr: Mảng chuỗi cần so sánh
    
    Returns:
        Dictionary chứa kết quả so sánh cho từng phần tử
    """
    n = len(arr)
    results = {}
    
    # Tính toán ma trận tương đồng một lần duy nhất
    similarity_matrix = calculate_similarity_batch(arr)
    
    for i in range(n):
        current_element = arr[i]
        comparisons = []
        
        for j in range(n):
            if i != j:  # Không so sánh với chính nó
                other_element = arr[j]
                similarity = similarity_matrix[i][j]
                distance = levenshtein_distance(current_element, other_element)
                percentage = round(similarity * 100, 2)
                
                comparison = {
                    'index': j,
                    'compared_with': other_element,
                    'similarity_percentage': f"{percentage}%",
                    'similarity_score': similarity,
                    'distance': distance
                }
                comparisons.append(comparison)
        
        results[f"element_{i}"] = {
            'value': current_element,
            'index': i,
            'comparisons': comparisons,
            'total_comparisons': len(comparisons)
        }
    
    return results

def clean_text(text):
    """
    Làm sạch văn bản: loại bỏ ký tự đặc biệt, khoảng trắng thừa
    """
    # Chuyển đổi text thành chuỗi nếu nó không phải là chuỗi
    if not isinstance(text, str):
        text = str(text)
    
    # Loại bỏ các ký tự đặc biệt
    text = SPECIAL_CHARS_PATTERN.sub(' ', text)
    
    # Loại bỏ khoảng trắng thừa và chuyển thành chữ thường
    text = ' '.join(text.lower().split())
    
    return text

def clean_array(arr):
    """
    Làm sạch mảng chuỗi văn bản
    """
    return [clean_text(item) for item in arr]

def prefilter_strings(strings, min_similarity=50, method='length_ratio'):
    """
    Tiền lọc các chuỗi có khả năng tương đồng để giảm số lượng phép so sánh
    
    Args:
        strings: Mảng các chuỗi cần so sánh
        min_similarity: Ngưỡng độ tương đồng tối thiểu (0-100)
        method: Phương pháp tiền lọc ('length_ratio', 'prefix', hoặc 'combined')
    
    Returns:
        Dictionary với key là chỉ số và value là danh sách các chỉ số tiềm năng
    """
    n = len(strings)
    potential_pairs = defaultdict(list)
    
    # Chuyển đổi ngưỡng từ phần trăm sang tỷ lệ (0-1)
    min_similarity_ratio = min_similarity / 100
    
    if method == 'length_ratio' or method == 'combined':
        # Sử dụng tỷ lệ độ dài để tiền lọc
        lengths = np.array([len(s) for s in strings])
        
        for i in range(n):
            len_i = lengths[i]
            if len_i == 0:
                continue
                
            # Tính tỷ lệ độ dài: len_j / len_i nếu len_j <= len_i, ngược lại len_i / len_j
            # Nếu tỷ lệ này > min_similarity_ratio, thì độ tương đồng tối đa có thể đạt được là ratio
            # (do sự khác biệt về độ dài)
            
            # Sử dụng vectorization để tăng tốc
            ratios = np.minimum(len_i, lengths) / np.maximum(len_i, lengths)
            
            # Lọc các chuỗi có tỷ lệ độ dài >= min_similarity_ratio
            potential_indices = np.where((ratios >= min_similarity_ratio) & (np.arange(n) != i))[0]
            
            for j in potential_indices:
                if i != j:  # Tránh so sánh với chính nó
                    potential_pairs[i].append(j)
    
    elif method == 'prefix' or method == 'combined':
        # Sử dụng tiền tố chung để tiền lọc
        # Ý tưởng: Nếu hai chuỗi có tiền tố chung dài, chúng có khả năng tương đồng
        # Tính dần độ dài tiền tố, từ ngắn đến dài
        min_prefix_len = int(min_similarity_ratio * min(len(s) for s in strings if s))
        
        # Dict lưu trữ ánh xạ từ tiền tố đến các chuỗi có tiền tố đó
        prefix_map = defaultdict(list)
        
        for i, s in enumerate(strings):
            if not s:
                continue
                
            # Thêm các tiền tố với độ dài khác nhau
            for prefix_len in range(min_prefix_len, len(s) + 1):
                prefix = s[:prefix_len]
                prefix_map[prefix].append(i)
        
        # Xây dựng các cặp tiềm năng từ prefix_map
        for indices in prefix_map.values():
            if len(indices) > 1:  # Chỉ xử lý nếu có từ 2 chuỗi trở lên có cùng tiền tố
                for i in indices:
                    for j in indices:
                        if i != j:
                            potential_pairs[i].append(j)
    
    # Loại bỏ các cặp trùng lặp
    for i in potential_pairs:
        potential_pairs[i] = list(set(potential_pairs[i]))
    
    # Nếu không có cặp nào được tìm thấy, trả về tất cả các cặp
    if sum(len(pairs) for pairs in potential_pairs.values()) == 0:
        for i in range(n):
            potential_pairs[i] = [j for j in range(n) if j != i]
    
    return potential_pairs

def compare_array_simple(arr, use_optimized=True, batch_size=50, similarity_threshold=75, prefilter=True, num_workers=None):
    """
    Phiên bản đơn giản - giữ nguyên mảng object và thêm các giá trị tương đồng
    OPTIMIZED VERSION 2.0 - Cải tiến performance đáng kể
    
    Args:
        arr: Mảng object có các trường id, value cần so sánh
        use_optimized: Sử dụng phiên bản tối ưu cho bộ dữ liệu lớn
        batch_size: Kích thước batch cho xử lý dữ liệu lớn
        similarity_threshold: Ngưỡng độ tương đồng (0-100) để giữ lại kết quả
        prefilter: Sử dụng tiền lọc để giảm số lượng phép so sánh
        num_workers: Số lượng worker cho multiprocessing (mặc định là số core - 1)
    
    Returns:
        Mảng object gốc với trường similarities chứa độ tương đồng với các object khác (score >= threshold)
    """
    if num_workers is None:
        num_workers = MAX_CORES
        
    print(f"Sử dụng {num_workers} workers cho xử lý song song")
    
    start_time = time.time()
    values = [item['value'] for item in arr]
    cleaned_values = clean_array(values)
    ids = [item['id'] for item in arr]
    n = len(arr)
    
    print(f"Bắt đầu xử lý {n} phần tử...")
    
    # Tiền lọc các cặp có thể tương đồng để giảm số lượng phép so sánh
    potential_pairs = None
    if prefilter and n > 100:
        print("Đang tiền lọc các cặp có thể tương đồng...")
        prefilter_time = time.time()
        potential_pairs = prefilter_strings(cleaned_values, min_similarity=similarity_threshold * 0.7, method='combined')
        total_pairs = sum(len(pairs) for pairs in potential_pairs.values())
        print(f"Đã giảm từ {n*(n-1)} phép so sánh xuống {total_pairs} phép so sánh")
        print(f"Thời gian tiền lọc: {time.time() - prefilter_time:.2f} giây")
    
    # Tạo bản sao của mảng gốc để không làm thay đổi mảng gốc
    result = []
    
    # Khởi tạo ma trận tương đồng
    similarity_percentages = np.zeros((n, n))
    
    # Xử lý theo batch nếu mảng quá lớn và yêu cầu tối ưu
    if n > batch_size and use_optimized:
        print(f"Mảng có {n} phần tử, xử lý theo batch...")
        
        # Sử dụng multiprocessing để xử lý song song
        if n > 200 and potential_pairs:
            process_pairs_time = time.time()
            print(f"Đang xử lý song song các cặp tiềm năng với {num_workers} workers...")
            
            # Sử dụng multiprocessing để xử lý song song
            similarity_matrix = parallel_process_strings(potential_pairs, cleaned_values, num_workers)
            similarity_percentages = similarity_matrix
            
            print(f"Thời gian xử lý song song: {time.time() - process_pairs_time:.2f} giây")
        else:
            # Phương pháp xử lý batch cũ
            process_time = time.time()
            
            if potential_pairs:
                # Chỉ xử lý các cặp tiềm năng
                for i in range(n):
                    pairs_j = potential_pairs[i]
                    if not pairs_j:
                        continue
                        
                    # Chỉ tính toán các cặp tiềm năng
                    for j in pairs_j:
                        # Tính độ tương đồng giữa 2 chuỗi cụ thể
                        if torch.cuda.is_available() and len(cleaned_values[i]) > 100 and len(cleaned_values[j]) > 100:
                            distance = levenshtein_distance_torch(cleaned_values[i], cleaned_values[j])
                        elif len(cleaned_values[i]) > 50 or len(cleaned_values[j]) > 50:
                            distance = levenshtein_distance_numpy(cleaned_values[i], cleaned_values[j])
                        else:
                            distance = levenshtein_distance(cleaned_values[i], cleaned_values[j])
                            
                        max_length = max(len(cleaned_values[i]), len(cleaned_values[j]))
                        similarity = (max_length - distance) / max_length if max_length > 0 else 1.0
                        similarity_percentages[i, j] = round(similarity * 100, 2)
            else:
                # Xử lý theo ô vuông (phân mảnh ma trận tương đồng)
                for i, batch_i in enumerate(batch_process_large_arrays(cleaned_values, batch_size)):
                    i_start = i * batch_size
                    i_end = min(i_start + len(batch_i), n)
                    
                    for j, batch_j in enumerate(batch_process_large_arrays(cleaned_values, batch_size)):
                        j_start = j * batch_size
                        j_end = min(j_start + len(batch_j), n)
                        
                        # Bỏ qua các phép so sánh dư thừa khi đã có tiền lọc
                        if potential_pairs:
                            skip = True
                            for idx_i in range(i_start, i_end):
                                pairs_j = [j for j in potential_pairs[idx_i] if j_start <= j < j_end]
                                if pairs_j:
                                    skip = False
                                    break
                            if skip:
                                continue
                        
                        # Tính toán ma trận tương đồng cho batch hiện tại
                        if torch.cuda.is_available() and len(batch_i) * len(batch_j) > 1000:
                            batch_distances = levenshtein_distance_batch_optimized(batch_i, batch_j)
                            
                            # Tính max_lengths và độ tương đồng
                            batch_lengths_i = np.array([len(s) for s in batch_i])
                            batch_lengths_j = np.array([len(s) for s in batch_j])
                            max_lengths = np.maximum.outer(batch_lengths_i, batch_lengths_j)
                            max_lengths = np.maximum(max_lengths, 1)  # Tránh chia cho 0
                            
                            batch_similarities = (max_lengths - batch_distances) / max_lengths
                            batch_percentages = np.round(batch_similarities * 100, 2)
                        else:
                            # Sử dụng phiên bản cũ nếu không có GPU hoặc batch nhỏ
                            batch_matrix = calculate_similarity_batch(batch_i, batch_j)
                            batch_percentages = np.round(batch_matrix * 100, 2)
                        
                        # Cập nhật vào ma trận kết quả
                        similarity_percentages[i_start:i_end, j_start:j_end] = batch_percentages
            
            print(f"Thời gian tính toán tương đồng: {time.time() - process_time:.2f} giây")
    else:
        # Xử lý đơn giản cho tập dữ liệu nhỏ
        if potential_pairs and n > 50:
            # Sử dụng xử lý song song cho tập dữ liệu nhỏ
            process_time = time.time()
            similarity_matrix = parallel_process_strings(potential_pairs, cleaned_values, num_workers)
            similarity_percentages = similarity_matrix
            print(f"Thời gian xử lý song song: {time.time() - process_time:.2f} giây")
        else:
            # Tính toán ma trận tương đồng một lần cho toàn bộ dữ liệu
            similarity_matrix = calculate_similarity_batch(cleaned_values)
            
            # Chuyển đổi thành phần trăm và làm tròn
            similarity_percentages = np.round(similarity_matrix * 100, 2)
    
    # Tạo result cho từng object
    build_result_time = time.time()
    print("Đang xây dựng kết quả...")
    
    # Sử dụng ThreadPoolExecutor để xây dựng kết quả song song
    def build_result_item(i):
        # Tạo bản sao của object gốc
        obj = arr[i].copy()
        # Thêm trường similarities
        obj['similarities'] = {}
        
        # Nếu có tiền lọc, chỉ kiểm tra các cặp tiềm năng
        check_indices = potential_pairs[i] if potential_pairs else range(n)
        
        for j in check_indices:
            if i == j:  # Không cần so sánh với chính nó
                continue
                
            similarity_score = similarity_percentages[i][j]
            if similarity_score >= similarity_threshold:
                obj['similarities'][ids[j]] = {
                    'id': ids[j],
                    'value': arr[j]['value_original'],
                    'score': float(similarity_score),
                    'check_admin': arr[j]['check_admin']
                }
        
        return obj
    
    # Xử lý song song việc xây dựng kết quả
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        result = list(executor.map(build_result_item, range(n)))
    
    print(f"Thời gian xây dựng kết quả: {time.time() - build_result_time:.2f} giây")
    
    end_time = time.time()
    print(f"Tổng thời gian xử lý: {end_time - start_time:.2f} giây")
    
    return result

def benchmark_similarity_algorithms(data_size=100, str_length=20, repeat=3):
    """
    So sánh hiệu suất của các phiên bản thuật toán tính độ tương đồng
    
    Args:
        data_size: Số lượng chuỗi cần so sánh
        str_length: Độ dài trung bình của mỗi chuỗi
        repeat: Số lần lặp lại để lấy trung bình thời gian
        
    Returns:
        Dictionary chứa kết quả benchmark
    """
    import random
    import string
    import time
    
    # Tạo dữ liệu ngẫu nhiên
    print(f"Tạo {data_size} chuỗi ngẫu nhiên độ dài {str_length}...")
    
    def random_string(length):
        return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))
    
    strings = [random_string(random.randint(str_length // 2, str_length * 2)) for _ in range(data_size)]
    cleaned_strings = clean_array(strings)
    
    # Chuẩn bị dữ liệu cho API
    test_array = [
        {"id": str(i), "value": s, "value_original": s.upper(), "check_admin": i % 2 == 0}
        for i, s in enumerate(strings)
    ]
    
    results = {}
    
    # Benchmark cho từng phiên bản
    print("\nBenchmark cho levenshtein_distance:")
    total_time = 0
    for _ in range(repeat):
        sample_a = random.choice(cleaned_strings)
        sample_b = random.choice(cleaned_strings)
        
        start = time.time()
        _ = levenshtein_distance(sample_a, sample_b)
        elapsed = time.time() - start
        total_time += elapsed
    
    results["levenshtein_standard"] = total_time / repeat
    print(f"  Phiên bản tiêu chuẩn: {results['levenshtein_standard']:.6f} giây/cặp")
    
    if torch.cuda.is_available():
        print("\nBenchmark cho levenshtein_distance_torch:")
        total_time = 0
        for _ in range(repeat):
            sample_a = random.choice(cleaned_strings)
            sample_b = random.choice(cleaned_strings)
            
            start = time.time()
            _ = levenshtein_distance_torch(sample_a, sample_b)
            elapsed = time.time() - start
            total_time += elapsed
        
        results["levenshtein_torch"] = total_time / repeat
        print(f"  Phiên bản PyTorch: {results['levenshtein_torch']:.6f} giây/cặp")
    
    # Benchmark cho các phiên bản batch
    print("\nBenchmark cho ma trận tương đồng toàn bộ:")
    
    # Chọn kích thước nhỏ hơn để so sánh nhanh hơn
    small_size = min(data_size, 100)
    small_strings = cleaned_strings[:small_size]
    
    start = time.time()
    _ = calculate_similarity_batch(small_strings)
    elapsed = time.time() - start
    results["similarity_batch_standard"] = elapsed
    print(f"  Phiên bản thông thường ({small_size}x{small_size}): {elapsed:.4f} giây")
    
    if torch.cuda.is_available() and small_size > 10:
        start = time.time()
        _ = levenshtein_distance_batch_optimized(small_strings, small_strings)
        elapsed = time.time() - start
        results["similarity_batch_optimized"] = elapsed
        print(f"  Phiên bản vector hóa ({small_size}x{small_size}): {elapsed:.4f} giây")
    
    # Benchmark cho so sánh mảng
    print("\nBenchmark cho compare_array_simple:")
    small_array = test_array[:small_size]
    
    start = time.time()
    _ = compare_array_simple(small_array)
    elapsed = time.time() - start
    results["compare_array_simple"] = elapsed
    print(f"  Thời gian xử lý {small_size} phần tử: {elapsed:.4f} giây")
    
    # Tính toán tăng tốc
    if torch.cuda.is_available():
        if "levenshtein_torch" in results and "levenshtein_standard" in results:
            speedup = results["levenshtein_standard"] / results["levenshtein_torch"]
            print(f"\nTăng tốc Levenshtein đơn cặp: {speedup:.2f}x")
        
        if "similarity_batch_optimized" in results and "similarity_batch_standard" in results:
            speedup = results["similarity_batch_standard"] / results["similarity_batch_optimized"]
            print(f"Tăng tốc ma trận tương đồng: {speedup:.2f}x")
    
    return results

# Ví dụ sử dụng
def test_check_string_similarity():
    print("\n" + "="*60)
    print(f"Using device: {device}")
    
    # Tạo mảng test với cấu trúc giống thực tế
    test_array = [
        {"id": "1", "value": "abc", "value_original": "ABC", "check_admin": True},
        {"id": "2", "value": "xyzabc", "value_original": "XYZABC", "check_admin": False},
        {"id": "3", "value": "abcd", "value_original": "ABCD", "check_admin": True},
        {"id": "4", "value": "xyz", "value_original": "XYZ", "check_admin": False},
        {"id": "5", "value": "hello", "value_original": "HELLO", "check_admin": True},
        {"id": "6", "value": "helo", "value_original": "HELO", "check_admin": False},
        {"id": "7", "value": "abc", "value_original": "ABC", "check_admin": True}
    ]
    
    # So sánh thời gian thực thi
    start = time.time()
    result = compare_array_simple(test_array)
    end = time.time()
    
    print(f"Thời gian thực thi: {end - start:.4f} giây")
    print(result[0]['similarities'])  # In ra kết quả của phần tử đầu tiên
    
    # Chạy benchmark nếu cần
    print("\nBắt đầu benchmark...")
    benchmark_similarity_algorithms(data_size=100, str_length=30, repeat=3)

def test_with_large_dataset(max_size=1000, steps=6, str_length_range=(10, 50)):
    """
    Kiểm tra hiệu suất thuật toán với bộ dữ liệu có kích thước khác nhau
    
    Args:
        max_size: Kích thước tối đa của dữ liệu
        steps: Số lượng kích thước khác nhau cần kiểm tra
        str_length_range: Khoảng độ dài chuỗi ngẫu nhiên (min, max)
    
    Returns:
        Dictionary chứa kết quả benchmark
    """
    import random
    import string
    import time
    import numpy as np
    
    try:
        import matplotlib.pyplot as plt
        has_matplotlib = True
    except ImportError:
        has_matplotlib = False
    
    print("\n" + "="*60)
    print("Kiểm tra hiệu suất với bộ dữ liệu lớn")
    print("="*60)
    
    # Tạo dữ liệu ngẫu nhiên với nhiều kích thước khác nhau
    # Chia theo thang logarit để thấy rõ sự khác biệt
    if steps <= 1:
        sizes = [max_size]
    else:
        sizes = [int(max_size * (i / (steps - 1))) for i in range(steps)]
        sizes[0] = max(sizes[0], 10)  # Đảm bảo kích thước tối thiểu là 10
    
    std_times = []
    opt_times = []
    
    def random_string(length):
        return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))
    
    for size in sizes:
        print(f"\nKích thước dữ liệu: {size}")
        strings = [random_string(random.randint(str_length_range[0], str_length_range[1])) for _ in range(size)]
        cleaned_strings = clean_array(strings)
        
        # Tạo dữ liệu test cho API
        test_array = [
            {"id": str(i), "value": s, "value_original": s.upper(), "check_admin": i % 2 == 0}
            for i, s in enumerate(strings)
        ]
        
        # Đo thời gian cho phiên bản tiêu chuẩn
        start = time.time()
        _ = compare_array_simple(test_array, use_optimized=False)
        std_time = time.time() - start
        std_times.append(std_time)
        print(f"  Phiên bản tiêu chuẩn: {std_time:.4f} giây")
        
        # Đo thời gian cho phiên bản tối ưu
        start = time.time()
        _ = compare_array_simple(test_array, use_optimized=True)
        opt_time = time.time() - start
        opt_times.append(opt_time)
        print(f"  Phiên bản tối ưu: {opt_time:.4f} giây")
        
        if std_time > 0:
            speedup = std_time / opt_time
            print(f"  Tăng tốc: {speedup:.2f}x")
    
    # Tạo biểu đồ so sánh nếu có matplotlib
    if has_matplotlib:
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(sizes))
        width = 0.35
        
        plt.bar(x - width/2, std_times, width, label='Phiên bản tiêu chuẩn')
        plt.bar(x + width/2, opt_times, width, label='Phiên bản tối ưu')
        
        plt.xlabel('Kích thước dữ liệu')
        plt.ylabel('Thời gian thực thi (giây)')
        plt.title('So sánh hiệu suất với kích thước dữ liệu khác nhau')
        plt.xticks(x, sizes)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png')
        print("\nĐã lưu biểu đồ so sánh tại benchmark_results.png")
    else:
        print("\nKhông có matplotlib, bỏ qua việc tạo biểu đồ")
    
    return {"sizes": sizes, "standard_times": std_times, "optimized_times": opt_times}

# Chuẩn bị một hàm có thể chạy trong quy trình riêng biệt
def process_string_batch(data_batch):
    """Xử lý một batch các cặp chuỗi trong một quy trình riêng biệt."""
    results = []
    for item in data_batch:
        i, j, str1, str2 = item
        distance = levenshtein_distance(str1, str2)
        max_length = max(len(str1), len(str2))
        similarity = (max_length - distance) / max_length if max_length > 0 else 1.0
        results.append((i, j, round(similarity * 100, 2)))
    return results

def parallel_process_strings(pairs, cleaned_values, num_workers=None):
    """Xử lý song song các cặp chuỗi sử dụng multiple cores."""
    if num_workers is None:
        num_workers = MAX_CORES
    
    # Chuẩn bị dữ liệu đầu vào
    data_batches = []
    for i, j_list in pairs.items():
        for j in j_list:
            data_batches.append((i, j, cleaned_values[i], cleaned_values[j]))
    
    # Chia thành các batches nhỏ hơn
    batch_size = max(100, len(data_batches) // (num_workers * 10))
    all_batches = [data_batches[i:i + batch_size] for i in range(0, len(data_batches), batch_size)]
    
    # Xử lý song song
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for batch_result in executor.map(process_string_batch, all_batches):
            results.extend(batch_result)
    
    # Chuyển đổi kết quả thành ma trận
    similarity_matrix = np.zeros((len(cleaned_values), len(cleaned_values)))
    for i, j, similarity in results:
        similarity_matrix[i, j] = similarity
    
    return similarity_matrix

def simhash(text, hash_bits=64):
    """
    Tạo SimHash cho văn bản.
    
    Args:
        text: Chuỗi cần tạo hash
        hash_bits: Số bit của hash (mặc định: 64)
        
    Returns:
        Giá trị hash dạng số nguyên
    """
    if not text:
        return 0
        
    # Chuyển văn bản thành chữ thường và chuẩn hóa
    text = clean_text(text)
    
    # Tạo các shingle (n-gram) từ văn bản
    shingles = []
    n = 3  # kích thước n-gram
    for i in range(len(text) - n + 1):
        shingles.append(text[i:i+n])
    
    # Nếu văn bản quá ngắn, sử dụng toàn bộ văn bản
    if not shingles:
        shingles = [text]
    
    # Khởi tạo vector trọng số cho các bit
    v = [0] * hash_bits
    
    # Mã hóa các shingle và tính trọng số
    for shingle in shingles:
        # Sử dụng MD5 để tạo hash cho shingle
        h = hashlib.md5(shingle.encode('utf-8')).digest()
        
        # Chuyển đổi hash thành số nguyên
        h_int = int(binascii.hexlify(h), 16)
        
        # Cập nhật trọng số cho mỗi bit
        for i in range(hash_bits):
            bitmask = 1 << i
            if h_int & bitmask:
                v[i] += 1
            else:
                v[i] -= 1
    
    # Tạo fingerprint từ các trọng số
    fingerprint = 0
    for i in range(hash_bits):
        if v[i] > 0:
            fingerprint |= (1 << i)
    
    return fingerprint

def hamming_distance(hash1, hash2):
    """
    Tính khoảng cách Hamming giữa hai giá trị hash.
    
    Args:
        hash1, hash2: Các giá trị hash
        
    Returns:
        Số bit khác nhau giữa hai hash
    """
    xor = hash1 ^ hash2
    distance = 0
    
    # Đếm số bit 1 trong phép XOR
    while xor:
        distance += xor & 1
        xor >>= 1
    
    return distance

def simhash_similarity(hash1, hash2, hash_bits=64):
    """
    Tính độ tương đồng dựa trên khoảng cách Hamming giữa hai SimHash.
    
    Args:
        hash1, hash2: Các giá trị SimHash
        hash_bits: Số bit của hash
        
    Returns:
        Độ tương đồng từ 0 đến 1
    """
    distance = hamming_distance(hash1, hash2)
    return 1.0 - (distance / hash_bits)

def compare_array_simhash(arr, similarity_threshold=75, num_workers=None, hash_bits=64):
    """
    So sánh mảng các chuỗi sử dụng SimHash để tăng tốc độ.
    
    Args:
        arr: Mảng object có các trường id, value cần so sánh
        similarity_threshold: Ngưỡng độ tương đồng (0-100)
        num_workers: Số lượng worker cho multiprocessing
        hash_bits: Số bit của SimHash
        
    Returns:
        Mảng object với trường similarities
    """
    if num_workers is None:
        num_workers = MAX_CORES
    
    start_time = time.time()
    values = [item['value'] for item in arr]
    cleaned_values = clean_array(values)
    ids = [item['id'] for item in arr]
    n = len(arr)
    
    print(f"Bắt đầu xử lý {n} phần tử với SimHash ({hash_bits} bits)...")
    
    # Tính SimHash cho mỗi chuỗi
    print("Đang tính SimHash cho các chuỗi...")
    
    # Tính song song SimHash
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        hashes = list(executor.map(lambda x: simhash(x, hash_bits), cleaned_values))
    
    # Tìm các cặp có khả năng tương đồng
    print("Đang tìm các cặp có khả năng tương đồng...")
    potential_pairs = defaultdict(list)
    
    # Ngưỡng khoảng cách Hamming tương ứng với ngưỡng tương đồng
    hamming_threshold = int((1 - similarity_threshold/100) * hash_bits)
    
    # Tìm các cặp có khoảng cách Hamming nhỏ
    for i in range(n):
        for j in range(i+1, n):
            # Tính khoảng cách Hamming
            distance = hamming_distance(hashes[i], hashes[j])
            
            # Nếu khoảng cách nhỏ hơn ngưỡng, thêm vào danh sách tiềm năng
            if distance <= hamming_threshold:
                potential_pairs[i].append(j)
                potential_pairs[j].append(i)
    
    total_pairs = sum(len(pairs) for pairs in potential_pairs.values())
    print(f"Đã giảm từ {n*(n-1)//2} phép so sánh xuống {total_pairs} phép so sánh")
    
    # Tạo bản sao của mảng gốc để không làm thay đổi mảng gốc
    result = []
    
    # Tính toán độ tương đồng Levenshtein chính xác cho các cặp tiềm năng
    print("Đang tính toán độ tương đồng Levenshtein chính xác...")
    similarity_percentages = np.zeros((n, n))
    
    # Chuẩn bị các cặp cần tính toán
    pairs_to_calculate = []
    for i in range(n):
        for j in potential_pairs[i]:
            if i < j:  # Tránh tính toán trùng lặp
                pairs_to_calculate.append((i, j, cleaned_values[i], cleaned_values[j]))
    
    # Chia thành các batch và tính song song
    batch_size = max(100, len(pairs_to_calculate) // (num_workers * 10))
    all_batches = [pairs_to_calculate[i:i + batch_size] for i in range(0, len(pairs_to_calculate), batch_size)]
    
    # Xử lý song song
    results_list = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for batch_result in executor.map(process_string_batch, all_batches):
            results_list.extend(batch_result)
    
    # Điền kết quả vào ma trận tương đồng
    for i, j, similarity in results_list:
        similarity_percentages[i, j] = similarity
        similarity_percentages[j, i] = similarity  # Ma trận đối xứng
    
    # Xây dựng kết quả
    print("Đang xây dựng kết quả...")
    
    def build_result_item(i):
        # Tạo bản sao của object gốc
        obj = arr[i].copy()
        # Thêm trường similarities
        obj['similarities'] = {}
        
        for j in potential_pairs[i]:
            similarity_score = similarity_percentages[i][j]
            if similarity_score >= similarity_threshold:
                obj['similarities'][ids[j]] = {
                    'id': ids[j],
                    'value': arr[j].get('value_original', arr[j]['value']),
                    'score': float(similarity_score),
                    'check_admin': arr[j].get('check_admin', False)
                }
        
        return obj
    
    # Xử lý song song việc xây dựng kết quả
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        result = list(executor.map(build_result_item, range(n)))
    
    end_time = time.time()
    print(f"Tổng thời gian xử lý: {end_time - start_time:.2f} giây")
    
    return result

def benchmark_simhash_vs_standard(data_size=1000, str_length=30, hash_bits=64):
    """
    So sánh hiệu suất giữa SimHash và phương pháp tiêu chuẩn
    
    Args:
        data_size: Số lượng chuỗi cần so sánh
        str_length: Độ dài trung bình của mỗi chuỗi
        hash_bits: Số bit của SimHash
        
    Returns:
        Kết quả benchmark
    """
    import random
    import string
    
    # Tạo dữ liệu ngẫu nhiên
    print(f"Tạo {data_size} chuỗi ngẫu nhiên độ dài {str_length}...")
    
    def random_string(length):
        return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))
    
    strings = [random_string(random.randint(str_length // 2, str_length * 2)) for _ in range(data_size)]
    
    # Chuẩn bị dữ liệu cho API
    test_array = [
        {"id": str(i), "value": s, "value_original": s.upper(), "check_admin": i % 2 == 0}
        for i, s in enumerate(strings)
    ]
    
    # Benchmark cho SimHash
    print("\nBenchmark cho SimHash:")
    start = time.time()
    result_simhash = compare_array_simhash(test_array, hash_bits=hash_bits)
    time_simhash = time.time() - start
    print(f"  SimHash ({hash_bits} bits): {time_simhash:.4f} giây")
    
    # Benchmark cho phương pháp tiêu chuẩn
    print("\nBenchmark cho phương pháp tiêu chuẩn:")
    start = time.time()
    result_standard = compare_array_simple(test_array)
    time_standard = time.time() - start
    print(f"  Phương pháp tiêu chuẩn: {time_standard:.4f} giây")
    
    # Tính toán tăng tốc
    speedup = time_standard / time_simhash
    print(f"\nTăng tốc: {speedup:.2f}x")
    
    # Kiểm tra độ chính xác
    print("\nKiểm tra độ chính xác:")
    
    # Đếm tổng số cặp được tìm thấy bởi mỗi phương pháp
    pairs_simhash = sum(len(item['similarities']) for item in result_simhash)
    pairs_standard = sum(len(item['similarities']) for item in result_standard)
    
    print(f"  Số cặp tìm thấy bởi SimHash: {pairs_simhash}")
    print(f"  Số cặp tìm thấy bởi phương pháp tiêu chuẩn: {pairs_standard}")
    
    # Đếm số cặp chung
    common_pairs = 0
    for i in range(data_size):
        simhash_pairs = set(result_simhash[i]['similarities'].keys())
        standard_pairs = set(result_standard[i]['similarities'].keys())
        common_pairs += len(simhash_pairs.intersection(standard_pairs))
    
    # Tính độ chính xác và độ nhạy
    if pairs_standard > 0:
        precision = common_pairs / pairs_simhash if pairs_simhash > 0 else 0
        recall = common_pairs / pairs_standard
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  Độ chính xác (Precision): {precision:.4f}")
        print(f"  Độ nhạy (Recall): {recall:.4f}")
        print(f"  F1-Score: {f1_score:.4f}")
    
    return {
        "time_simhash": time_simhash,
        "time_standard": time_standard,
        "speedup": speedup,
        "pairs_simhash": pairs_simhash,
        "pairs_standard": pairs_standard,
        "common_pairs": common_pairs
    }

def simhash_optimized(text, hash_bits=64):
    """
    Phiên bản tối ưu của SimHash sử dụng vectorization.
    
    Args:
        text: Chuỗi cần tạo hash
        hash_bits: Số bit của hash (mặc định: 64)
        
    Returns:
        Giá trị hash dạng số nguyên
    """
    if not text:
        return 0
        
    # Chuyển văn bản thành chữ thường và chuẩn hóa
    text = clean_text(text)
    
    # Tạo các shingle (n-gram) từ văn bản
    n = 3  # kích thước n-gram
    if len(text) < n:
        shingles = [text]
    else:
        shingles = [text[i:i+n] for i in range(len(text) - n + 1)]
    
    # Khởi tạo vector trọng số cho các bit
    v = np.zeros(hash_bits, dtype=np.int32)
    
    # Sử dụng NumPy để vector hóa tính toán
    for shingle in shingles:
        # Sử dụng MD5 để tạo hash cho shingle
        h = hashlib.md5(shingle.encode('utf-8')).digest()
        h_int = int(binascii.hexlify(h), 16)
        
        # Vector hóa cập nhật bit
        for i in range(hash_bits):
            bitmask = 1 << i
            if h_int & bitmask:
                v[i] += 1
            else:
                v[i] -= 1
    
    # Tạo fingerprint từ các trọng số
    fingerprint = 0
    for i in range(hash_bits):
        if v[i] > 0:
            fingerprint |= (1 << i)
    
    return fingerprint

class LSHIndex:
    """
    Lớp triển khai LSH (Locality-Sensitive Hashing) cho SimHash.
    Tối ưu hóa tìm kiếm các cặp tương tự bằng cách sử dụng bảng băm.
    """
    
    def __init__(self, hash_bits=64, bands=16):
        """
        Khởi tạo LSH Index.
        
        Args:
            hash_bits: Số bit của SimHash
            bands: Số bands để chia hash
        """
        self.hash_bits = hash_bits
        self.bands = bands
        self.rows = hash_bits // bands
        self.hash_tables = [defaultdict(list) for _ in range(bands)]
        self.hashes = []
        self.items = []
    
    def add_item(self, item_idx, simhash_value):
        """
        Thêm một item vào LSH Index.
        
        Args:
            item_idx: Chỉ số của item
            simhash_value: Giá trị SimHash của item
        """
        self.hashes.append(simhash_value)
        self.items.append(item_idx)
        
        # Băm item vào các bands
        for band in range(self.bands):
            # Tạo một band hash bằng cách lấy một phần của simhash
            start_bit = band * self.rows
            end_bit = min((band + 1) * self.rows, self.hash_bits)
            
            # Tạo mask để lấy các bit trong band
            mask = 0
            for i in range(start_bit, end_bit):
                mask |= (1 << i)
            
            # Tạo band hash
            band_hash = simhash_value & mask
            
            # Thêm vào bảng băm tương ứng
            self.hash_tables[band][band_hash].append(len(self.items) - 1)
    
    def get_candidates(self, query_hash):
        """
        Tìm các ứng viên tiềm năng có SimHash tương tự.
        
        Args:
            query_hash: Giá trị SimHash cần tìm
            
        Returns:
            Set các chỉ số của các ứng viên tiềm năng
        """
        candidates = set()
        
        # Kiểm tra mỗi band
        for band in range(self.bands):
            # Tạo band hash cho query
            start_bit = band * self.rows
            end_bit = min((band + 1) * self.rows, self.hash_bits)
            
            mask = 0
            for i in range(start_bit, end_bit):
                mask |= (1 << i)
            
            band_hash = query_hash & mask
            
            # Lấy các ứng viên từ bảng băm
            candidates.update(self.hash_tables[band][band_hash])
        
        return candidates
    
    def batch_query(self, hamming_threshold):
        """
        Tìm tất cả các cặp có khoảng cách Hamming nhỏ hơn ngưỡng.
        
        Args:
            hamming_threshold: Ngưỡng khoảng cách Hamming
            
        Returns:
            Dictionary các cặp tiềm năng
        """
        potential_pairs = defaultdict(list)
        
        # Duyệt qua tất cả các item
        for i, query_hash in enumerate(self.hashes):
            # Lấy các ứng viên
            candidates = self.get_candidates(query_hash)
            
            # Kiểm tra khoảng cách Hamming
            for candidate_idx in candidates:
                j = self.items[candidate_idx]
                
                # Bỏ qua nếu là cùng một item hoặc đã kiểm tra
                if i >= j:
                    continue
                
                # Tính khoảng cách Hamming
                distance = hamming_distance(query_hash, self.hashes[candidate_idx])
                
                # Nếu khoảng cách nhỏ hơn ngưỡng, thêm vào danh sách
                if distance <= hamming_threshold:
                    potential_pairs[i].append(j)
                    potential_pairs[j].append(i)
        
        return potential_pairs

def compare_array_simhash_lsh(arr, similarity_threshold=75, num_workers=None, hash_bits=64, bands=16):
    """
    So sánh mảng các chuỗi sử dụng SimHash kết hợp với LSH để tăng tốc độ tìm kiếm.
    
    Args:
        arr: Mảng object có các trường id, value cần so sánh
        similarity_threshold: Ngưỡng độ tương đồng (0-100)
        num_workers: Số lượng worker cho multiprocessing
        hash_bits: Số bit của SimHash
        bands: Số bands cho LSH
        
    Returns:
        Mảng object với trường similarities
    """
    if num_workers is None:
        num_workers = MAX_CORES
    
    start_time = time.time()
    values = [item['value'] for item in arr]
    cleaned_values = clean_array(values)
    ids = [item['id'] for item in arr]
    n = len(arr)
    
    print(f"Bắt đầu xử lý {n} phần tử với SimHash-LSH ({hash_bits} bits, {bands} bands)...")
    
    # Tính SimHash cho mỗi chuỗi
    print("Đang tính SimHash cho các chuỗi...")
    
    # Tính song song SimHash
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        hashes = list(executor.map(lambda x: simhash_optimized(x, hash_bits), cleaned_values))
    
    # Xây dựng LSH Index
    print("Đang xây dựng LSH Index...")
    lsh_index = LSHIndex(hash_bits=hash_bits, bands=bands)
    
    # Thêm các hash vào LSH Index
    for i, h in enumerate(hashes):
        lsh_index.add_item(i, h)
    
    # Ngưỡng khoảng cách Hamming tương ứng với ngưỡng tương đồng
    hamming_threshold = int((1 - similarity_threshold/100) * hash_bits)
    
    # Tìm các cặp có khả năng tương đồng sử dụng LSH
    print("Đang tìm các cặp có khả năng tương đồng...")
    potential_pairs = lsh_index.batch_query(hamming_threshold)
    
    total_pairs = sum(len(pairs) for pairs in potential_pairs.values())
    print(f"Đã giảm từ {n*(n-1)//2} phép so sánh xuống {total_pairs} phép so sánh")
    
    # Tạo bản sao của mảng gốc để không làm thay đổi mảng gốc
    result = []
    
    if total_pairs == 0:
        print("Không tìm thấy cặp tiềm năng nào, thử giảm ngưỡng tương đồng hoặc tăng số bands")
        # Trả về mảng các object với similarities rỗng
        for i in range(n):
            obj = arr[i].copy()
            obj['similarities'] = {}
            result.append(obj)
        return result
    
    # Tính toán độ tương đồng Levenshtein chính xác cho các cặp tiềm năng
    print("Đang tính toán độ tương đồng Levenshtein chính xác...")
    
    # Sử dụng ma trận thưa (sparse matrix) để lưu trữ kết quả
    from scipy.sparse import lil_matrix
    similarity_matrix = lil_matrix((n, n))
    
    # Chuẩn bị các cặp cần tính toán
    pairs_to_calculate = []
    for i in range(n):
        for j in potential_pairs[i]:
            if i < j:  # Tránh tính toán trùng lặp
                pairs_to_calculate.append((i, j, cleaned_values[i], cleaned_values[j]))
    
    # Xử lý song song
    print(f"Đang tính toán {len(pairs_to_calculate)} cặp tiềm năng...")
    
    # Sử dụng ProcessPoolExecutor để tính toán song song
    # Chia nhỏ công việc để tránh overhead
    batch_size = max(100, len(pairs_to_calculate) // (num_workers * 10))
    batch_size = min(batch_size, 1000)  # Giới hạn kích thước batch tối đa
    
    all_batches = [pairs_to_calculate[i:i + batch_size] for i in range(0, len(pairs_to_calculate), batch_size)]
    
    # Xử lý song song với batch processing
    results_list = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for batch_result in executor.map(process_string_batch, all_batches):
            results_list.extend(batch_result)
    
    # Điền kết quả vào ma trận tương đồng
    for i, j, similarity in results_list:
        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity  # Ma trận đối xứng
    
    # Chuyển ma trận thưa sang dạng dễ truy cập
    similarity_coo = similarity_matrix.tocoo()
    
    # Xây dựng kết quả
    print("Đang xây dựng kết quả...")
    
    # Tạo dict để lưu trữ kết quả tạm thời
    similarity_dict = defaultdict(dict)
    
    # Điền giá trị từ ma trận thưa vào dict
    for i, j, v in zip(similarity_coo.row, similarity_coo.col, similarity_coo.data):
        if v >= similarity_threshold:
            similarity_dict[i][j] = v
    
    # Xây dựng kết quả cuối cùng
    def build_result_item(i):
        obj = arr[i].copy()
        obj['similarities'] = {}
        
        for j, score in similarity_dict[i].items():
            obj['similarities'][ids[j]] = {
                'id': ids[j],
                'value': arr[j].get('value_original', arr[j]['value']),
                'score': float(score),
                'check_admin': arr[j].get('check_admin', False)
            }
        
        return obj
    
    # Xử lý song song việc xây dựng kết quả
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        result = list(executor.map(build_result_item, range(n)))
    
    end_time = time.time()
    print(f"Tổng thời gian xử lý: {end_time - start_time:.2f} giây")
    
    return result

def benchmark_all_methods(data_size=1000, str_length=30, hash_bits=64, bands=16):
    """
    So sánh hiệu suất giữa tất cả các phương pháp
    
    Args:
        data_size: Số lượng chuỗi cần so sánh
        str_length: Độ dài trung bình của mỗi chuỗi
        hash_bits: Số bit của SimHash
        bands: Số bands cho LSH
        
    Returns:
        Kết quả benchmark
    """
    import random
    import string
    
    try:
        import matplotlib.pyplot as plt
        has_matplotlib = True
    except ImportError:
        has_matplotlib = False
        print("Matplotlib không có sẵn, sẽ không tạo biểu đồ")
    
    # Tạo dữ liệu ngẫu nhiên
    print(f"Tạo {data_size} chuỗi ngẫu nhiên độ dài {str_length}...")
    
    def random_string(length):
        return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))
    
    strings = [random_string(random.randint(str_length // 2, str_length * 2)) for _ in range(data_size)]
    
    # Chuẩn bị dữ liệu cho API
    test_array = [
        {"id": str(i), "value": s, "value_original": s.upper(), "check_admin": i % 2 == 0}
        for i, s in enumerate(strings)
    ]
    
    # Chạy các phiên bản khác nhau và đo thời gian
    methods = []
    times = []
    pair_counts = []
    
    # Benchmark cho phương pháp tiêu chuẩn
    print("\n1. Benchmark cho phương pháp tiêu chuẩn:")
    start = time.time()
    result_standard = compare_array_simple(test_array)
    time_standard = time.time() - start
    print(f"  Phương pháp tiêu chuẩn: {time_standard:.4f} giây")
    methods.append("Standard")
    times.append(time_standard)
    pairs_standard = sum(len(item['similarities']) for item in result_standard)
    pair_counts.append(pairs_standard)
    
    # Benchmark cho SimHash cơ bản
    print("\n2. Benchmark cho SimHash cơ bản:")
    start = time.time()
    result_simhash = compare_array_simhash(test_array, hash_bits=hash_bits)
    time_simhash = time.time() - start
    print(f"  SimHash ({hash_bits} bits): {time_simhash:.4f} giây")
    methods.append(f"SimHash-{hash_bits}")
    times.append(time_simhash)
    pairs_simhash = sum(len(item['similarities']) for item in result_simhash)
    pair_counts.append(pairs_simhash)
    
    # Benchmark cho SimHash-LSH
    print("\n3. Benchmark cho SimHash-LSH:")
    start = time.time()
    result_lsh = compare_array_simhash_lsh(test_array, hash_bits=hash_bits, bands=bands)
    time_lsh = time.time() - start
    print(f"  SimHash-LSH ({hash_bits} bits, {bands} bands): {time_lsh:.4f} giây")
    methods.append(f"SimHash-LSH-{bands}")
    times.append(time_lsh)
    pairs_lsh = sum(len(item['similarities']) for item in result_lsh)
    pair_counts.append(pairs_lsh)
    
    # So sánh tăng tốc
    print("\nSo sánh tăng tốc:")
    if time_standard > 0:
        print(f"  SimHash vs Standard: {time_standard / time_simhash:.2f}x")
        print(f"  SimHash-LSH vs Standard: {time_standard / time_lsh:.2f}x")
        print(f"  SimHash-LSH vs SimHash: {time_simhash / time_lsh:.2f}x")
    
    # So sánh số cặp tìm thấy
    print("\nSo sánh số cặp tìm thấy:")
    print(f"  Standard: {pairs_standard} cặp")
    print(f"  SimHash: {pairs_simhash} cặp")
    print(f"  SimHash-LSH: {pairs_lsh} cặp")
    
    # Tính F1 score
    def calculate_f1(result1, result2):
        common_pairs = 0
        for i in range(data_size):
            pairs1 = set(result1[i]['similarities'].keys())
            pairs2 = set(result2[i]['similarities'].keys())
            common_pairs += len(pairs1.intersection(pairs2))
        
        precision = common_pairs / sum(len(item['similarities']) for item in result1) if sum(len(item['similarities']) for item in result1) > 0 else 0
        recall = common_pairs / sum(len(item['similarities']) for item in result2) if sum(len(item['similarities']) for item in result2) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1
    
    print("\nF1 scores so với Standard:")
    p_simhash, r_simhash, f1_simhash = calculate_f1(result_simhash, result_standard)
    p_lsh, r_lsh, f1_lsh = calculate_f1(result_lsh, result_standard)
    
    print(f"  SimHash: Precision={p_simhash:.4f}, Recall={r_simhash:.4f}, F1={f1_simhash:.4f}")
    print(f"  SimHash-LSH: Precision={p_lsh:.4f}, Recall={r_lsh:.4f}, F1={f1_lsh:.4f}")
    
    # Tạo biểu đồ so sánh nếu có matplotlib
    if has_matplotlib:
        # Biểu đồ thời gian thực thi
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.bar(methods, times)
        plt.title('Thời gian thực thi (giây)')
        plt.ylabel('Thời gian (giây)')
        plt.xticks(rotation=45)
        
        # Biểu đồ số cặp tìm thấy
        plt.subplot(1, 2, 2)
        plt.bar(methods, pair_counts)
        plt.title('Số cặp tìm thấy')
        plt.ylabel('Số cặp')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('benchmark_comparison.png')
        print("\nĐã lưu biểu đồ so sánh tại benchmark_comparison.png")
    
    return {
        "methods": methods,
        "times": times,
        "pair_counts": pair_counts,
        "f1_scores": {
            "simhash": f1_simhash,
            "simhash_lsh": f1_lsh
        }
    }

if __name__ == "__main__":
    # test_check_string_similarity()
    print("\nTesting các phương pháp so sánh chuỗi:")
    benchmark_all_methods(data_size=500, str_length=30, hash_bits=64, bands=16)
    


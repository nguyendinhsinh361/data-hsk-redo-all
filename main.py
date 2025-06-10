import os
import sys
import json
import argparse
import time
from utils import common, algorithm
from tqdm import tqdm
import concurrent.futures
import numpy as np
from collections import defaultdict

DEFAULT_QUESTION_CHECK_1_INPUT_PATH = "input_all/admin_hsk_question_check_admin_1.json"
DEFAULT_QUESTION_CHECK_2_INPUT_PATH = "input_all/admin_hsk_question_check_admin_2.json"
DEFAULT_QUESTION_CHECK_3_INPUT_PATH = "input_all/admin_hsk_question_check_admin_3.json"
DEFAULT_QUESTION_CHECK_4_INPUT_PATH = "input_all/admin_hsk_question_check_admin_4.json"
DEFAULT_QUESTION_CHECK_ALL_INPUT_PATH = "input_all/admin_hsk_question.json"


def generate_value_1(tmp, break_point):
    general_text = break_point.join(tmp['general']['G_text'])
    general_text_audio = tmp['general']['G_text_audio']
    content_parts = []
    for index, tmp_content in enumerate(tmp['content']):
        q_text = f"Question {index+1}: {tmp_content['Q_text']}"
        a_text = f"Answer {index+1}: {break_point.join(tmp_content['A_text'])}"
        content_parts.append(f"{q_text}{break_point}{a_text}")
    content_text = break_point.join(content_parts)
    return f"_____GENERAL____: \n{general_text} \n_____GENERAL_AUDIO_TEXT____: \n{general_text_audio} \n_____CONTENT____: \n{content_text}"


def main():
    data_question_check_admin_1 = common.get_raw_data(DEFAULT_QUESTION_CHECK_1_INPUT_PATH)
    data_question_check_admin_2 = common.get_raw_data(DEFAULT_QUESTION_CHECK_2_INPUT_PATH)
    data_question_check_admin_3 = common.get_raw_data(DEFAULT_QUESTION_CHECK_3_INPUT_PATH)
    data_question_check_admin_4 = common.get_raw_data(DEFAULT_QUESTION_CHECK_4_INPUT_PATH)
    
    data_final = data_question_check_admin_1 + data_question_check_admin_2 + data_question_check_admin_3 + data_question_check_admin_4
    common.save_data_to_json(data_final, DEFAULT_QUESTION_CHECK_ALL_INPUT_PATH)


def preprocess_data(data_question_kind, break_point):
    """Tiền xử lý dữ liệu để tăng tốc độ xử lý sau này."""
    test_data = []
    for tmp in data_question_kind:
        # Tạo chuỗi compact cho so sánh nhanh
        compact_value = f"{''.join(tmp['general']['G_text'])}{tmp['general']['G_text_audio']}{''.join([tmp_content['Q_text'] + ''.join(tmp_content['A_text']) for tmp_content in tmp['content']])}"
        
        # Loại bỏ khoảng trắng, dấu câu và chuyển thành chữ thường
        normalized_value = ''.join(c.lower() for c in compact_value if c.isalnum())
        
        test_data.append({
            "id": tmp["id"],
            "value": normalized_value,
            "value_original": generate_value_1(tmp, break_point),
            "kind": tmp["kind"],
            "check_admin": tmp["check_admin"],
            # Tính hash để tăng tốc so sánh
            "hash": hash(normalized_value),
            # Lưu độ dài để sử dụng cho bộ lọc
            "length": len(normalized_value)
        })
    
    return test_data


def process_batch(batch, similarity_threshold, use_lsh=True, hash_bits=64, bands=16):
    """Xử lý một batch dữ liệu với thuật toán SimHash và LSH."""
    if use_lsh:
        # Sử dụng SimHash + LSH - thuật toán hiệu quả hơn cho tập dữ liệu lớn
        return algorithm.compare_array_simhash_lsh(
            batch,
            similarity_threshold=similarity_threshold,
            hash_bits=hash_bits,
            bands=bands
        )
    else:
        # Sử dụng thuật toán SimHash thông thường
        return algorithm.compare_array_simhash(
            batch,
            similarity_threshold=similarity_threshold
        )


def parallel_process_batches(test_data, batch_size, similarity_threshold, use_lsh=True):
    """Xử lý song song các batch dữ liệu."""
    num_items = len(test_data)
    if num_items <= batch_size:
        # Nếu số lượng nhỏ hơn kích thước batch, xử lý toàn bộ
        return process_batch(test_data, similarity_threshold, use_lsh)
    
    # Chia dữ liệu thành các batch
    batches = [test_data[i:i + batch_size] for i in range(0, num_items, batch_size)]
    
    # Xác định số lượng worker hợp lý dựa trên CPU
    num_workers = min(len(batches), os.cpu_count() or 4)
    
    results = []
    # Xử lý song song các batch
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_batch = {
            executor.submit(process_batch, batch, similarity_threshold, use_lsh): i 
            for i, batch in enumerate(batches)
        }
        
        # Thu thập kết quả từ các batch
        for future in tqdm(concurrent.futures.as_completed(future_to_batch), 
                          total=len(batches),
                          desc="Xử lý các batch"):
            batch_index = future_to_batch[future]
            try:
                batch_result = future.result()
                results.extend(batch_result)
            except Exception as e:
                print(f"Lỗi xử lý batch {batch_index}: {e}")
    
    return results


def prefilter_by_length(test_data, threshold=0.7):
    """Lọc sơ bộ dựa trên độ dài chuỗi để giảm số lượng so sánh."""
    n = len(test_data)
    potential_pairs = []
    
    # Sắp xếp theo độ dài để tối ưu hóa lọc
    sorted_data = sorted(enumerate(test_data), key=lambda x: x[1]["length"])
    
    for i in range(n):
        idx, item = sorted_data[i]
        length = item["length"]
        min_length = int(length * threshold)
        max_length = int(length / threshold)
        
        # Tìm các phần tử có độ dài phù hợp
        j = i + 1
        while j < n and sorted_data[j][1]["length"] <= max_length:
            if sorted_data[j][1]["length"] >= min_length:
                potential_pairs.append((idx, sorted_data[j][0]))
            j += 1
    
    return potential_pairs


def test_compare_array_simple(data_question, output_path, use_optimized=True, batch_size=100, 
                              similarity_threshold=75, kind_filter=None, use_lsh=True):
    kind_list = ["110001", "110002", "110003", "110004", "120001", "120002", "120003", "120004", 
                "210001", "210002", "210003", "210004", "220001", "220002", "220003", "220004", 
                "310001", "310002", "310003", "310004", "320001", "320002", "320003", "330001", 
                "330002", "410001", "410002", "410003_1", "410003_2", "420001", "420002", 
                "420003_1", "420003_2", "430001", "430002", "510001", "510002_1", "510002_2", 
                "520001", "520002", "520003", "530001", "530002", "530003", "610001", "610002", 
                "610003", "620001", "620002", "620003", "620004", "630001"]
    
    # Lọc danh sách kind nếu có chỉ định
    if kind_filter:
        kind_list = [k for k in kind_list if k in kind_filter]
    
    break_point = "\n"
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Đo thời gian xử lý
    start_time = time.time()
    
    for kind in tqdm(kind_list, desc="Xử lý các loại câu hỏi"):
        data_question_kind = [tmp for tmp in data_question if tmp["kind"] == kind]
        
        if not data_question_kind:
            print(f"Không có dữ liệu cho loại: {kind}")
            continue
        
        print(f"\nĐang xử lý loại {kind} với {len(data_question_kind)} câu hỏi...")
        
        # Tiền xử lý dữ liệu
        kind_start_time = time.time()
        test_data = preprocess_data(data_question_kind, break_point)
        
        # Nếu số lượng câu hỏi quá ít, sử dụng thuật toán đơn giản
        if len(test_data) <= 10:
            if use_optimized:
                # Sử dụng thuật toán SimHash cho dataset nhỏ
                data = algorithm.compare_array_simhash(
                    test_data,
                    similarity_threshold=similarity_threshold
                )
            else:
                # Sử dụng thuật toán cũ nếu không muốn tối ưu
                data = algorithm.compare_array_simple(
                    test_data,
                    use_optimized=False,
                    batch_size=batch_size,
                    similarity_threshold=similarity_threshold
                )
        else:
            # Sử dụng xử lý song song cho dataset lớn
            data = parallel_process_batches(
                test_data,
                batch_size=batch_size,
                similarity_threshold=similarity_threshold,
                use_lsh=use_lsh
            )
        
        kind_end_time = time.time()
        
        # Lọc dữ liệu kết quả
        data = [{
            "id": tmp["id"],
            "value": tmp["value_original"],
            "kind": tmp["kind"],
            "check_admin": tmp["check_admin"],
            "similarities": tmp["similarities"],
        } for tmp in data if tmp["similarities"]]
        
        # Lưu kết quả
        common.save_data_to_json(data, f"{output_path}/{kind}.json")
        
        processing_time = kind_end_time - kind_start_time
        print(f"Đã hoàn thành xử lý loại {kind} trong {processing_time:.2f} giây")
        print(f"Số lượng kết quả trùng lặp: {len(data)}/{len(test_data)}")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nHoàn thành toàn bộ xử lý trong {total_time:.2f} giây")


if __name__ == "__main__":
    # Cấu hình tham số dòng lệnh
    parser = argparse.ArgumentParser(description='So sánh độ tương đồng giữa các câu hỏi HSK')
    parser.add_argument('--mode', choices=['all', '1', '2', '3', '4'], default='all',
                        help='Chọn bộ dữ liệu để xử lý (mặc định: all)')
    parser.add_argument('--output', type=str, default='output',
                        help='Thư mục đầu ra (mặc định: output)')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Kích thước batch cho xử lý (mặc định: 100)')
    parser.add_argument('--threshold', type=int, default=75,
                        help='Ngưỡng độ tương đồng (0-100, mặc định: 75)')
    parser.add_argument('--no-optimize', action='store_true',
                        help='Tắt tối ưu hóa (sử dụng thuật toán gốc)')
    parser.add_argument('--no-lsh', action='store_true',
                        help='Không sử dụng LSH (Locality Sensitive Hashing)')
    parser.add_argument('--kind', type=str, nargs='+',
                        help='Chỉ xử lý các loại câu hỏi cụ thể (ví dụ: 110001 210003)')
    parser.add_argument('--combine', action='store_true',
                        help='Kết hợp dữ liệu trước khi xử lý')
    
    args = parser.parse_args()
    
    # Xử lý tham số
    use_optimized = not args.no_optimize
    use_lsh = not args.no_lsh
    batch_size = args.batch_size
    similarity_threshold = args.threshold
    output_base = args.output
    mode = args.mode
    
    print(f"=== SO SÁNH ĐỘ TƯƠNG ĐỒNG GIỮA CÁC CÂU HỎI HSK ===")
    print(f"Chế độ: {mode}")
    print(f"Tối ưu hóa: {'Tắt' if args.no_optimize else 'Bật'}")
    print(f"Sử dụng LSH: {'Tắt' if args.no_lsh else 'Bật'}")
    print(f"Kích thước batch: {batch_size}")
    print(f"Ngưỡng tương đồng: {similarity_threshold}%")
    
    if args.kind:
        print(f"Chỉ xử lý loại: {', '.join(args.kind)}")
    
    # Kết hợp dữ liệu nếu cần
    if args.combine:
        print("\nĐang kết hợp dữ liệu từ tất cả các nguồn...")
        main()
    
    # Xử lý dữ liệu theo chế độ
    if mode == 'all' or args.combine:
        print("\nĐang xử lý tất cả dữ liệu...")
        data_question_all = common.get_raw_data(DEFAULT_QUESTION_CHECK_ALL_INPUT_PATH)
        test_compare_array_simple(
            data_question_all, 
            f"{output_base}/data_question_check_admin_all",
            use_optimized=use_optimized,
            batch_size=batch_size,
            similarity_threshold=similarity_threshold,
            kind_filter=args.kind,
            use_lsh=use_lsh
        )
    else:
        data_path = f"DEFAULT_QUESTION_CHECK_{mode}_INPUT_PATH"
        output_folder = f"{output_base}/data_question_check_admin_{mode}"
        
        print(f"\nĐang xử lý dữ liệu từ nguồn {mode}...")
        data_question = common.get_raw_data(globals()[data_path])
        test_compare_array_simple(
            data_question, 
            output_folder,
            use_optimized=use_optimized,
            batch_size=batch_size,
            similarity_threshold=similarity_threshold,
            kind_filter=args.kind,
            use_lsh=use_lsh
        )
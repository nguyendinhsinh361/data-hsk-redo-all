# So Sánh Độ Tương Đồng Giữa Các Câu Hỏi HSK

Dự án này cung cấp công cụ để phát hiện và đánh giá độ tương đồng giữa các câu hỏi HSK, sử dụng thuật toán Levenshtein Distance đã được tối ưu hóa với PyTorch và NumPy.

## Tính Năng

- Tối ưu hóa thuật toán Levenshtein Distance sử dụng:
  - PyTorch với GPU (nếu có)
  - NumPy cho vector hóa trên CPU
  - Xử lý batch để tăng hiệu suất với dữ liệu lớn
  - Bộ nhớ đệm (caching) để tránh tính toán lặp lại
- Hỗ trợ xử lý dữ liệu lớn với nhiều loại câu hỏi
- Tùy chỉnh ngưỡng độ tương đồng, kích thước batch, và loại câu hỏi
- Hiển thị tiến trình và thống kê thời gian xử lý

## Cài Đặt

1. Cài đặt các thư viện cần thiết:

```bash
pip install numpy torch tqdm matplotlib
```

2. Sao chép mã nguồn:

```bash
git clone <đường-dẫn-repo>
cd <thư-mục-dự-án>
```

## Sử Dụng

### Chạy từ dòng lệnh

Để chạy chương trình với các tham số mặc định:

```bash
python main.py
```

### Các tùy chọn

- `--mode`: Chọn bộ dữ liệu để xử lý (`all`, `1`, `2`, `3`, `4`)
- `--output`: Thư mục đầu ra (mặc định: `output`)
- `--batch-size`: Kích thước batch cho xử lý (mặc định: 100)
- `--threshold`: Ngưỡng độ tương đồng (0-100, mặc định: 75)
- `--no-optimize`: Tắt tối ưu hóa (sử dụng thuật toán gốc)
- `--kind`: Chỉ xử lý các loại câu hỏi cụ thể (ví dụ: `110001 210003`)
- `--combine`: Kết hợp dữ liệu trước khi xử lý

### Ví dụ

```bash
# Xử lý tất cả dữ liệu với tối ưu hóa
python main.py --mode all

# Xử lý một loại dữ liệu cụ thể với ngưỡng độ tương đồng 80%
python main.py --mode 1 --threshold 80

# Chỉ xử lý một số loại câu hỏi cụ thể
python main.py --kind 110001 210003

# Sử dụng kích thước batch lớn hơn cho dữ liệu lớn
python main.py --batch-size 200

# Kết hợp tất cả dữ liệu và xử lý
python main.py --combine
```

## Benchmark

Để đánh giá hiệu suất của thuật toán với các kích thước dữ liệu khác nhau:

```bash
python benchmark.py
```

Các tùy chọn benchmark:

- `--quick`: Chạy benchmark nhanh với ít dữ liệu hơn
- `--full`: Chạy benchmark đầy đủ với dữ liệu lớn hơn (chạy lâu)
- `--max-size`: Kích thước tối đa của bộ dữ liệu (mặc định: 400)
- `--steps`: Số lượng kích thước khác nhau cần kiểm tra (mặc định: 4)

## Cấu Trúc Thư Mục

- `main.py`: Tập tin chính để chạy chương trình
- `benchmark.py`: Tập tin để đánh giá hiệu suất thuật toán
- `utils/`: Thư mục chứa các module tiện ích
  - `algorithm.py`: Mô-đun chứa thuật toán Levenshtein Distance đã tối ưu
  - `common.py`: Các hàm tiện ích dùng chung
- `input_all/`: Thư mục chứa dữ liệu đầu vào
- `output/`: Thư mục chứa kết quả đầu ra

## Tối Ưu Hóa

Thuật toán đã được tối ưu hóa bằng cách:

1. Sử dụng GPU thông qua PyTorch nếu có
2. Sử dụng vector hóa NumPy khi không có GPU
3. Xử lý batch để tránh hết bộ nhớ và tối ưu hiệu suất
4. Sử dụng bộ nhớ đệm để tránh tính toán lặp lại
5. Xử lý song song và phân mảnh ma trận tương đồng cho dữ liệu lớn 
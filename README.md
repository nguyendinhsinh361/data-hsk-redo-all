# ID Counter for Excel File

Script để đếm số lượng ID trong từng sheet của file Excel "Check Dữ Liệu Câu Hỏi Trùng Lặp Các Dạng HSK.xlsx".

## Cài đặt

1. Cài đặt Python (phiên bản 3.6 trở lên)
2. Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

## Sử dụng

Chạy script với lệnh:

```bash
python count_id_checks.py
```

Script sẽ tự động tìm file Excel "Check Dữ Liệu Câu Hỏi Trùng Lặp Các Dạng HSK.xlsx" trong thư mục con "excel" hoặc theo đường dẫn tuyệt đối. Nếu không tìm thấy, script sẽ yêu cầu bạn nhập đường dẫn đầy đủ đến file.

## Kết quả

Script sẽ hiển thị:

1. Số lượng ID trong từng sheet
2. Thông tin chi tiết về mỗi sheet:
   - Tên cột chứa ID
   - Tổng số ID
   - Số lượng ID duy nhất
   - Mẫu một số ID đầu tiên

## Lưu ý

Script này tìm kiếm cột ID dựa trên tên cột phổ biến như 'ID', 'Mã', 'Mã câu hỏi', v.v. Nếu không tìm thấy, script sẽ sử dụng cột đầu tiên làm cột ID mặc định. 
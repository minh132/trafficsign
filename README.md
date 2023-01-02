# Nhận dạng biển báo
Họ tên sinh viên:Trịnh Xuân Minh  
MSSV:20204589

Về dữ liệu em sẽ sử dụng bộ dữ liệu biển báo giao thông nổi tiếng đó là German Traffic Sign:[Link tải](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html/) . Bộ dữ liệu này gồm khoảng gần 40k ảnh chia thành 43 folder là 43 loại biển báo khác nhau. Dữ liệu sau khi được tải về được giải nén sẽ đưa vào thư mục data.  
Phần source code bao gồm 5 file:
+ utils.py
+ dataset.py
+ model.py
+ train.py
+ test.py  

File dataset.py để đọc và xử lí dữ liệu. Từ tập dữ liệu train em chia thành tập validation với tỉ lệ 10%. Dữ liệu sẽ được chuẩn hóa và biến đổi thành các tensor.   
Em sử dụng mô hình pretrained MobileNetV3 Large bởi độ chính xác và hiệu năng tính toán.  
File util.py được sử dụng để lưu model đã train và lưu lại biểu đồ loss và accuracy .  
File train.py được sử dụng để trai model. Ví dụ để thực hiện train với 30 epochs và learningrate=0.0001 nhập câu lệnh vào terminal: 
```shell
python train.py --pretrained --fine-tune --epochs 30 --learning-rate 0.0001.
```
File test được sử đụng để kiểm tra độ chính xác .Nhập câu lệnh vào terminal:
```shell
python py
```
Kết quả dự đoán đúng 12419 trên tổng số 

# Model pretrained và hợp đồng tensor

Đã đọc README của [reference ONNX](https://github.com/ibaiGorordo/onnx-Ultra-Fast-Lane-Detection-Inference) và [upstream v2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2). Reference ONNX liên kết đến **UFLD v1**, không phải v2. Bản đầu của LaneGuard dùng v1 cho pipeline nhỏ và nhất quán; không hỗ trợ trọng số v2 hoặc `.pth`.

Các thông số bên dưới lấy từ [mã inference tham chiếu](https://github.com/ibaiGorordo/onnx-Ultra-Fast-Lane-Detection-Inference/blob/main/ultrafastLaneDetector/ultrafastLaneDetector.py) và [script chuyển đổi PINTO](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/140_Ultra-Fast-Lane-Detection/convert_script.txt). README reference có dòng mô tả kích thước khác; adapter theo mã thực thi và script chuyển đổi, đồng thời kiểm tra metadata khi nạp.

| Thuộc tính | Hợp đồng |
| --- | --- |
| Input | float32 NCHW `[1,3,288,800]`, RGB, resize toàn ảnh |
| Chuẩn hóa | `(RGB / 255 - [0.485,0.456,0.406]) / [0.229,0.224,0.225]` |
| TuSimple output | `[1,101,56,4]` hoặc `[101,56,4]` logits |
| CULane output | `[1,201,18,4]` hoặc `[201,18,4]` logits |
| Giải mã | Đảo chiều anchor, softmax trên grid không-background, kỳ vọng chỉ số grid bắt đầu từ 1; background argmax là điểm vắng |
| Anchors TuSimple | 64..284, bước 4, trên hệ quy chiếu chiều cao 288 |
| Anchors CULane | Mảng 18 hàng trong mã tham chiếu và `detector.py` |

Adapter chuyển tọa độ thẳng về kích thước ảnh gốc thay vì resize overlay về kích thước dataset. Giữ điểm float để nội suy; bổ sung gate xác suất không-background và loại điểm ngoài ảnh. Không dùng SciPy vì NumPy đủ để tính softmax ổn định. Các gate này là heuristic, không phải xác suất cảnh báo đã hiệu chuẩn.

## Lấy trọng số TuSimple

Nguồn chuyển đổi được reference dẫn tới là [PINTO model zoo, mục 140](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/140_Ultra-Fast-Lane-Detection). URL dưới đây được xác minh trong [download_tusimple.sh](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/140_Ultra-Fast-Lane-Detection/download_tusimple.sh), không phải URL tự suy đoán.

Gói tải là khoảng **6,4 GiB**, chứa nhiều định dạng không cần dùng. Chuẩn bị dung lượng và đường truyền; có thể dùng trình tải của bạn để lấy gói rồi chạy script trích xuất. Script chỉ ghi một file ONNX, không giải nén toàn bộ hoặc ghi đường dẫn tùy ý từ archive.

```bash
mkdir -p models
curl -L --fail --retry 3 --continue-at - 'https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/140_Ultra-Fast-Lane-Detection/resources_tusimple.tar.gz' -o /tmp/laneguard-tusimple.tar.gz
python tools/extract_model.py /tmp/laneguard-tusimple.tar.gz
python -m laneguard --image /path/to/road.jpg --headless --output runs/image.jpg
```

Đích mặc định: `models/ultra_falst_lane_detection_tusimple_288x800.onnx`. Script không ghi đè model có sẵn. Nếu đã có ONNX v1 tương thích, dùng `--model /path/to/model.onnx` và `--dataset` tương ứng. Nếu không tìm được file, hãy kiểm tra archive và danh sách `.onnx` từ nguồn; không đổi tên model v2 để ép chạy v1. Kiểm tra tensor không chứng minh nguồn gốc hoặc chất lượng trọng số: cần kiểm tra ảnh đường thực sau khi lấy đúng pretrained model.

Không có checksum được xác minh từ nguồn trong phiên này. Khi thu thập bằng chứng, ghi `sha256sum models/*.onnx`, URL và ngày tải để nhận diện đúng artifact. Không thêm trọng số vào Git. Không có model pretrained hoàn chỉnh nào được lấy hoặc chạy trong phiên triển khai hiện tại; tải thử đã timeout sau 120 giây.

## Giấy phép

Mã reference có MIT, copyright 2021 Ibai Gorordo. [Giấy phép model nguồn do PINTO đính kèm](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/140_Ultra-Fast-Lane-Detection/LICENSE) có MIT, copyright 2020 cfzd. Bảo toàn thông báo trong `THIRD_PARTY_NOTICES.md`; không gán lại quyền sở hữu trọng số. Dữ liệu/video dùng đánh giá cần tuân theo điều kiện riêng của nguồn dữ liệu. Không suy diễn giấy phép mã thành giấy phép cho mọi bộ dữ liệu.

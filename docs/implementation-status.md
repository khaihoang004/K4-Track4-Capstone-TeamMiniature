# Trạng thái triển khai — 16/10/2026

## Phát hiện ban đầu

Kho chưa có commit, chỉ có `writing-block.md` và các thư mục cấu hình môi trường. Không có README, dependency, pipeline webcam hay model để tái sử dụng; không tìm thấy `AGENTS.md`. Dùng Python 3.10.12. File yêu cầu ban đầu được giữ nguyên; chưa tạo commit.

## Đã thực hiện

- `laneguard/detector.py`: adapter UFLD **v1**, tiền/hậu xử lý theo reference đã đọc, kiểm tra tensor, lỗi model và fallback CPU khi provider không có hoặc khởi tạo thất bại.
- `laneguard/geometry.py`: biên trái/phải, nội suy cục bộ trên đường cong, kiểm tra tính hợp lệ và offset chuẩn hóa.
- `laneguard/monitor.py`: ba trạng thái, EMA, persistence, hysteresis và reset khi mất độ tin cậy.
- `laneguard/source.py`: webcam giữ khung mới nhất, video tuần tự, timeout/retry và giải phóng nguồn.
- `laneguard/overlay.py`, `__main__.py`, `__init__.py`: entry point, overlay, ảnh đơn/video/webcam, CSV, metadata và số liệu runtime.
- `tools/evaluate.py`: đánh giá nhãn thủ công, confusion matrix, precision/recall và reliable coverage.
- `tools/extract_model.py`: trích riêng ONNX từ gói model, không ghi đường dẫn tùy ý trong archive.
- `tests/`: kiểm thử hình học/thời gian, detector/source, tích hợp video và đánh giá, trích model.
- `requirements.txt`, `requirements-dev.txt`, `.gitignore`, `README.md`, `THIRD_PARTY_NOTICES.md`, `docs/model.md`, `docs/evaluation.md` và tài liệu trạng thái này.

## Model và trọng số

Đã chọn adapter TuSimple UFLD v1 theo kho ONNX tham chiếu và model chuyển đổi PINTO mục 140. Hỗ trợ thêm hợp đồng tensor CULane v1. Không nhầm pipeline v1 này với upstream v2 trong đề bài.

**Chưa sử dụng trọng số pretrained nào để chạy inference.** URL tải chính thức phản hồi gói 6.875.285.956 byte; lần tải có giới hạn 120 giây kết thúc bằng curl exit 28, chỉ nhận 315.702.776 byte. Gói chưa hoàn chỉnh nằm tại `/tmp/laneguard-tusimple.tar.gz`, có thể tiếp tục bằng lệnh curl trong `docs/model.md`. Không có ONNX hoàn chỉnh trong repository.

## Kiểm chứng thực tế

Môi trường ảo độc lập `.venv` đã cài theo `requirements-dev.txt`:

| Thành phần | Phiên bản dùng kiểm thử |
| --- | --- |
| Python | 3.10.12 |
| NumPy | 2.2.6 |
| OpenCV | 4.14.0.94 (runtime 4.14.0) |
| ONNX Runtime | 1.23.2 |
| pytest | 9.1.1 |

Các lệnh đã chạy:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q laneguard tools tests
.venv/bin/python -m pip check
.venv/bin/python -m laneguard --help
.venv/bin/python tools/evaluate.py --help
.venv/bin/python tools/extract_model.py --help
```

- Lần kiểm thử cuối: **27 passed**, 0 failed, 0,37 giây. Thời gian này là thời gian test, không phải latency inference.
- Compileall: thành công. Pip check: `No broken requirements found`.
- CLI help: thành công. Chạy CLI thiếu model trả exit 1 kèm hướng dẫn rõ ràng.
- File ONNX hỏng thử bằng ONNX Runtime thật: exit 1, báo `INVALID_PROTOBUF` qua thông báo lỗi ứng dụng.
- `CameraSource(0)` thử thực tế: báo không mở được webcam. Môi trường không có `/dev/video*`; không chạy live webcam inference.
- Test tích hợp dùng video MJPEG tổng hợp 8 frame và detector mô phỏng, đi qua đọc video thật, hình học, cảnh báo, mất làn/phục hồi, overlay, CSV, ảnh cuối và đánh giá. Đây không phải đánh giá chất lượng model.
- Provider thật khả dụng: `AzureExecutionProvider`, `CPUExecutionProvider`. Fallback GPU được kiểm thử bằng mock; không xác minh GPU thật.

Không có FPS/latency pretrained, clip đường thực hay precision/recall thực tế để báo cáo. Không lấy số đo của detector mô phỏng làm benchmark.

## Các bước chạy tiếp

Trong thư mục repository:

```bash
source .venv/bin/activate
# Nếu dựng lại môi trường: python3 -m venv .venv rồi pip install -r requirements-dev.txt

# Tải tiếp gói trọng số và trích đúng ONNX theo docs/model.md trước.
python tools/extract_model.py /tmp/laneguard-tusimple.tar.gz

# Dùng ảnh đường có sẵn để xác minh inference/preprocessing và quan sát overlay.
python -m laneguard --image /path/to/road.jpg --headless --output runs/image.jpg --log runs/image.csv

# Máy có webcam và desktop:
python -m laneguard --camera 0 --log runs/webcam.csv

# Hoặc video có sẵn:
python -m laneguard --video /path/to/road.mp4 --headless --max-frames 300 --log runs/clear.csv --output runs/last.jpg
```

Lệnh tải đầy đủ và nguồn đã xác minh ở [model.md](model.md). Thay các đường dẫn mẫu bằng dữ liệu của bạn. Cần kết nối webcam trên máy có desktop hoặc cung cấp video/ảnh đường có sẵn. Sau khi model chạy, gán nhãn và chạy [quy trình đánh giá](evaluation.md); ghi một failure case thực tế và số liệu phần cứng/cấu hình. Giới hạn còn lại gồm kiểm chứng trọng số, chất lượng nhận diện trên cảnh thật, webcam/GUI/GPU và hiệu chỉnh ngưỡng theo góc đặt camera. Không dùng cho lái xe thực tế.

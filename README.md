# LaneGuard AI

Prototype Python nhận diện làn bằng **UFLD v1 pretrained + ONNX Runtime**, ước lượng độ lệch tương đối trên ảnh và hiển thị cảnh báo. Hỗ trợ webcam, video có sẵn và ảnh đơn. Không huấn luyện model.

Đây là demo nghiên cứu, không phải hệ thống ADAS hoàn chỉnh, không có chứng nhận an toàn và **không dùng để hỗ trợ lái xe thực tế**. Không ước lượng động lực học hoặc khoảng cách theo mét.

## Cài đặt và chạy

Môi trường mục tiêu: Linux desktop, Python 3.10 trở lên. Khi dùng webcam cần quyền truy cập camera; cửa sổ OpenCV cần phiên đồ họa. Máy chủ không có màn hình dùng `--headless`. Các nền tảng khác chưa được kiểm thử.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Nếu Ubuntu thiếu module `venv`, cài gói hệ thống `python3-venv`. Máy desktop có lỗi `libGL.so.1` cần gói `libgl1`. Chỉ chạy ứng dụng thì có thể cài `requirements.txt` thay cho `requirements-dev.txt`.

**Cần lấy trọng số trước khi chạy:** xem [hướng dẫn model](docs/model.md). File mặc định:

```text
models/ultra_falst_lane_detection_tusimple_288x800.onnx
```

Tên `falst` là cách viết của nguồn chuyển đổi, không sửa tên khi dùng lệnh mặc định. Ứng dụng không tự tải trọng số lúc khởi động.

Kiểm tra suy luận ảnh đơn trước:

```bash
python -m laneguard --image /path/to/road.jpg --headless --output runs/image.jpg --log runs/image.csv
```

Chạy webcam (Q hoặc Escape để thoát, Ctrl+C cũng được):

```bash
python -m laneguard --camera 0 --log runs/webcam.csv
```

Chạy video có sẵn:

```bash
python -m laneguard --video /path/to/road.mp4 --log runs/clear.csv
python -m laneguard --video /path/to/road.mp4 --headless --max-frames 300 --log runs/clear.csv --output runs/last.jpg
```

Video xử lý tuần tự đến hết, không ép tốc độ phát bằng FPS gốc. Webcam dùng một luồng đọc và một ô giữ khung hình mới nhất, thay thế khung cũ khi suy luận chậm. `CAP_PROP_BUFFERSIZE=1` chỉ là yêu cầu best effort cho backend. Webcam quay màn hình đang phát dashcam là **mô phỏng**, cần ghi rõ trong báo cáo.

## Kiến trúc

```text
CameraSource / ảnh → LaneDetector (ONNX) → LaneGeometry
                                           ↓
                               DepartureMonitor (EMA + persistence)
                                           ↓
                         OverlayRenderer + CSV + số liệu runtime
```

Mã nguồn nằm trong `laneguard/`: `source.py` quản lý nguồn, `detector.py` tiền/hậu xử lý và model, `geometry.py` tính hình học, `monitor.py` giữ trạng thái thời gian, `overlay.py` vẽ và `__main__.py` điều phối. Khi thoát, nguồn và file log được đóng. Nếu backend webcam bị treo trong lệnh đọc native, worker sẽ giải phóng camera khi lệnh đọc trả về; không gọi `release()` đồng thời với `read()`.

## Hình học và trạng thái

UFLD trả bốn slot làn; chỉ dùng slot 1 và 2 làm biên trái/phải của làn hiện tại. Không lấy làn kế bên thay thế biên bị mất. Nội suy hai điểm bao quanh hàng quan sát, giữ được độ cong cục bộ; không ngoại suy. Mỗi biên cần ít nhất ba điểm, các hàng không trùng, khoảng trống quanh hàng quan sát không vượt 10% chiều cao ảnh. Biên phải nằm trong ảnh và độ rộng phải hợp lệ.

```text
lane_center = (left + right) / 2
offset = (reference_x - lane_center) / ((right - left) / 2)
```

**Dương = điểm tham chiếu camera nằm bên phải tâm làn; âm = bên trái.** Đây là độ lệch chuẩn hóa theo nửa chiều rộng làn tại hàng quan sát, không phải vị trí xe đã hiệu chuẩn. Camera mặc định được giả định đặt ở giữa, không lật gương ảnh.

| Trạng thái | Điều kiện |
| --- | --- |
| `NORMAL` | Hình học hợp lệ, chưa đủ điều kiện phát cảnh báo; có thể đang đếm `Pending`. |
| `LANE DEPARTURE WARNING` | Cả độ lệch gốc và EMA vượt ngưỡng, cùng hướng trong đủ số khung liên tiếp. |
| `LANE NOT RELIABLE` | Thiếu biên, thiếu điểm tại hàng quan sát, hình học vô lý hoặc đầu ra không hữu hạn. Xóa EMA, bộ đếm và cảnh báo cũ ngay. |

Cảnh báo giữ đến khi cả độ lệch gốc và EMA xuống dưới ngưỡng xóa. Đổi hướng trực tiếp sẽ khởi tạo lại bộ đếm. Số khung tồn tại tính theo **khung đã xử lý**, vì vậy thời gian phản ứng thay đổi theo tốc độ suy luận. Không cảnh báo khi chỉ có một khung nhiễu.

## Cấu hình

`python -m laneguard --help` liệt kê toàn bộ tùy chọn.

| Tùy chọn | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `--model` | Đường dẫn TuSimple ở trên | File ONNX cục bộ |
| `--dataset` | `tusimple` | `tusimple` hoặc `culane`, phải khớp model v1 |
| `--camera` | `0` | Chỉ số webcam; loại trừ `--video`/`--image` |
| `--provider` | `CPUExecutionProvider` | Provider ONNX Runtime yêu cầu |
| `--threshold` / `--clear-threshold` | `0.65` / `0.50` | Ngưỡng phát/xóa cảnh báo |
| `--persistence` | `5` | Số khung liên tiếp, tối thiểu 2 |
| `--alpha` | `0.35` | Trọng số mẫu mới trong EMA |
| `--lookahead` | `0.80` | Hàng quan sát theo tỷ lệ chiều cao |
| `--reference` | `0.50` | Điểm tham chiếu theo tỷ lệ chiều rộng |
| `--min-width` / `--max-width` | `0.15` / `0.90` | Giới hạn chiều rộng làn theo chiều rộng ảnh |
| `--min-lane-probability` | `0.50` | Ngưỡng xác suất không-background, không phải confidence đã hiệu chuẩn |
| `--max-frames` | `0` | 0 = đến EOF hoặc thoát |
| `--headless` | Tắt | Không mở cửa sổ |
| `--log` / `--output` | Không ghi | CSV số liệu / ảnh cuối có overlay |

CPU là lựa chọn mặc định. Với môi trường CUDA được cấu hình đúng, thay gói `onnxruntime` bằng `onnxruntime-gpu` tương thích với CUDA/cuDNN của máy, không cài đồng thời cả hai, rồi dùng `--provider CUDAExecutionProvider`. Cài gói GPU không đảm bảo chạy GPU. Ứng dụng ghi provider yêu cầu và danh sách provider của session; nếu provider không khả dụng thì báo và chuyển về CPU. Danh sách session không chứng minh mọi node đều chạy GPU. Chưa xác minh GPU trên môi trường hiện tại.

## Hiệu năng và bằng chứng

Overlay và CSV dùng số đo `perf_counter()`, không dùng FPS cố định:

- `inference_ms`: thời gian gọi `session.run()`; không gồm tiền/hậu xử lý.
- `latency_ms`: đọc/lấy khung + tiền xử lý + suy luận + hình học/cảnh báo + vẽ overlay chính. Không gồm ghi CSV, lưu ảnh, `imshow`/`waitKey` hoặc thời gian exposure/driver của camera.
- `frame_age_ms`: từ lúc đọc xong khung tới overlay; giúp theo dõi độ cũ của khung webcam, không phải độ trễ vật lý từ cảm biến tới màn hình.
- `fps`: tốc độ xử lý trung bình qua các mốc hoàn thành trong cửa sổ khoảng hai giây; khung đầu là 0 vì chưa có khoảng thời gian để tính.
- JSON cuối phiên có FPS toàn phiên (gồm thời gian ghi log/hiển thị) và latency xử lý trung bình. Dùng video headless để so sánh hiệu năng, không dùng ảnh đơn có thời gian chờ phím.

CSV kèm file `.metadata.json` ghi cấu hình và provider; `source_frame` webcam có thể nhảy số do chủ động bỏ khung cũ. File cùng tên sẽ bị ghi đè ở lần chạy tiếp theo. Quy trình gán nhãn và đánh giá ở [docs/evaluation.md](docs/evaluation.md). Trạng thái kiểm chứng thực tế ở [docs/implementation-status.md](docs/implementation-status.md).

## Kiểm thử

```bash
python -m pytest -q
python -m compileall -q laneguard tools tests
```

Unit test dùng logits/detections giả lập; test tích hợp tạo video tổng hợp rồi chạy đọc video, hình học, cảnh báo, overlay, CSV và đánh giá. Không cần camera/trọng số để chạy bộ test. Những test này không đo chất lượng model pretrained.

## Track 4 và giới hạn

- **Geometry:** biên trái/phải, tâm làn, nội suy tại hàng quan sát, offset chuẩn hóa.
- **Time:** EMA, đếm khung cùng hướng, hysteresis và reset khi mất độ tin cậy.
- **Uncertainty:** trạng thái không đáng tin cậy và các điều kiện chặn hình học.
- **Evidence:** log số đo thật, metadata, script đánh giá nhãn và test lỗi. Precision/recall trên đường thực vẫn đang chờ dữ liệu.

Camera đặt lệch/nghiêng, quay gương, giao lộ, chuyển làn, đường cong gắt, vạch mờ, bóng đổ, mặt đường lạ và màn hình phản chiếu đều có thể gây sai. UFLD nhận diện vạch, không xác nhận vạch đó là ranh giới làn xe đang chạy. Gate hình học không phát hiện được mọi dự đoán sai nhưng có vẻ hợp lý. Không có hiệu chuẩn camera, dự báo chuyển động, turn-signal suppression hay xử lý điều khiển xe.

Nguồn model, khác biệt v1/v2 và giấy phép được ghi trong [docs/model.md](docs/model.md) và [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Trọng số, video cá nhân, môi trường ảo và kết quả sinh ra không được thêm vào Git.

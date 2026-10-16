# Thu thập bằng chứng

Kết quả đường thực hiện **chưa có**. Không dùng output test tổng hợp làm benchmark pretrained.

## Quy trình tái lập

1. Chuẩn bị tối thiểu năm clip đại diện: vạch rõ, đường cong, vạch mờ/che khuất, bóng đổ/vạch gây nhầm và mất một/hai biên. Ghi tên nguồn, quyền sử dụng, độ phân giải, FPS gốc và đây là video trực tiếp hay webcam quay màn hình. Không commit video riêng tư.
2. Ghi phần cứng, hệ điều hành, `python --version`, `python -m pip freeze`, SHA-256 model, provider và toàn bộ tham số. Tách clip dùng chỉnh ngưỡng khỏi clip đánh giá.
3. Chạy mỗi clip trong phiên mới để reset trạng thái, ví dụ:

   ```bash
   python -m laneguard --video /path/to/clear.mp4 --headless --log runs/clear.csv
   python -m laneguard --video /path/to/curve.mp4 --headless --log runs/curve.csv
   ```

4. Xem video gốc và gán nhãn độc lập, không xem dự đoán trước. Chọn frame đại diện và quanh thời điểm bắt đầu/kết thúc lệch làn. Nhãn `warning` nghĩa là tình huống theo quy tắc gán nhãn thủ công cần cảnh báo; `normal` nghĩa là không cần cảnh báo. Định nghĩa tiêu chí, dung sai thời gian, camera reference và số mẫu từ trước. Các frame mơ hồ không đưa vào CSV; ghi chúng riêng với lý do. Video dùng số khung gốc bắt đầu từ **1**, không dùng `elapsed_s` (đó là thời gian xử lý).
5. Tạo `runs/labels.csv`, tên `clip` khớp tên file log không có `.csv`. Ví dụ sau chỉ minh họa định dạng, không phải nhãn thật:

   ```csv
   clip,source_frame,label
   clear,1,normal
   clear,120,warning
   curve,60,normal
   ```

6. Tính confusion matrix và số liệu runtime:

   ```bash
   python tools/evaluate.py --labels runs/labels.csv --predictions runs/clear.csv runs/curve.csv > runs/evaluation.json
   ```

Script yêu cầu mỗi mẫu nhãn có dự đoán tương ứng; từ chối nhãn thiếu dự đoán hoặc khóa trùng. Dùng tên clip khác nhau cho mỗi file. Không tự suy ra nhãn từ offset model.

## Diễn giải chỉ số

Positive = `LANE DEPARTURE WARNING`. `NORMAL` và `LANE NOT RELIABLE` đều không phát cảnh báo trong **confusion matrix nhị phân về cảnh báo**. Vì vậy unreliable trên mẫu warning tính là FN; unreliable trên mẫu normal tính là TN nhưng **không có nghĩa hệ thống xác nhận trạng thái normal**. Luôn báo `unreliable_samples` và `reliable_coverage` bên cạnh precision/recall để tránh che giấu việc hệ thống không đưa ra ước lượng.

Precision = TP/(TP+FP), recall = TP/(TP+FN). Khi mẫu số bằng 0, JSON trả `null`. Báo số clip, số mẫu warning/normal và cả TP/FP/TN/FN. Đây là số liệu theo frame được chọn, không phải số liệu theo sự kiện hoặc độ chính xác an toàn của xe. Những frame liên tiếp tương quan cao, cần ghi cách lấy mẫu.

`mean_processing_ms`, `mean_inference_ms` và `mean_rolling_fps` được tính trên toàn bộ frame của các log đầu vào, không chỉ các frame có nhãn. Mean rolling FPS không bằng throughput toàn phiên. Lưu JSON tổng kết ứng dụng để ghi throughput, so sánh cùng chế độ headless, cùng nguồn và phần cứng; ghi rõ có/không loại warm-up. Webcam có thể bỏ khung, vì vậy đánh giá chuẩn theo video trước.

Chất lượng detector (ví dụ F1/IoU biên làn trên dataset) là bài toán riêng. Script này không đo benchmark nhận diện làn và không sử dụng con số từ paper làm kết quả của LaneGuard.

## Failure cases và bảng báo cáo

Các failure case **đã tái hiện bằng test giả lập**: mất hai biên chuyển ngay sang unreliable; đầu ra có NaN không tạo cảnh báo; mất tất cả frame của video báo lỗi; model không có/lỗi định dạng bị từ chối. Đây không phải quan sát trên video đường thực.

Khi chạy thực tế, lưu ít nhất một frame sai, clip và source frame, điều kiện ánh sáng, kỳ vọng, trạng thái thật của ứng dụng, offset và nguyên nhân nghi ngờ. Các giả thuyết cần thử gồm bóng đổ, nhầm làn kế bên và phản chiếu màn hình.

| Hạng mục | Trạng thái hiện tại |
| --- | --- |
| Clip đường thực đã chạy | 0 |
| Mẫu thủ công normal / warning | 0 / 0 |
| TP / FP / TN / FN | Chờ dữ liệu |
| Precision / recall | Chờ dữ liệu |
| FPS / latency pretrained | Chờ model hoàn chỉnh và clip |
| Failure case đường thực | Chờ quan sát |

Điền bảng từ kết quả đo, kèm cấu hình và bằng chứng; không thay số liệu đang chờ bằng kết quả test tổng hợp.

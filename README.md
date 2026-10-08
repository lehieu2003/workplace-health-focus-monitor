# Workplace Health & Focus Monitor 🖥️ 🧘‍♂️

Hệ thống giám sát tư thế làm việc, phát hiện buồn ngủ và cảnh báo mất tập trung (dùng điện thoại) thời gian thực, tối ưu hóa cho **Mac Mini M4 (Apple Silicon)** và macOS.

---

## 🌟 Tính năng nổi bật

1. **Giám sát tư thế (Posture Monitor)**:
   - Đo góc nghiêng cổ (Neck Angle), góc nghiêng cột sống/thân mình (Torso Tilt) và độ lệch vai.
   - **Cơ chế Calibrate cá nhân hóa (`phím C`)**: Lưu lại baseline khi người dùng ngồi thẳng đúng form, tự động cảnh báo khi lệch khỏi ngưỡng công thái học.
   - Hiển thị trạng thái **GOOD POSTURE** (Xanh lá) hoặc **BAD POSTURE** (Đỏ) + Tiếng Bíp cảnh báo.
2. **Phát hiện buồn ngủ & mệt mỏi (Drowsiness & Fatigue)**:
   - Tính toán chỉ số mở mắt **EAR (Eye Aspect Ratio)**: phân biệt chớp mắt tự nhiên vs nhắm mắt kéo dài $\ge 1.5s$.
   - Tính toán độ mở miệng **MAR (Mouth Aspect Ratio)**: phát hiện ngáp kéo dài $\ge 2.0s$.
   - Đổi trạng thái sang **DROWSY: EYES CLOSED** / **YAWNING** (Cam/Vàng) + Bíp cảnh báo.
3. **Phát hiện dùng điện thoại (Phone Distraction)**:
   - Mô hình **YOLOv8n** nhận diện smartphone trong khung hình / cầm nghe điện thoại.
   - Cơ chế chạy ngắt quãng (Frame-skipping) giữ FPS luôn ở mức 30 - 60 FPS mượt mà.
4. **Cảnh báo âm thanh macOS không chặn luồng (Non-blocking Audio)**:
   - Sử dụng âm thanh hệ thống macOS (`Ping.aiff` qua `afplay` chạy nền).
   - Cơ chế Debounce 2 giây: không gây giật video stream, không phát tiếng dồn dập khó chịu.
5. **Giao diện HUD Dashboard chuyên nghiệp**:
   - Thanh trạng thái màu sắc nổi bật trên cùng.
   - Khung thông số Telemetry thời gian thực (Góc cổ, góc lưng, chênh lệch baseline, EAR, FPS, số lần vi phạm).
   - Bộ khung xương (Skeleton) đổi màu trực tiếp trên người.

---

## 🚀 Hướng dẫn Cài đặt & Chạy

### 1. Kích hoạt môi trường ảo:
```bash
source .venv/bin/activate
```

### 2. Khởi động ứng dụng:

#### 🍏 Chế độ 1: macOS Menu Bar App (Chạy ngầm tàng hình - Khuyến nghị khi làm việc):
```bash
python3 -m src.main
```
- Nằm gọn trên thanh **Menu Bar (góc trên bên phải màn hình Mac)** với icon `🧘 100%`.
- Không chiếm màn hình làm việc.
- **Tiết kiệm CPU**: Tự động giảm nhịp xử lý (CPU ăn $<3\%$).
- **Thông báo macOS**: Khi gù lưng/ngáp liên tục $>5s$, tự động gửi banner thông báo ở góc màn hình.
- Click chuột vào icon Menu Bar để: **Căn chỉnh tư thế (Calibrate)**, **Bật/Tắt Cửa sổ Camera**, hoặc **Xem thống kê chi tiết hôm nay**.

#### 🖥️ Chế độ 2: Standalone Interactive Window (Mở cửa sổ camera để demo/thuyết trình):
```bash
python3 -m src.main --cli
```
- Mở cửa sổ OpenCV toàn màn hình với thước đo góc, khung xương skeleton và bảng điều khiển HUD.

> [!TIP]
> **Cấu hình camera qua `.env`:**
> Địa chỉ IP Webcam điện thoại và góc xoay được lưu trong file [`.env`](file:///Users/lehieu/Projects/personal/Workplace%20Health%20&%20Focus%20Monitor/.env). Bạn có thể đổi IP bất kỳ lúc nào mà không cần gõ dài dòng trong terminal!

---

## ⌨️ Phím tắt Điều khiển khi Demo (Chế độ `--cli` hoặc khi mở Camera Preview)

| Phím | Chức năng | Mô tả chi tiết |
|:---:|:---|:---|
| **`C`** | **Calibrate Baseline** | Nhấn khi bạn đang ngồi thẳng lưng chuẩn công thái học để hệ thống lưu form chuẩn của bạn. |
| **`O`** | **Rotate Camera** | Xoay góc camera 90° tức thì (rất tiện khi dựng điện thoại dọc/ngang). |
| **`M`** | **Mute / Unmute** | Bật / Tắt âm thanh Bíp (tiện khi bạn đang nói chuyện/thuyết trình để không bị tiếng bíp ngắt lời). |
| **`R`** | **Reset Stats** | Đặt lại bộ đếm số lần vi phạm về 0. |
| **`Q`** | **Quit Demo** | Thoát ứng dụng và in bảng thống kê tổng kết phiên làm việc. |

---

## 🎬 Kịch bản Demo tại nhà (Step-by-Step Presentation Script)

1. **Khởi động**: Bật ứng dụng `python3 -m src.main`. Cửa sổ HUD xuất hiện với thông báo *"Press [C] when sitting upright to Calibrate!"*.
2. **Căn chỉnh ban đầu**:
   - Ngồi thẳng lưng, vai mở tự nhiên, nhìn thẳng vào màn hình.
   - Nhấn phím **`C`**. Màn hình xuất hiện thông báo *"Calibrated! Baseline: Neck=... deg, Torso=... deg"* và thanh trạng thái chuyển sang **XANH LÁ: GOOD POSTURE & FOCUSED**.
3. **Demo Gù lưng / Cúi đầu sâu**:
   - Cúi gập cổ về phía trước hoặc ngồi gù lưng xuống bàn như khi mỏi mệt.
   - **Kết quả**: Skeleton và thanh trên cùng lập tức chuyển sang **ĐỎ: BAD POSTURE: SLOUCH DETECTED**, hệ thống phát tiếng **BÍP** cảnh báo.
   - Ngồi thẳng lại: Hệ thống lập tức hồi phục về màu **XANH**.
4. **Demo Buồn ngủ**:
   - Nhắm mắt lại giữ nguyên trong hơn 1.5 giây (hoặc há to miệng giả vờ ngáp).
   - **Kết quả**: Thanh trạng thái chuyển sang **CAM: DROWSY: EYES CLOSED** và phát tiếng Bíp nhắc tỉnh ngủ.
5. **Demo Dùng điện thoại**:
   - Cầm smartphone đưa lên trước mặt hoặc đưa lên tai nghe.
   - **Kết quả**: Khung viền **TÍM** bám theo điện thoại kèm nhãn `PHONE (XX%)`, trạng thái báo **DISTRACTED: PHONE IN USE** + Bíp cảnh báo.
6. **Demo tương tác linh hoạt**:
   - Nhấn **`M`** để Mute âm thanh (thanh trên cùng hiển thị `SOUND: MUTED`).
   - Nhấn **`Q`** để kết thúc, terminal sẽ in ra báo cáo tóm tắt thời lượng và số lần vi phạm.

---

## 🧪 Kiểm thử Tự động (Automated Tests)

Chạy bộ test suite hoàn chỉnh:
```bash
.venv/bin/pytest tests/ -v
```


import cv2
import numpy as np
import time

# --- BƯỚC 1: TIỀN XỬ LÝ (Kết hợp trong hàm process_pipeline) ---
# Bao gồm Grayscale (ảnh xám) và Gaussian Blur (làm mờ)

# --- BƯỚC 2: DÒ CẠNH (CANNY) ---
def detect_edges(frame):
    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Làm mờ ảnh để giảm nhiễu
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Dùng thuật toán Canny để phát hiện biên (cạnh)
    # 50 và 150 là ngưỡng dưới và ngưỡng trên
    canny = cv2.Canny(blur, 50, 150)
    return canny

# --- BƯỚC 3: TẠO VÀ ÁP DỤNG MẶT NẠ (MASK / ROI) ---
def create_and_apply_mask(frame_edges, frame_shape):
    height, width = frame_shape
    
    # Định nghĩa các đỉnh của hình thang (Vùng quan tâm - Region of Interest)
    # Bạn PHẢI tinh chỉnh các giá trị này cho phù hợp với camera của bạn
    polygons = np.array([
        (int(width * 0.1), height),  # Đỉnh dưới bên trái
        (int(width * 0.9), height),  # Đỉnh dưới bên phải
        (int(width * 0.55), int(height * 0.6)), # Đỉnh trên bên phải (gần tâm)
        (int(width * 0.45), int(height * 0.6))  # Đỉnh trên bên trái (gần tâm)
    ])
    
    # Tạo một ảnh đen có cùng kích thước
    mask = np.zeros_like(frame_edges)
    
    # Vẽ hình thang màu trắng (255) lên ảnh đen
    cv2.fillPoly(mask, [polygons], 255)
    
    # Dùng phép "AND" bit-wise để chỉ giữ lại các cạnh nằm trong hình thang
    # Đây là bước tối ưu quan trọng: chỉ giữ lại các cạnh trong ROI
    masked_edges = cv2.bitwise_and(frame_edges, mask)
    return masked_edges

# --- BƯỚC 4: ĐÁNH DẤU, VẼ VẠCH KẺ ĐƯỜNG (HOUGH TRANSFORM) ---
def detect_lines(masked_edges):
    # Sử dụng HoughLinesP (Probabilistic Hough Transform) vì nó nhanh hơn
    # 2: Độ phân giải rho (pixel)
    # np.pi/180: Độ phân giải theta (radian)
    # 50: Ngưỡng (số phiếu tối thiểu để coi là 1 đường thẳng)
    # minLineLength=40: Độ dài tối thiểu của 1 đường (pixel)
    # maxLineGap=5: Khoảng cách tối đa giữa các đoạn để coi là 1 đường (pixel)
    lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 50, minLineLength=40, maxLineGap=5)
    return lines

def display_detected_lines(frame, lines):
    # Tạo một ảnh đen để vẽ các đường line (xanh lá) đã phát hiện
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            # lines trả về 1 mảng [[x1, y1, x2, y2]]
            x1, y1, x2, y2 = line.reshape(4)
            # Vẽ đường thẳng màu xanh (0, 255, 0) với độ dày 5
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    # Trộn ảnh gốc với ảnh chứa các đường line
    # (ảnh gốc 80%, ảnh line 100%)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return combo_image

# --- BƯỚC 5: TÍNH TOÁN LOGIC ĐIỀU KHIỂN ---

def make_coordinates(frame, line_parameters):
    """
    Tạo tọa độ (x1, y1, x2, y2) từ độ dốc (slope) và điểm cắt (intercept).
    """
    slope, intercept = line_parameters
    height = frame.shape[0]
    # Chúng ta muốn vẽ từ y = 0.6 * height (đỉnh ROI)
    # đến y = height (đáy ảnh)
    y1 = height
    y2 = int(height * 0.6)
    
    # Từ y = mx + b => x = (y - b) / m
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(frame, lines):
    """
    Phân loại các đoạn thẳng thành làn trái/phải và tính trung bình
    để ra 2 đường thẳng duy nhất.
    """
    left_fit = []  # Tập hợp các (slope, intercept) của làn trái
    right_fit = [] # Tập hợp các (slope, intercept) của làn phải
    
    if lines is None:
        return None, None, None

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        
        # Bỏ qua các đường thẳng đứng (x1=x2)
        if x1 == x2:
            continue
            
        # Tính slope (độ dốc) và intercept (điểm cắt y)
        # y = mx + b
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        
        # Làn trái sẽ có độ dốc âm (trong tọa độ ảnh)
        # Làn phải sẽ có độ dốc dương
        # Lọc bỏ các đường gần như nằm ngang (slope gần 0)
        if slope < -0.5:
            left_fit.append((slope, intercept))
        elif slope > 0.5:
            right_fit.append((slope, intercept))
            
    left_line = None
    right_line = None
    left_fit_average = None
    right_fit_average = None
    
    # Tính trung bình các (slope, intercept) của làn trái
    if len(left_fit) > 0:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coordinates(frame, left_fit_average)

    # Tính trung bình các (slope, intercept) của làn phải
    if len(right_fit) > 0:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(frame, right_fit_average)
        
    # Trả về 2 đường (để vẽ) và các tham số (để tính toán)
    return [left_line, right_line], left_fit_average, right_fit_average

def display_averaged_lines(frame, lines):
    """
    Vẽ 2 đường trung bình (màu đỏ) lên ảnh.
    """
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line
                # Vẽ đường thẳng màu đỏ (0, 0, 255) với độ dày 10
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
    
    # Trộn ảnh (đã có đường xanh) với ảnh 2 đường đỏ
    combo_image = cv2.addWeighted(frame, 1, line_image, 1, 0)
    return combo_image

def calculate_steering(frame, left_params, right_params):
    """
    Tính toán góc lái dựa trên vị trí của 2 làn.
    """
    if left_params is None or right_params is None:
        # Nếu mất 1 trong 2 làn, không thể tính
        return "Lane Not Found", 0

    height, width = frame.shape[:2]
    
    # Tâm của xe (camera)
    car_center_x = width // 2
    
    left_m, left_b = left_params
    right_m, right_b = right_params
    
    # Tính tọa độ x của 2 làn ở dưới đáy ảnh (y = height)
    left_x_bottom = int((height - left_b) / left_m)
    right_x_bottom = int((height - right_b) / right_m)
    
    # Tâm của làn đường
    lane_center_x = (left_x_bottom + right_x_bottom) / 2
    
    # Tính độ lệch (offset)
    # offset > 0: Xe đang ở bên phải -> cần "Steer Left"
    # offset < 0: Xe đang ở bên trái -> cần "Steer Right"
    offset = car_center_x - lane_center_x
    
    # Đặt một ngưỡng (ví dụ 20 pixels) để quyết định
    threshold = 20
    
    if offset > threshold:
        command = "Steer Left"
    elif offset < -threshold:
        command = "Steer Right"
    else:
        command = "Straight"
        
    return command, offset

def display_steering_info(frame, command, offset):
    """
    Hiển thị lệnh điều khiển và độ lệch lên màn hình.
    """
    steering_color = (0, 255, 0) # Xanh lá (Straight)
    if "Left" in command:
        steering_color = (0, 255, 255) # Vàng
    elif "Right" in command:
        steering_color = (0, 255, 255) # Vàng

    # Hiển thị lệnh
    cv2.putText(frame, command, (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, steering_color, 2, cv2.LINE_AA)
    
    # Hiển thị độ lệch
    cv2.putText(frame, f"Offset: {offset:.2f} px", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    
    return frame

# --- HÀM TỔNG HỢP TOÀN BỘ QUY TRÌNH ---
def process_pipeline(frame):
    """
    Xử lý toàn bộ quy trình cho mỗi khung hình (frame).
    """
    # 0. Giữ lại bản sao của frame gốc
    original_frame = frame.copy()
    
    # BƯỚC 1 & 2: Tiền xử lý (Blur) và Dò cạnh (Canny)
    canny_edges = detect_edges(frame)
    
    # BƯỚC 3: Tạo và áp dụng Mask (Vùng quan tâm - ROI)
    # frame.shape[:2] là (height, width) của ảnh gốc
    masked_canny = create_and_apply_mask(canny_edges, frame.shape[:2])
    
    # BƯỚC 4: Phát hiện đoạn thẳng (HoughLinesP)
    lines = detect_lines(masked_canny)
    
    # BƯỚC 5: TÍNH TOÁN, ĐIỀU KHIỂN
    
    # 5.1. Tính toán 2 đường trung bình (trái/phải)
    averaged_lines, left_params, right_params = average_slope_intercept(original_frame, lines)
    
    # 5.2. Tính toán lệnh điều khiển
    steering_command, offset = calculate_steering(original_frame, left_params, right_params)
    
    # BƯỚC 4 (tiếp): Vẽ các đường ảo
    
    # 4.1. Vẽ các đoạn thẳng (xanh lá) đã phát hiện
    detected_line_image = display_detected_lines(original_frame, lines)
    
    # 4.2. Vẽ 2 đường trung bình (đỏ)
    averaged_line_image = display_averaged_lines(detected_line_image, averaged_lines)
    
    # 5.3. Hiển thị thông tin điều khiển lên ảnh
    final_image = display_steering_info(averaged_line_image, steering_command, offset)
                
    return final_image

# --- HÀM CHÍNH (MAIN) ĐỂ CHẠY VIDEO ---
def main():
    # --- THAY ĐỔI NGUỒN VIDEO TẠI ĐÂY ---
    # 1. Dùng video file (thay 'test_video.mp4' bằng video của bạn)
    VIDEO_SOURCE = "test2.mp4" 
    
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở nguồn video '{VIDEO_SOURCE}'")
        return

    prev_time = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Kết thúc video hoặc lỗi đọc frame.")
            # Quay lại đầu video nếu là file
            # Chuyển 0 thành chuỗi '0' để so sánh
            if str(VIDEO_SOURCE) != '0': 
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        # Gửi frame vào quy trình xử lý (đã bỏ resize)
        processed_frame = process_pipeline(frame)
        
        # Tính toán và hiển thị FPS (Frames Per Second)
        curr_time = time.time()
        # Tránh lỗi chia cho 0 ở frame đầu tiên
        if (curr_time - prev_time) > 0:
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Hiển thị kết quả
        cv2.imshow("Phat hien lan duong (Nhan 'q' de thoat)", processed_frame)
        
        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Dọn dẹp
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


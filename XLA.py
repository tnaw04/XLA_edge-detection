import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import time
from collections import deque

# =============================================================================
# PHẦN 1: CÁC HÀM XỬ LÝ ẢNH CỐT LÕI (GIỮ NGUYÊN)
# =============================================================================

def load_image_grayscale(image_path):
    """Tải ảnh và chuyển sang ảnh xám."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Không thể tải ảnh. Vui lòng kiểm tra đường dẫn.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def apply_gaussian_blur(gray_img, kernel_size=(5, 5)):
    """Áp dụng lọc Gaussian để giảm nhiễu."""
    return cv2.GaussianBlur(gray_img, kernel_size, 0)

def apply_bilateral_filter(gray_img, d=9, sigma_color=75, sigma_space=75):
    """Áp dụng bilateral filter - giữ cạnh tốt hơn Gaussian."""
    return cv2.bilateralFilter(gray_img, d, sigma_color, sigma_space)

def sobel_detector(gray_img):
    """Phát hiện biên bằng Sobel với cải thiện."""
    blurred = apply_gaussian_blur(gray_img, (3, 3))
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = np.uint8(np.clip(magnitude, 0, 255))
    return magnitude

def laplacian_detector(gray_img):
    """Phát hiện biên bằng Laplacian với cải thiện."""
    blurred_img = apply_bilateral_filter(gray_img)
    lap = cv2.Laplacian(blurred_img, cv2.CV_64F, ksize=3)
    laplacian_result = cv2.convertScaleAbs(lap)
    _, thresholded = cv2.threshold(laplacian_result, 30, 255, cv2.THRESH_BINARY)
    return thresholded

def canny_detector(gray_img, auto_threshold=True, t_lower=50, t_upper=150):
    """Phát hiện biên bằng Canny với auto threshold cải thiện."""
    blurred_img = apply_bilateral_filter(gray_img)
    if auto_threshold:
        v = np.median(blurred_img)
        sigma = 0.33
        t_lower = int(max(0, (1.0 - sigma) * v))
        t_upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(blurred_img, t_lower, t_upper)
    return edges

# =============================================================================
# PHẦN 2: ỨNG DỤNG ĐẾM VẬT THỂ (GIỮ NGUYÊN)
# =============================================================================

def count_objects(original_img, gray_img, canny_t1, canny_t2, kernel_size, min_area):
    """Đếm vật thể với Canny."""
    blurred = apply_bilateral_filter(gray_img)
    canny_edges = cv2.Canny(blurred, canny_t1, canny_t2)
    
    k_size = int(kernel_size)
    if k_size % 2 == 0: 
        k_size += 1
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    dilated = cv2.dilate(canny_edges, kernel, iterations=2)
    closed_edges = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=1)
    closed_edges = cv2.erode(closed_edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    object_count = 0
    output_image = original_img.copy()
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        if area > min_area and perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.1:
                object_count += 1
                cv2.drawContours(output_image, [cnt], -1, (0, 255, 0), 2)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(output_image, str(object_count), (cX-10, cY+10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    text = f"So luong vat the: {object_count}"
    cv2.putText(output_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return output_image, closed_edges, object_count

# =============================================================================
# PHẦN 3: PHÁT HIỆN LÀN ĐƯỜNG (NÂNG CẤP VỚI IPM VÀ PIPELINE KẾT HỢP)
# =============================================================================

class LaneTracker:
    """Class để theo dõi làn đường với smoothing."""
    def __init__(self):
        self.left_fit_history = deque(maxlen=5)
        self.right_fit_history = deque(maxlen=5)
        
    def add_measurement(self, left_fit, right_fit):
        if left_fit is not None:
            self.left_fit_history.append(left_fit)
        if right_fit is not None:
            self.right_fit_history.append(right_fit)
    
    def get_smoothed_fit(self):
        left_avg = np.mean(self.left_fit_history, axis=0) if len(self.left_fit_history) > 0 else None
        right_avg = np.mean(self.right_fit_history, axis=0) if len(self.right_fit_history) > 0 else None
        return left_avg, right_avg

# [CẢI TIẾN] Hàm này giờ chỉ trả về MASK (ảnh nhị phân)
def lane_filter_color_mask(frame):
    """Phát hiện vạch kẻ đường màu trắng và vàng, trả về MASK."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Ngưỡng màu trắng
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Ngưỡng màu vàng
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    return combined_mask

# [CẢI TIẾN] Hàm này chạy Canny trên ảnh xám
def lane_detect_edges_grayscale(frame):
    """Phát hiện cạnh trên ảnh xám (grayscale)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Dùng Bilateral Filter để giảm nhiễu mà vẫn giữ cạnh
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    # Ngưỡng Canny
    canny = cv2.Canny(blur, 50, 150)
    return canny

# [CẢI TIẾN] Cân chỉnh lại IPM cho video mới
def lane_perspective_transform(frame):
    """
    Áp dụng Inverse Perspective Mapping (IPM) để có bird's-eye view.
    Trả về ảnh đã warp, ma trận biến đổi (M) và ma trận nghịch đảo (Minv).
    """
    height, width = frame.shape[:2]
    
    # [ĐÃ CÂN CHỈNH] Các điểm này được chỉnh cho video videoplayback.mp4
    # (Điểm tụ cao hơn, đường rộng hơn ở dưới)
    src_pts = np.float32([
        (int(width * 0.45), int(height * 0.55)),  # Top-left (Đã nâng lên)
        (int(width * 0.55), int(height * 0.55)),  # Top-right (Đã nâng lên)
        (int(width * 0.95), height),              # Bottom-right (Đã mở rộng)
        (int(width * 0.05), height)               # Bottom-left (Đã mở rộng)
    ])
    
    # Điểm đích (dst) giữ nguyên
    dst_pts = np.float32([
        (int(width * 0.1), 0),                   # Top-left
        (int(width * 0.9), 0),                   # Top-right
        (int(width * 0.9), height),              # Bottom-right
        (int(width * 0.1), height)               # Bottom-left
    ])
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    
    # Biến đổi ảnh (hoặc mask)
    warped_frame = cv2.warpPerspective(frame, M, (width, height), flags=cv2.INTER_LINEAR)
    
    return warped_frame, M, Minv

# [GIỮ NGUYÊN] Phát hiện đường thẳng
def lane_detect_lines(masked_edges):
    """Phát hiện đường thẳng với Hough Transform."""
    lines = cv2.HoughLinesP(
        masked_edges, 
        rho=1,
        theta=np.pi/180,
        threshold=30,
        minLineLength=30,
        maxLineGap=100
    )
    return lines

# [GIỮ NGUYÊN] Tạo tọa độ
def lane_make_coordinates(frame, line_parameters):
    if line_parameters is None or len(line_parameters) != 2:
        return None
    slope, intercept = line_parameters
    if abs(slope) < 0.01:
        return None
    height = frame.shape[0]
    y1 = height; y2 = 0
    try:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])
    except:
        return None

# [GIỮ NGUYÊN] Tính trung bình
def lane_average_slope_intercept(frame, lines):
    left_fit = []; right_fit = []
    if lines is None: 
        return [None, None], None, None # ĐÃ SỬA LỖI NONETYPE TỪ LẦN TRƯỚC
    
    width = frame.shape[1]
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x1 == x2: 
            continue
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]; intercept = parameters[1]
        
        # [CẢI TIẾN] Phân loại trong ảnh bird's-eye CHỈ NÊN dựa vào vị trí x
        # 1. Chỉ giữ lại các đường gần thẳng đứng (lọc nhiễu ngang)
        if abs(slope) > 0.3: # Ngưỡng 0.3 để linh hoạt hơn
            
            # 2. Phân loại thuần túy dựa trên vị trí X
            if x1 < (width // 2): # Làn trái
                left_fit.append((slope, intercept))
            elif x1 > (width // 2): # Làn phải
                right_fit.append((slope, intercept))

    left_fit_avg = np.average(left_fit, axis=0) if len(left_fit) > 0 else None
    right_fit_avg = np.average(right_fit, axis=0) if len(right_fit) > 0 else None
    
    left_line = lane_make_coordinates(frame, left_fit_avg) if left_fit_avg is not None else None
    right_line = lane_make_coordinates(frame, right_fit_avg) if right_fit_avg is not None else None
    
    return [left_line, right_line], left_fit_avg, right_fit_avg

# [GIỮ NGUYÊN] Vẽ vùng
def lane_draw_area(frame, lines, Minv):
    if lines is None or lines[0] is None or lines[1] is None:
        return frame 
    warp_zero = np.zeros_like(frame).astype(np.uint8)
    color_warp = cv2.cvtColor(warp_zero, cv2.COLOR_BGR2RGB)
    pts_left = np.array([(lines[0][0], lines[0][1]), (lines[0][2], lines[0][3])], dtype=np.int32)
    pts_right = np.array([(lines[1][0], lines[1][1]), (lines[1][2], lines[1][3])], dtype=np.int32)
    pts = np.vstack([pts_left, np.flipud(pts_right)])
    cv2.fillPoly(color_warp, [pts], (0, 255, 0))
    new_warp = cv2.warpPerspective(color_warp, Minv, (frame.shape[1], frame.shape[0]))
    result = cv2.addWeighted(frame, 1, new_warp, 0.3, 0)
    return result

# [GIỮ NGUYÊN] Tính toán hướng lái
def lane_calculate_steering(frame, left_params, right_params):
    if left_params is None or right_params is None: 
        return "Lane Not Found", 0
    height, width = frame.shape[:2]
    car_center_x = width // 2
    try:
        left_x_bottom = int((height - left_params[1]) / left_params[0])
        right_x_bottom = int((height - right_params[1]) / right_params[0])
        lane_center_x = (left_x_bottom + right_x_bottom) / 2
        offset = car_center_x - lane_center_x
        
        if offset > 30: command = "Steer Left"
        elif offset < -30: command = "Steer Right"
        else: command = "Straight"
        return command, offset
    except:
        return "Error", 0

# [GIỮ NGUYÊN] Hiển thị thông tin
def lane_display_info(frame, command, offset):
    overlay = frame.copy()
    if command == "Straight": color, bg_color = (0, 255, 0), (0, 100, 0)
    elif "Steer" in command: color, bg_color = (0, 255, 255), (0, 100, 100)
    else: color, bg_color = (0, 0, 255), (0, 0, 100)
    
    cv2.rectangle(overlay, (10, 10), (350, 130), bg_color, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "LANE DETECTION (IPM-Robust)", (20, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Direction: {command}", (20, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Offset: {offset:.1f} px", (20, 105), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

# [CẢI TIẾN] PIPELINE TỔNG THỂ ĐÃ THAY ĐỔI
def lane_process_pipeline(frame, tracker=None):
    """Pipeline xử lý phát hiện làn đường (Nâng cấp: Kết hợp Kênh Màu và Kênh Cạnh)."""
    original_frame = frame.copy()
    height, width = frame.shape[:2]
    
    # 1. TẠO KÊNH XỬ LÝ (TRƯỚC KHI BIẾN ĐỔI)
    # Kênh 1: Lọc Cạnh (Canny trên ảnh xám)
    canny_edges = lane_detect_edges_grayscale(frame)
    # Kênh 2: Lọc Màu (Trắng/Vàng)
    color_mask = lane_filter_color_mask(frame)
    # Kết hợp (OR): Lấy cả cạnh Canny VÀ màu
    combined_mask = cv2.bitwise_or(canny_edges, color_mask)

    # 2. Biến đổi ảnh (IPM) - CHỈ BIẾN ĐỔI MASK KẾT HỢP
    warped_mask, M, Minv = lane_perspective_transform(combined_mask)
    
    # 3. Phát hiện đường thẳng (trên mask đã biến đổi)
    lines = lane_detect_lines(warped_mask)
    
    # 4. Tính trung bình (trên ảnh warped)
    averaged_lines, left_params, right_params = lane_average_slope_intercept(warped_mask, lines)
    
    # 5. Smoothing (Tracker)
    if tracker is not None:
        tracker.add_measurement(left_params, right_params)
        left_params_smooth, right_params_smooth = tracker.get_smoothed_fit()
        
        if left_params_smooth is not None:
            left_line_smooth = lane_make_coordinates(warped_mask, left_params_smooth)
        else:
            left_line_smooth = averaged_lines[0]
            left_params_smooth = left_params
            
        if right_params_smooth is not None:
            right_line_smooth = lane_make_coordinates(warped_mask, right_params_smooth)
        else:
            right_line_smooth = averaged_lines[1]
            right_params_smooth = right_params
        
        averaged_lines = [left_line_smooth, right_line_smooth]
        left_params = left_params_smooth
        right_params = right_params_smooth
    
    # 6. Tính toán hướng lái
    steering_command, offset = lane_calculate_steering(warped_mask, left_params, right_params)
    
    # 7. Vẽ vùng làn đường (polygon) lên ảnh gốc
    final_image = lane_draw_area(original_frame, averaged_lines, Minv)

    # 8. Hiển thị thông tin
    final_image = lane_display_info(final_image, steering_command, offset)
    
    # Hiển thị ảnh mask (bird's-eye) đã kết hợp ở góc để debug
    scale = 0.25
    small_edges = cv2.resize(warped_mask, (int(width*scale), int(height*scale)))
    small_edges_bgr = cv2.cvtColor(small_edges, cv2.COLOR_GRAY2BGR)
    final_image[10:10+small_edges_bgr.shape[0], 
                width-10-small_edges_bgr.shape[1]:width-10] = small_edges_bgr
    
    return final_image

# =============================================================================
# PHẦN 4: GUI (GIỮ NGUYÊN)
# =============================================================================

class EdgeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng dụng Xử lý ảnh - Edge Detection & Lane Finding (IPM-Robust)")
        self.root.geometry("1500x900")

        self.original_img = None
        self.gray_img = None
        self.processed_img = None
        self.image_path = None
        self.video_running = False
        self.cap = None
        self.lane_tracker = None

        frame_controls = tk.Frame(root, width=320, bg='#f0f0f0', relief=tk.RIDGE, borderwidth=2)
        frame_controls.pack(side=tk.LEFT, fill=tk.Y)
        frame_controls.pack_propagate(False)

        frame_images = tk.Frame(root, bg='white')
        frame_images.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.panel_original = tk.Label(frame_images, bg='white', text="Ảnh gốc", 
                                       font=("Arial", 12, "bold"), compound='top')
        self.panel_original.pack(side=tk.LEFT, padx=10, pady=10, expand=True)
        
        self.panel_processed = tk.Label(frame_images, bg='white', text="Ảnh xử lý", 
                                       font=("Arial", 12, "bold"), compound='top')
        self.panel_processed.pack(side=tk.RIGHT, padx=10, pady=10, expand=True)

        lbl_title = tk.Label(frame_controls, text="📷 ẢNH TĨNH", 
                           font=("Arial", 14, "bold"), bg='#f0f0f0', fg='#2c3e50')
        lbl_title.pack(pady=(15, 10), padx=10)

        btn_style = {'font': ('Arial', 10), 'bg': '#3498db', 'fg': 'white', 
                    'activebackground': '#2980b9', 'cursor': 'hand2'}

        tk.Button(frame_controls, text="📂 Tải ảnh", command=self.load_image, 
                 **btn_style).pack(fill=tk.X, padx=10, pady=3)
        tk.Button(frame_controls, text="🔍 Sobel Detector", command=self.run_sobel, 
                 **btn_style).pack(fill=tk.X, padx=10, pady=3)
        tk.Button(frame_controls, text="🔍 Laplacian Detector", command=self.run_laplacian, 
                 **btn_style).pack(fill=tk.X, padx=10, pady=3)
        tk.Button(frame_controls, text="🔍 Canny Detector (Auto)", command=self.run_canny, 
                 **btn_style).pack(fill=tk.X, padx=10, pady=3)

        separator = ttk.Separator(frame_controls, orient='horizontal')
        separator.pack(fill=tk.X, padx=10, pady=15)

        tk.Label(frame_controls, text="⚙️ TINH CHỈNH ĐẾM", 
                font=("Arial", 12, "bold"), bg='#f0f0f0', fg='#2c3e50').pack(pady=(5, 10), padx=10)

        tk.Label(frame_controls, text="Ngưỡng Canny T1:", bg='#f0f0f0', 
                font=('Arial', 9)).pack(padx=10, anchor='w')
        self.slider_t1 = tk.Scale(frame_controls, from_=0, to=255, orient=tk.HORIZONTAL, 
                                 bg='#f0f0f0', highlightthickness=0, 
                                 troughcolor='#3498db', sliderlength=20)
        self.slider_t1.set(50)
        self.slider_t1.pack(fill=tk.X, padx=10)
        
        tk.Label(frame_controls, text="Ngưỡng Canny T2:", bg='#f0f0f0', 
                font=('Arial', 9)).pack(padx=10, anchor='w')
        self.slider_t2 = tk.Scale(frame_controls, from_=0, to=255, orient=tk.HORIZONTAL, 
                                 bg='#f0f0f0', highlightthickness=0,
                                 troughcolor='#3498db', sliderlength=20)
        self.slider_t2.set(150)
        self.slider_t2.pack(fill=tk.X, padx=10)

        tk.Label(frame_controls, text="Kích thước Kernel:", bg='#f0f0f0', 
                font=('Arial', 9)).pack(padx=10, anchor='w')
        self.slider_kernel = tk.Scale(frame_controls, from_=1, to=21, orient=tk.HORIZONTAL, 
                                     bg='#f0f0f0', highlightthickness=0,
                                     troughcolor='#e74c3c', sliderlength=20)
        self.slider_kernel.set(5)
        self.slider_kernel.pack(fill=tk.X, padx=10)

        tk.Label(frame_controls, text="Diện tích tối thiểu:", bg='#f0f0f0', 
                font=('Arial', 9)).pack(padx=10, anchor='w')
        self.slider_area = tk.Scale(frame_controls, from_=0, to=2000, orient=tk.HORIZONTAL, 
                                   bg='#f0f0f0', highlightthickness=0,
                                   troughcolor='#27ae60', sliderlength=20)
        self.slider_area.set(100)
        self.slider_area.pack(fill=tk.X, padx=10)

        tk.Button(frame_controls, text="🔢 Đếm Vật thể", command=self.run_counting, 
                 bg='#e67e22', fg='white', font=('Arial', 11, 'bold'), 
                 activebackground='#d35400', cursor='hand2').pack(fill=tk.X, padx=10, pady=8)

        separator2 = ttk.Separator(frame_controls, orient='horizontal')
        separator2.pack(fill=tk.X, padx=10, pady=15)

        lbl_video = tk.Label(frame_controls, text="🎥 VIDEO", 
                           font=("Arial", 14, "bold"), bg='#f0f0f0', fg='#2c3e50')
        lbl_video.pack(pady=(5, 10), padx=10)

        self.btn_lane = tk.Button(frame_controls, text="🛣️ Phát hiện làn đường", 
                                  command=self.run_lane_detection_video, 
                                  bg='#27ae60', fg='white', font=("Arial", 11, "bold"),
                                  activebackground='#229954', cursor='hand2')
        self.btn_lane.pack(fill=tk.X, padx=10, pady=5)

        self.btn_stop_video = tk.Button(frame_controls, text="⏹️ Dừng Video", 
                                        command=self.stop_video, state=tk.DISABLED,
                                        bg='#e74c3c', fg='white', font=("Arial", 10, "bold"),
                                        activebackground='#c0392b', cursor='hand2')
        self.btn_stop_video.pack(fill=tk.X, padx=10, pady=3)

        tk.Button(frame_controls, text="💾 Lưu ảnh kết quả", command=self.save_image,
                 bg='#9b59b6', fg='white', font=('Arial', 10, 'bold'),
                 activebackground='#8e44ad', cursor='hand2').pack(fill=tk.X, padx=10, 
                                                                   pady=(20, 15), side=tk.BOTTOM)

        self.status_bar = tk.Label(root, text="Sẵn sàng", bd=1, relief=tk.SUNKEN, 
                                  anchor=tk.W, bg='#ecf0f1', font=('Arial', 9))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_image(self):
        self.image_path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        if not self.image_path: 
            return
        try:
            self.original_img, self.gray_img = load_image_grayscale(self.image_path)
            self.display_image(self.original_img, self.panel_original, "Ảnh gốc")
            self.display_image(self.gray_img, self.panel_processed, "Ảnh xám")
            self.status_bar.config(text=f"Đã tải: {self.image_path}")
        except Exception as e: 
            messagebox.showerror("Lỗi", f"Không thể tải ảnh: {e}")

    def run_sobel(self):
        if self.gray_img is None: 
            return messagebox.showwarning("Cảnh báo", "Vui lòng tải ảnh trước!")
        self.processed_img = sobel_detector(self.gray_img)
        self.display_image(self.processed_img, self.panel_processed, "Kết quả Sobel")
        self.status_bar.config(text="Hoàn thành Sobel")

    def run_laplacian(self):
        if self.gray_img is None: 
            return messagebox.showwarning("Cảnh báo", "Vui lòng tải ảnh trước!")
        self.processed_img = laplacian_detector(self.gray_img)
        self.display_image(self.processed_img, self.panel_processed, "Kết quả Laplacian")
        self.status_bar.config(text="Hoàn thành Laplacian")

    def run_canny(self):
        if self.gray_img is None: 
            return messagebox.showwarning("Cảnh báo", "Vui lòng tải ảnh trước!")
        self.processed_img = canny_detector(self.gray_img, auto_threshold=True)
        self.display_image(self.processed_img, self.panel_processed, "Kết quả Canny")
        self.status_bar.config(text="Hoàn thành Canny")

    def run_counting(self):
        if self.original_img is None: 
            return messagebox.showwarning("Cảnh báo", "Vui lòng tải ảnh trước!")
        t1, t2 = self.slider_t1.get(), self.slider_t2.get()
        if t1 >= t2: 
            t2 = t1 + 1
            self.slider_t2.set(t2)
        result_img, _, count = count_objects(
            self.original_img, self.gray_img, t1, t2, 
            self.slider_kernel.get(), self.slider_area.get())
        self.processed_img = result_img
        self.display_image(self.processed_img, self.panel_processed, f"Kết quả: {count} vật thể")
        self.status_bar.config(text=f"Đã phát hiện {count} vật thể")
        
    def run_lane_detection_video(self):
        """Chọn và chạy video lane detection (phiên bản non-blocking Tkinter)."""
        video_path = filedialog.askopenfilename(
            title="Chọn file video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if not video_path:
            return

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", f"Không thể mở video: {video_path}")
            return

        self.video_running = True
        self.lane_tracker = LaneTracker()
        self.btn_lane.config(state=tk.DISABLED)
        self.btn_stop_video.config(state=tk.NORMAL)

        self._video_frame_count = 0
        self._prev_time = time.time()
        self._video_update()


    def _video_update(self):
        """Đọc từng frame và hiển thị (dùng Tkinter .after, không block GUI)."""
        if not self.video_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_video()
            return

        self._video_frame_count += 1

        try:
            processed_frame = lane_process_pipeline(frame, self.lane_tracker)
        except Exception as e:
            # In lỗi chi tiết ra console
            print(f"❌ Lỗi xử lý frame {self._video_frame_count}: {e}")
            # Dừng video một cách an toàn
            self.stop_video()
            # Hiển thị lỗi cho người dùng
            messagebox.showerror("Lỗi Runtime", f"Đã xảy ra lỗi khi xử lý video:\n{e}")
            return

        curr_time = time.time()
        fps = 1.0 / (curr_time - self._prev_time) if (curr_time - self._prev_time) > 0 else 0.0
        self._prev_time = curr_time

        cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        self.processed_img = processed_frame
        # Hiển thị video lên panel
        self.display_image(self.processed_img, self.panel_processed, f"Video Frame {self._video_frame_count}")
        # Hiển thị ảnh gốc
        self.display_image(frame, self.panel_original, "Video Gốc")


        if self._video_frame_count % 30 == 0:
            self.status_bar.config(text=f"Frame {self._video_frame_count}, FPS: {fps:.1f}")

        # Lặp lại
        self.root.after(10, self._video_update)

        
    def stop_video(self):
        """Dừng phát video và giải phóng tài nguyên."""
        self.video_running = False
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()
            self.cap = None

        self.btn_lane.config(state=tk.NORMAL)
        self.btn_stop_video.config(state=tk.DISABLED)
        self.status_bar.config(text="Video đã dừng.")

    def save_image(self):
        if self.processed_img is None: 
            return messagebox.showwarning("Cảnh báo", "Không có ảnh để lưu!")
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png", 
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
        if save_path:
            cv2.imwrite(save_path, self.processed_img)
            messagebox.showinfo("Thành công", f"Đã lưu ảnh tại:\n{save_path}")
            self.status_bar.config(text=f"Đã lưu: {save_path}")

    def display_image(self, img, panel, title_text):
        max_width, max_height = 700, 750
        if img is None:
            return
        
        h, w = img.shape[:2]
        ratio = min(max_width / w, max_height / h)
        
        if ratio < 1:
            new_w, new_h = int(w * ratio), int(h * ratio)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        panel.config(image=img_tk, text=title_text, 
                     font=("Arial", 12, "bold"), compound='top')
        panel.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = EdgeDetectionApp(root)
    root.mainloop()
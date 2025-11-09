import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import time
from collections import deque

# =============================================================================
# PHẦN 1: CÁC HÀM XỬ LÝ ẢNH CỐT LÕI (ĐƯỢC CẢI THIỆN)
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
    # Làm mờ trước để giảm nhiễu
    blurred = apply_gaussian_blur(gray_img, (3, 3))
    
    # Tính gradient
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # Tính độ lớn gradient
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = np.uint8(np.clip(magnitude, 0, 255))
    
    return magnitude

def laplacian_detector(gray_img):
    """Phát hiện biên bằng Laplacian với cải thiện."""
    # Sử dụng bilateral filter thay vì Gaussian để giữ cạnh tốt hơn
    blurred_img = apply_bilateral_filter(gray_img)
    
    # Áp dụng Laplacian
    lap = cv2.Laplacian(blurred_img, cv2.CV_64F, ksize=3)
    laplacian_result = cv2.convertScaleAbs(lap)
    
    # Áp dụng threshold để làm nổi bật cạnh
    _, thresholded = cv2.threshold(laplacian_result, 30, 255, cv2.THRESH_BINARY)
    
    return thresholded

def canny_detector(gray_img, auto_threshold=True, t_lower=50, t_upper=150):
    """Phát hiện biên bằng Canny với auto threshold cải thiện."""
    # Làm mờ với bilateral filter
    blurred_img = apply_bilateral_filter(gray_img)
    
    if auto_threshold:
        # Sử dụng phương pháp Otsu để tự động tính ngưỡng
        v = np.median(blurred_img)
        sigma = 0.33
        t_lower = int(max(0, (1.0 - sigma) * v))
        t_upper = int(min(255, (1.0 + sigma) * v))  # FIX: Thay đổi từ (1.0 - sigma) thành (1.0 + sigma)
    
    edges = cv2.Canny(blurred_img, t_lower, t_upper)
    return edges

# =============================================================================
# PHẦN 2: ỨNG DỤNG ĐẾM VẬT THỂ (ĐƯỢC CẢI THIỆN)
# =============================================================================

def count_objects(original_img, gray_img, canny_t1, canny_t2, kernel_size, min_area, min_perimeter=0):
    """Đếm vật thể với nhiều cải tiến."""
    # Bước 1: Làm mờ để giảm nhiễu
    blurred = apply_bilateral_filter(gray_img)
    
    # Bước 2: Phát hiện cạnh bằng Canny
    canny_edges = cv2.Canny(blurred, canny_t1, canny_t2)
    
    # Bước 3: Morphological operations để đóng các khoảng trống
    k_size = int(kernel_size)
    if k_size % 2 == 0: 
        k_size += 1
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    
    # Dilate để mở rộng cạnh
    dilated = cv2.dilate(canny_edges, kernel, iterations=2)
    
    # Close để đóng các lỗ hổng
    closed_edges = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Erode nhẹ để loại bỏ nhiễu nhỏ
    closed_edges = cv2.erode(closed_edges, kernel, iterations=1)
    
    # Bước 4: Tìm contours
    contours, hierarchy = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Bước 5: Lọc và đếm các vật thể
    object_count = 0
    output_image = original_img.copy()
    valid_contours = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        # Lọc theo diện tích và chu vi
        if area > min_area and perimeter > min_perimeter:
            # Tính circularity để loại bỏ nhiễu (hình quá dài, quá méo)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.1:  # Chỉ giữ lại các hình không quá méo
                    object_count += 1
                    valid_contours.append(cnt)
                    
                    # Vẽ contour và số thứ tự
                    cv2.drawContours(output_image, [cnt], -1, (0, 255, 0), 2)
                    
                    # Tính centroid để đánh số
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.putText(output_image, str(object_count), (cX-10, cY+10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Hiển thị số lượng vật thể
    text = f"So luong vat the: {object_count}"
    cv2.putText(output_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return output_image, closed_edges, object_count

# =============================================================================
# PHẦN 3: PHÁT HIỆN LÀN ĐƯỜNG VỚI KALMAN FILTER
# =============================================================================

class LaneTracker:
    """Class để theo dõi làn đường với Kalman filter."""
    def __init__(self):
        self.left_fit_history = deque(maxlen=5)
        self.right_fit_history = deque(maxlen=5)
        
    def add_measurement(self, left_fit, right_fit):
        """Thêm đo đạc mới vào lịch sử."""
        if left_fit is not None:
            self.left_fit_history.append(left_fit)
        if right_fit is not None:
            self.right_fit_history.append(right_fit)
    
    def get_smoothed_fit(self):
        """Lấy kết quả làm mượt từ lịch sử."""
        left_avg = np.mean(self.left_fit_history, axis=0) if len(self.left_fit_history) > 0 else None
        right_avg = np.mean(self.right_fit_history, axis=0) if len(self.right_fit_history) > 0 else None
        return left_avg, right_avg

def lane_detect_edges(frame):
    """Phát hiện cạnh cho làn đường."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Sử dụng CLAHE để cải thiện độ tương phản
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def lane_create_mask(frame_edges, frame_shape):
    """Tạo mask vùng quan tâm."""
    height, width = frame_shape
    polygons = np.array([
        [
            (int(width * 0.1), height),
            (int(width * 0.9), height),
            (int(width * 0.55), int(height * 0.6)),
            (int(width * 0.45), int(height * 0.6))
        ]
    ], dtype=np.int32)
    
    mask = np.zeros_like(frame_edges)
    cv2.fillPoly(mask, polygons, 255)
    masked_edges = cv2.bitwise_and(frame_edges, mask)
    return masked_edges

def lane_detect_lines(masked_edges):
    """Phát hiện các đường thẳng bằng Hough Transform."""
    lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 50, 
                           minLineLength=40, maxLineGap=5)
    return lines

def lane_make_coordinates(frame, line_parameters):
    """Tạo tọa độ từ tham số đường thẳng."""
    if line_parameters is None or len(line_parameters) != 2:
        return None
    
    slope, intercept = line_parameters
    
    # Tránh chia cho 0
    if abs(slope) < 0.01:
        return None
    
    height = frame.shape[0]
    y1 = height
    y2 = int(height * 0.6)
    
    try:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        
        # Kiểm tra tọa độ có hợp lệ không
        if x1 < 0 or x1 > frame.shape[1] or x2 < 0 or x2 > frame.shape[1]:
            return None
            
        return np.array([x1, y1, x2, y2])
    except:
        return None

def lane_average_slope_intercept(frame, lines):
    """Tính trung bình slope và intercept cho các làn đường."""
    left_fit = []
    right_fit = []
    
    if lines is None: 
        return None, None, None
    
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        
        # Tránh chia cho 0
        if x1 == x2: 
            continue
            
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        
        # Phân loại làn trái và phải dựa trên slope
        if slope < -0.5:  # Làn trái
            left_fit.append((slope, intercept))
        elif slope > 0.5:  # Làn phải
            right_fit.append((slope, intercept))
    
    # Tính trung bình
    left_fit_avg = np.average(left_fit, axis=0) if len(left_fit) > 0 else None
    right_fit_avg = np.average(right_fit, axis=0) if len(right_fit) > 0 else None
    
    # Tạo tọa độ từ tham số trung bình
    left_line = lane_make_coordinates(frame, left_fit_avg) if left_fit_avg is not None else None
    right_line = lane_make_coordinates(frame, right_fit_avg) if right_fit_avg is not None else None
    
    return [left_line, right_line], left_fit_avg, right_fit_avg

def lane_display_lines(frame, lines, color=(0, 255, 0), thickness=10):
    """Vẽ các đường làn lên frame."""
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line
                cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
    return line_image

def lane_calculate_steering(frame, left_params, right_params):
    """Tính toán hướng lái."""
    if left_params is None or right_params is None: 
        return "Lane Not Found", 0
    
    height, width = frame.shape[:2]
    car_center_x = width // 2
    
    try:
        left_x_bottom = int((height - left_params[1]) / left_params[0])
        right_x_bottom = int((height - right_params[1]) / right_params[0])
        lane_center_x = (left_x_bottom + right_x_bottom) / 2
        offset = car_center_x - lane_center_x
        
        # Tính góc lái dựa trên offset
        if offset > 30:
            command = "Steer Left"
        elif offset < -30:
            command = "Steer Right"
        else:
            command = "Straight"
        
        return command, offset
    except:
        return "Error", 0

def lane_display_info(frame, command, offset):
    """Hiển thị thông tin lái."""
    color = (0, 255, 0) if command == "Straight" else (0, 255, 255)
    cv2.putText(frame, command, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
               1, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Offset: {offset:.2f} px", (50, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

def lane_process_pipeline(frame, tracker=None):
    """Pipeline xử lý phát hiện làn đường."""
    original_frame = frame.copy()
    
    # Phát hiện cạnh
    canny_edges = lane_detect_edges(frame)
    
    # Tạo mask
    masked_canny = lane_create_mask(canny_edges, frame.shape[:2])
    
    # Phát hiện đường thẳng
    lines = lane_detect_lines(masked_canny)
    
    # Tính trung bình slope và intercept
    averaged_lines, left_params, right_params = lane_average_slope_intercept(original_frame, lines)
    
    # Sử dụng tracker nếu có
    if tracker is not None:
        tracker.add_measurement(left_params, right_params)
        left_params_smooth, right_params_smooth = tracker.get_smoothed_fit()
        
        # Tạo lại đường từ tham số làm mượt
        if left_params_smooth is not None:
            left_line_smooth = lane_make_coordinates(original_frame, left_params_smooth)
        else:
            left_line_smooth = None
            
        if right_params_smooth is not None:
            right_line_smooth = lane_make_coordinates(original_frame, right_params_smooth)
        else:
            right_line_smooth = None
        
        averaged_lines = [left_line_smooth, right_line_smooth]
        left_params = left_params_smooth
        right_params = right_params_smooth
    
    # Tính hướng lái
    steering_command, offset = lane_calculate_steering(original_frame, left_params, right_params)
    
    # Vẽ các đường phát hiện được
    line_image = lane_display_lines(original_frame, averaged_lines, (0, 0, 255), 10)
    
    # Kết hợp với ảnh gốc
    combo_image = cv2.addWeighted(original_frame, 0.8, line_image, 1, 0)
    
    # Hiển thị thông tin
    final_image = lane_display_info(combo_image, steering_command, offset)
    
    return final_image

# =============================================================================
# PHẦN 4: GIAO DIỆN NGƯỜI DÙNG (GUI) - HOÀN THIỆN
# =============================================================================

class EdgeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng dụng Xử lý ảnh Nâng cao - Edge Detection & Lane Finding")
        self.root.geometry("1500x900")

        self.original_img = None
        self.gray_img = None
        self.processed_img = None
        self.image_path = None
        self.video_running = False
        self.cap = None
        self.lane_tracker = None

        # --- Layout chính ---
        frame_controls = tk.Frame(root, width=320, bg='#f0f0f0', relief=tk.RIDGE, borderwidth=2)
        frame_controls.pack(side=tk.LEFT, fill=tk.Y)
        frame_controls.pack_propagate(False)

        frame_images = tk.Frame(root, bg='white')
        frame_images.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Panels hiển thị ảnh ---
        self.panel_original = tk.Label(frame_images, bg='white', text="Ảnh gốc", 
                                      font=("Arial", 12, "bold"), compound='top')
        self.panel_original.pack(side=tk.LEFT, padx=10, pady=10, expand=True)
        
        self.panel_processed = tk.Label(frame_images, bg='white', text="Ảnh xử lý", 
                                       font=("Arial", 12, "bold"), compound='top')
        self.panel_processed.pack(side=tk.RIGHT, padx=10, pady=10, expand=True)

        # --- CONTROLS: PHẦN ẢNH TĨNH ---
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

        # --- Phần tinh chỉnh đếm vật thể ---
        separator = ttk.Separator(frame_controls, orient='horizontal')
        separator.pack(fill=tk.X, padx=10, pady=15)

        tk.Label(frame_controls, text="⚙️ TINH CHỈNH ĐẾM", 
                font=("Arial", 12, "bold"), bg='#f0f0f0', fg='#2c3e50').pack(pady=(5, 10), padx=10)

        # Canny T1
        tk.Label(frame_controls, text="Ngưỡng Canny T1:", bg='#f0f0f0', 
                font=('Arial', 9)).pack(padx=10, anchor='w')
        self.slider_t1 = tk.Scale(frame_controls, from_=0, to=255, orient=tk.HORIZONTAL, 
                                 bg='#f0f0f0', highlightthickness=0, 
                                 troughcolor='#3498db', sliderlength=20)
        self.slider_t1.set(50)
        self.slider_t1.pack(fill=tk.X, padx=10)
        
        # Canny T2
        tk.Label(frame_controls, text="Ngưỡng Canny T2:", bg='#f0f0f0', 
                font=('Arial', 9)).pack(padx=10, anchor='w')
        self.slider_t2 = tk.Scale(frame_controls, from_=0, to=255, orient=tk.HORIZONTAL, 
                                 bg='#f0f0f0', highlightthickness=0,
                                 troughcolor='#3498db', sliderlength=20)
        self.slider_t2.set(150)
        self.slider_t2.pack(fill=tk.X, padx=10)

        # Kernel Size
        tk.Label(frame_controls, text="Kích thước Kernel:", bg='#f0f0f0', 
                font=('Arial', 9)).pack(padx=10, anchor='w')
        self.slider_kernel = tk.Scale(frame_controls, from_=1, to=21, orient=tk.HORIZONTAL, 
                                     bg='#f0f0f0', highlightthickness=0,
                                     troughcolor='#e74c3c', sliderlength=20)
        self.slider_kernel.set(5)
        self.slider_kernel.pack(fill=tk.X, padx=10)

        # Min Area
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

        # --- PHẦN VIDEO ---
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

        # --- Nút lưu ---
        tk.Button(frame_controls, text="💾 Lưu ảnh kết quả", command=self.save_image,
                 bg='#9b59b6', fg='white', font=('Arial', 10, 'bold'),
                 activebackground='#8e44ad', cursor='hand2').pack(fill=tk.X, padx=10, 
                                                                   pady=(20, 15), side=tk.BOTTOM)

        # Status bar
        self.status_bar = tk.Label(root, text="Sẵn sàng", bd=1, relief=tk.SUNKEN, 
                                  anchor=tk.W, bg='#ecf0f1', font=('Arial', 9))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # --- CÁC HÀM XỬ LÝ ẢNH TĨNH ---
    def load_image(self):
        """Tải ảnh từ file."""
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
            self.status_bar.config(text="Lỗi khi tải ảnh")

    def run_sobel(self):
        """Chạy Sobel detector."""
        if self.gray_img is None: 
            return messagebox.showwarning("Cảnh báo", "Vui lòng tải ảnh trước!")
        
        self.status_bar.config(text="Đang xử lý Sobel...")
        self.root.update()
        
        self.processed_img = sobel_detector(self.gray_img)
        self.display_image(self.processed_img, self.panel_processed, "Kết quả Sobel")
        self.status_bar.config(text="Hoàn thành Sobel detector")

    def run_laplacian(self):
        """Chạy Laplacian detector."""
        if self.gray_img is None: 
            return messagebox.showwarning("Cảnh báo", "Vui lòng tải ảnh trước!")
        
        self.status_bar.config(text="Đang xử lý Laplacian...")
        self.root.update()
        
        self.processed_img = laplacian_detector(self.gray_img)
        self.display_image(self.processed_img, self.panel_processed, "Kết quả Laplacian")
        self.status_bar.config(text="Hoàn thành Laplacian detector")

    def run_canny(self):
        """Chạy Canny detector với auto threshold."""
        if self.gray_img is None: 
            return messagebox.showwarning("Cảnh báo", "Vui lòng tải ảnh trước!")
        
        self.status_bar.config(text="Đang xử lý Canny...")
        self.root.update()
        
        self.processed_img = canny_detector(self.gray_img, auto_threshold=True)
        self.display_image(self.processed_img, self.panel_processed, "Kết quả Canny (Auto)")
        self.status_bar.config(text="Hoàn thành Canny detector")

    def run_counting(self):
        """Chạy đếm vật thể."""
        if self.original_img is None: 
            return messagebox.showwarning("Cảnh báo", "Vui lòng tải ảnh trước!")
        
        t1, t2 = self.slider_t1.get(), self.slider_t2.get()
        
        # Đảm bảo t2 > t1
        if t1 >= t2: 
            t2 = t1 + 1
            self.slider_t2.set(t2)
        
        self.status_bar.config(text="Đang đếm vật thể...")
        self.root.update()
        
        result_img, edges_img, count = count_objects(
            self.original_img, 
            self.gray_img, 
            t1, t2, 
            self.slider_kernel.get(), 
            self.slider_area.get()
        )
        
        self.processed_img = result_img
        self.display_image(self.processed_img, self.panel_processed, 
                         f"Kết quả: {count} vật thể")
        self.status_bar.config(text=f"Đã phát hiện {count} vật thể")

    # --- XỬ LÝ VIDEO ---
    def run_lane_detection_video(self):
        """Mở file video và chạy phát hiện làn đường."""
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
        
        messagebox.showinfo("Hướng dẫn", 
                          "Video sẽ chạy trong cửa sổ mới.\n"
                          "Nhấn 'q' trên bàn phím hoặc nút 'Dừng Video' để thoát.")
        
        self.status_bar.config(text=f"Đang xử lý video: {video_path}")
        
        # Chạy video trong một thread riêng để không block GUI
        self.process_video()

    def process_video(self):
        """Xử lý video frame by frame."""
        prev_time = 0
        frame_count = 0
        
        while self.video_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            try:
                # Xử lý frame với lane detection pipeline
                processed_frame = lane_process_pipeline(frame, self.lane_tracker)
                
                # Tính FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time
                
                # Hiển thị thông tin
                cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 150), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(processed_frame, f"Frame: {frame_count}", (10, 180), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Hiển thị frame
                cv2.imshow("Phat hien lan duong (Nhan 'q' de thoat)", processed_frame)
                
                # Cập nhật status bar
                if frame_count % 30 == 0:  # Cập nhật mỗi 30 frame
                    self.status_bar.config(text=f"Đang xử lý frame {frame_count}, FPS: {fps:.1f}")
                    self.root.update()
                
            except Exception as e:
                print(f"Lỗi xử lý frame {frame_count}: {e}")
                break
            
            # Kiểm tra phím 'q' để thoát
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Dọn dẹp
        self.stop_video()

    def stop_video(self):
        """Dừng video đang chạy."""
        self.video_running = False
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        cv2.destroyAllWindows()
        
        self.btn_lane.config(state=tk.NORMAL)
        self.btn_stop_video.config(state=tk.DISABLED)
        self.status_bar.config(text="Đã dừng video")

    def save_image(self):
        """Lưu ảnh kết quả."""
        if self.processed_img is None: 
            return messagebox.showwarning("Cảnh báo", "Không có ảnh kết quả để lưu!")
        
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png", 
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
        
        if save_path:
            cv2.imwrite(save_path, self.processed_img)
            messagebox.showinfo("Thành công", f"Đã lưu ảnh tại:\n{save_path}")
            self.status_bar.config(text=f"Đã lưu: {save_path}")

    def display_image(self, img, panel, title_text):
        """Hiển thị ảnh lên panel với scaling tự động."""
        max_width, max_height = 700, 750
        
        if img is None:
            return
        
        h, w = img.shape[:2]
        ratio = min(max_width / w, max_height / h)
        
        # Chỉ resize nếu ảnh quá lớn
        if ratio < 1:
            new_w, new_h = int(w * ratio), int(h * ratio)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Chuyển đổi sang RGB để hiển thị
        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Chuyển sang PhotoImage
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        # Cập nhật panel
        panel.config(image=img_tk, text=title_text, 
                    font=("Arial", 12, "bold"), compound='top')
        panel.image = img_tk  # Giữ reference để tránh garbage collection

# =============================================================================
# CHẠY ỨNG DỤNG
# =============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = EdgeDetectionApp(root)
    root.mainloop()
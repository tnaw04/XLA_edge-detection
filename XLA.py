import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import time

# =============================================================================
# PHẦN 1: CÁC HÀM XỬ LÝ ẢNH CỐT LÕI (CHO ẢNH TĨNH)
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

def sobel_detector(gray_img):
    """Phát hiện biên bằng Sobel."""
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    sobel_combined = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return sobel_combined

def laplacian_detector(gray_img):
    """Phát hiện biên bằng Laplacian."""
    blurred_img = apply_gaussian_blur(gray_img)
    lap = cv2.Laplacian(blurred_img, cv2.CV_64F, ksize=3)
    laplacian_result = cv2.convertScaleAbs(lap)
    return laplacian_result

def canny_detector(gray_img, auto_threshold=True, t_lower=50, t_upper=150):
    """Phát hiện biên bằng Canny."""
    blurred_img = apply_gaussian_blur(gray_img)
    if auto_threshold:
        v = np.median(blurred_img)
        sigma = 0.33
        t_lower = int(max(0, (1.0 - sigma) * v))
        t_upper = int(min(255, (1.0 - sigma) * v))
    edges = cv2.Canny(blurred_img, t_lower, t_upper)
    return edges

# =============================================================================
# PHẦN 2: ỨNG DỤNG ĐẾM VẬT THỂ (CHO ẢNH TĨNH)
# =============================================================================

def count_objects(original_img, gray_img, canny_t1, canny_t2, kernel_size, min_area):
    canny_edges = canny_detector(gray_img, auto_threshold=False, t_lower=canny_t1, t_upper=canny_t2)
    k_size = int(kernel_size)
    if k_size % 2 == 0: k_size += 1 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    closed_edges = cv2.morphologyEx(canny_edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    object_count = 0
    output_image = original_img.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            object_count += 1
            cv2.drawContours(output_image, [cnt], -1, (0, 255, 0), 2)
    text = f"So luong vat the: {object_count}"
    cv2.putText(output_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return output_image, closed_edges, object_count

# =============================================================================
# PHẦN 3: CÁC HÀM PHÁT HIỆN LÀN ĐƯỜNG (TỪ FILE LaneCanny.py)
# =============================================================================

def lane_detect_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def lane_create_mask(frame_edges, frame_shape):
    height, width = frame_shape
    polygons = np.array([
        (int(width * 0.1), height),
        (int(width * 0.9), height),
        (int(width * 0.55), int(height * 0.6)),
        (int(width * 0.45), int(height * 0.6))
    ])
    mask = np.zeros_like(frame_edges)
    cv2.fillPoly(mask, [polygons], 255)
    masked_edges = cv2.bitwise_and(frame_edges, mask)
    return masked_edges

def lane_detect_lines(masked_edges):
    lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 50, minLineLength=40, maxLineGap=5)
    return lines

def lane_display_detected(frame, lines):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return combo_image

def lane_make_coordinates(frame, line_parameters):
    slope, intercept = line_parameters
    height = frame.shape[0]
    y1 = height
    y2 = int(height * 0.6)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def lane_average_slope_intercept(frame, lines):
    left_fit = []
    right_fit = []
    if lines is None: return None, None, None
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x1 == x2: continue
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < -0.5: left_fit.append((slope, intercept))
        elif slope > 0.5: right_fit.append((slope, intercept))
            
    left_line = lane_make_coordinates(frame, np.average(left_fit, axis=0)) if len(left_fit) > 0 else None
    right_line = lane_make_coordinates(frame, np.average(right_fit, axis=0)) if len(right_fit) > 0 else None
    return [left_line, right_line], (np.average(left_fit, axis=0) if len(left_fit) > 0 else None), (np.average(right_fit, axis=0) if len(right_fit) > 0 else None)

def lane_display_averaged(frame, lines):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
    combo_image = cv2.addWeighted(frame, 1, line_image, 1, 0)
    return combo_image

def lane_calculate_steering(frame, left_params, right_params):
    if left_params is None or right_params is None: return "Lane Not Found", 0
    height, width = frame.shape[:2]
    car_center_x = width // 2
    left_x_bottom = int((height - left_params[1]) / left_params[0])
    right_x_bottom = int((height - right_params[1]) / right_params[0])
    lane_center_x = (left_x_bottom + right_x_bottom) / 2
    offset = car_center_x - lane_center_x
    
    if offset > 20: command = "Steer Left"
    elif offset < -20: command = "Steer Right"
    else: command = "Straight"
    return command, offset

def lane_display_info(frame, command, offset):
    color = (0, 255, 0) if command == "Straight" else (0, 255, 255)
    cv2.putText(frame, command, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Offset: {offset:.2f} px", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return frame

def lane_process_pipeline(frame):
    original_frame = frame.copy()
    canny_edges = lane_detect_edges(frame)
    masked_canny = lane_create_mask(canny_edges, frame.shape[:2])
    lines = lane_detect_lines(masked_canny)
    averaged_lines, left_params, right_params = lane_average_slope_intercept(original_frame, lines)
    steering_command, offset = lane_calculate_steering(original_frame, left_params, right_params)
    detected_line_image = lane_display_detected(original_frame, lines)
    averaged_line_image = lane_display_averaged(detected_line_image, averaged_lines)
    final_image = lane_display_info(averaged_line_image, steering_command, offset)
    return final_image

# =============================================================================
# PHẦN 4: GIAO DIỆN NGƯỜI DÙNG (GUI) - ĐÃ CẬP NHẬT
# =============================================================================

class EdgeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng dụng Xử lý ảnh Hợp nhất (Edge Detection & Lane Finding)")
        self.root.geometry("1400x850")

        self.original_img = None
        self.gray_img = None
        self.processed_img = None
        self.image_path = None

        # --- Layout ---
        frame_controls = tk.Frame(root, width=300, bg='lightgray')
        frame_controls.pack(side=tk.LEFT, fill=tk.Y)
        frame_controls.pack_propagate(False)

        frame_images = tk.Frame(root)
        frame_images.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.panel_original = tk.Label(frame_images)
        self.panel_original.pack(side=tk.LEFT, padx=10, pady=10, expand=True)
        self.panel_processed = tk.Label(frame_images)
        self.panel_processed.pack(side=tk.RIGHT, padx=10, pady=10, expand=True)

        # --- Controls ---
        lbl_title = tk.Label(frame_controls, text="Chức năng Ảnh Tĩnh", font=("Arial", 14, "bold"), bg='lightgray')
        lbl_title.pack(pady=(10, 5), padx=10)

        tk.Button(frame_controls, text="1. Tải ảnh", command=self.load_image).pack(fill=tk.X, padx=10, pady=2)
        tk.Button(frame_controls, text="2. Chạy Sobel", command=self.run_sobel).pack(fill=tk.X, padx=10, pady=2)
        tk.Button(frame_controls, text="3. Chạy Laplacian", command=self.run_laplacian).pack(fill=tk.X, padx=10, pady=2)
        tk.Button(frame_controls, text="4. Chạy Canny (Auto)", command=self.run_canny).pack(fill=tk.X, padx=10, pady=2)

        # --- Tinh chỉnh Đếm ---
        tk.Label(frame_controls, text="Tinh chỉnh Đếm Vật Thể", font=("Arial", 12, "bold"), bg='lightgray').pack(pady=(15, 5), padx=10)
        tk.Label(frame_controls, text="Canny T1:", bg='lightgray').pack(padx=10, anchor='w')
        self.slider_t1 = tk.Scale(frame_controls, from_=0, to=255, orient=tk.HORIZONTAL, bg='lightgray', highlightthickness=0)
        self.slider_t1.set(50)
        self.slider_t1.pack(fill=tk.X, padx=10)
        
        tk.Label(frame_controls, text="Canny T2:", bg='lightgray').pack(padx=10, anchor='w')
        self.slider_t2 = tk.Scale(frame_controls, from_=0, to=255, orient=tk.HORIZONTAL, bg='lightgray', highlightthickness=0)
        self.slider_t2.set(150)
        self.slider_t2.pack(fill=tk.X, padx=10)

        tk.Label(frame_controls, text="Kernel Size:", bg='lightgray').pack(padx=10, anchor='w')
        self.slider_kernel = tk.Scale(frame_controls, from_=1, to=21, orient=tk.HORIZONTAL, bg='lightgray', highlightthickness=0)
        self.slider_kernel.set(5)
        self.slider_kernel.pack(fill=tk.X, padx=10)

        tk.Label(frame_controls, text="Min Area:", bg='lightgray').pack(padx=10, anchor='w')
        self.slider_area = tk.Scale(frame_controls, from_=0, to=2000, orient=tk.HORIZONTAL, bg='lightgray', highlightthickness=0)
        self.slider_area.set(100)
        self.slider_area.pack(fill=tk.X, padx=10)

        tk.Button(frame_controls, text="5. Đếm Vật thể (Canny)", command=self.run_counting, bg='orange').pack(fill=tk.X, padx=10, pady=5)
        # ĐÃ XÓA NÚT SỐ 6 (OTSU)

        # --- PHẦN MỚI: LANE DETECTION ---
        lbl_video = tk.Label(frame_controls, text="Chức năng Video", font=("Arial", 14, "bold"), bg='lightgray')
        lbl_video.pack(pady=(20, 5), padx=10)

        btn_lane = tk.Button(frame_controls, text="6. Phát hiện làn đường (Video)", 
                             command=self.run_lane_detection_video, bg='lightgreen', font=("Arial", 11, "bold"))
        btn_lane.pack(fill=tk.X, padx=10, pady=10)

        tk.Button(frame_controls, text="Lưu ảnh kết quả (Tĩnh)", command=self.save_image).pack(fill=tk.X, padx=10, pady=(20, 10), side=tk.BOTTOM)

    # --- Các hàm xử lý ảnh tĩnh (Giữ nguyên từ XLA.py) ---
    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not self.image_path: return
        try:
            self.original_img, self.gray_img = load_image_grayscale(self.image_path)
            self.display_image(self.original_img, self.panel_original, "Ảnh gốc")
            self.display_image(self.gray_img, self.panel_processed, "Ảnh xám")
        except Exception as e: messagebox.showerror("Lỗi", f"Không thể tải ảnh: {e}")

    def run_sobel(self):
        if self.gray_img is None: return messagebox.showwarning("Cảnh báo", "Vui lòng tải ảnh trước.")
        self.processed_img = sobel_detector(self.gray_img)
        self.display_image(self.processed_img, self.panel_processed, "Kết quả Sobel")

    def run_laplacian(self):
        if self.gray_img is None: return messagebox.showwarning("Cảnh báo", "Vui lòng tải ảnh trước.")
        self.processed_img = laplacian_detector(self.gray_img)
        self.display_image(self.processed_img, self.panel_processed, "Kết quả Laplacian")

    def run_canny(self):
        if self.gray_img is None: return messagebox.showwarning("Cảnh báo", "Vui lòng tải ảnh trước.")
        self.processed_img = canny_detector(self.gray_img, auto_threshold=True)
        self.display_image(self.processed_img, self.panel_processed, "Kết quả Canny (Auto)")

    def run_counting(self):
        if self.original_img is None: return messagebox.showwarning("Cảnh báo", "Vui lòng tải ảnh trước.")
        t1, t2 = self.slider_t1.get(), self.slider_t2.get()
        if t1 >= t2: 
            t2 = t1 + 1
            self.slider_t2.set(t2)
        result_img, _, count = count_objects(self.original_img, self.gray_img, t1, t2, self.slider_kernel.get(), self.slider_area.get())
        self.processed_img = result_img
        self.display_image(self.processed_img, self.panel_processed, f"Kết quả Đếm: {count} vật thể")

    # --- HÀM MỚI: XỬ LÝ VIDEO ---
    def run_lane_detection_video(self):
        """Mở file video và chạy pipeline phát hiện làn đường."""
        video_path = filedialog.askopenfilename(title="Chọn file video", filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if not video_path: return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Lỗi", f"Không thể mở video: {video_path}")
            return

        messagebox.showinfo("Hướng dẫn", "Video sẽ chạy trong cửa sổ mới.\nNhấn phím 'q' trên cửa sổ video để thoát và quay lại ứng dụng chính.")

        prev_time = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break # Kết thúc video

            # Gọi pipeline xử lý từ phần 3
            try:
                processed_frame = lane_process_pipeline(frame)
            except Exception as e:
                print(f"Lỗi xử lý frame: {e}")
                break

            # Tính FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Phat hien lan duong (Nhan 'q' de thoat)", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

    def save_image(self):
        if self.processed_img is None: return messagebox.showwarning("Cảnh báo", "Không có ảnh kết quả để lưu.")
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
        if save_path:
            cv2.imwrite(save_path, self.processed_img)
            messagebox.showinfo("Thành công", f"Đã lưu ảnh tại: {save_path}")

    def display_image(self, img, panel, title_text):
        max_width, max_height = 650, 750
        h, w = img.shape[:2]
        ratio = min(max_width / w, max_height / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        if ratio < 1: img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        panel.config(image=img_tk, text=title_text, font=("Arial", 12), compound='top')
        panel.image = img_tk

if __name__ == "__main__":
    root = tk.Tk()
    app = EdgeDetectionApp(root)
    root.mainloop()
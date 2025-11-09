import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

# --- CÁC HÀM XỬ LÝ ẢNH CỐT LÕI ---

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

# --- 1. CÁC THUẬT TOÁN PHÁT HIỆN BIÊN ---

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

# --- 2. ỨNG DỤNG ĐẾM VẬT THỂ (THEO ĐỀ TÀI 2) ---

def count_objects(original_img, gray_img, canny_t1, canny_t2, kernel_size, min_area):
    """
    Pipeline đếm vật thể (TUÂN THỦ ĐỀ TÀI) với các tham số 
    có thể tinh chỉnh để tăng độ chính xác.
    """
    
    # 1. & 2. Phát hiện biên Canny với ngưỡng tùy chỉnh
    # Dùng hàm canny_detector nhưng truyền ngưỡng vào
    canny_edges = canny_detector(gray_img, 
                                 auto_threshold=False, 
                                 t_lower=canny_t1, 
                                 t_upper=canny_t2)

    # 3. Xử lý Hình thái học (Morphology)
    # Tạo kernel với kích thước tùy chỉnh
    # Đảm bảo kernel_size là số lẻ (yêu cầu của cv2.getStructuringElement)
    k_size = int(kernel_size)
    if k_size % 2 == 0: 
        k_size += 1 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    
    # Phép Đóng (Closing) để nối các biên bị đứt
    closed_edges = cv2.morphologyEx(canny_edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 4. Tìm Contours
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. Lọc và Đếm (với diện tích tối thiểu tùy chỉnh)
    object_count = 0
    output_image = original_img.copy()
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area: # Sử dụng min_area từ slider
            object_count += 1
            cv2.drawContours(output_image, [cnt], -1, (0, 255, 0), 2) # Vẽ contour xanh
            
    # 6. Trực quan hóa
    text = f"So luong vat the: {object_count}"
    cv2.putText(output_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return output_image, closed_edges, object_count

# --- 3. PHƯƠNG PHÁP NÂNG CAO (ĐỂ SO SÁNH) ---

def count_objects_advanced(original_img, gray_img):
    """
    Pipeline NÂNG CAO để đếm vật thể, sử dụng Phân ngưỡng Otsu.
    (Dùng để so sánh trong báo cáo)
    """
    blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)
    (T, thresh_inv) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned_mask = cv2.morphologyEx(thresh_inv, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    object_count = 0
    min_area = 100
    output_image = original_img.copy()
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            object_count += 1
            cv2.drawContours(output_image, [cnt], -1, (0, 255, 0), 2)
            
    text = f"So luong (Otsu): {object_count}"
    cv2.putText(output_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return output_image, cleaned_mask, object_count

# --- GIAO DIỆN NGƯỜI DÙNG (GUI) ---

class EdgeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bài tập lớn Xử lý ảnh - Đề tài 2 (v2.0 - Tinh chỉnh)")
        self.root.geometry("1400x800") # Tăng kích thước cửa sổ

        self.original_img = None
        self.gray_img = None
        self.processed_img = None
        self.image_path = None

        # --- Tạo khung (Frames) ---
        frame_controls = tk.Frame(root, width=300, bg='lightgray') # Tăng chiều rộng khung control
        frame_controls.pack(side=tk.LEFT, fill=tk.Y)
        frame_controls.pack_propagate(False) # Ngăn co lại

        frame_images = tk.Frame(root)
        frame_images.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Khung ảnh
        self.panel_original = tk.Label(frame_images)
        self.panel_original.pack(side=tk.LEFT, padx=10, pady=10, expand=True)
        self.panel_processed = tk.Label(frame_images)
        self.panel_processed.pack(side=tk.RIGHT, padx=10, pady=10, expand=True)

        # --- Thêm các nút điều khiển ---
        lbl_title = tk.Label(frame_controls, text="Chức năng", font=("Arial", 16), bg='lightgray')
        lbl_title.pack(pady=10, padx=10)

        btn_load = tk.Button(frame_controls, text="1. Tải ảnh", command=self.load_image)
        btn_load.pack(fill=tk.X, padx=10, pady=5)

        btn_sobel = tk.Button(frame_controls, text="2. Chạy Sobel", command=self.run_sobel)
        btn_sobel.pack(fill=tk.X, padx=10, pady=5)

        btn_laplacian = tk.Button(frame_controls, text="3. Chạy Laplacian", command=self.run_laplacian)
        btn_laplacian.pack(fill=tk.X, padx=10, pady=5)

        btn_canny = tk.Button(frame_controls, text="4. Chạy Canny (Auto)", command=self.run_canny)
        btn_canny.pack(fill=tk.X, padx=10, pady=5)

        # --- PHẦN CẢI TIẾN: TINH CHỈNH ỨNG DỤNG ---
        lbl_tune = tk.Label(frame_controls, text="Tinh chỉnh Ứng dụng Đếm", font=("Arial", 12), bg='lightgray')
        lbl_tune.pack(pady=(20, 5), padx=10)

        # Slider cho Canny Ngưỡng Thấp (T1)
        lbl_t1 = tk.Label(frame_controls, text="Canny T1 (Thấp):", bg='lightgray')
        lbl_t1.pack(padx=10, anchor='w')
        self.slider_t1 = tk.Scale(frame_controls, from_=0, to=255, orient=tk.HORIZONTAL, bg='lightgray', highlightthickness=0)
        self.slider_t1.set(50) # Giá trị mặc định
        self.slider_t1.pack(fill=tk.X, padx=10)

        # Slider cho Canny Ngưỡng Cao (T2)
        lbl_t2 = tk.Label(frame_controls, text="Canny T2 (Cao):", bg='lightgray')
        lbl_t2.pack(padx=10, anchor='w')
        self.slider_t2 = tk.Scale(frame_controls, from_=0, to=255, orient=tk.HORIZONTAL, bg='lightgray', highlightthickness=0)
        self.slider_t2.set(150) # Giá trị mặc định
        self.slider_t2.pack(fill=tk.X, padx=10)
        
        # Slider cho Kích thước Kernel (Morphology)
        lbl_kernel = tk.Label(frame_controls, text="Kernel Size (Closing):", bg='lightgray')
        lbl_kernel.pack(padx=10, anchor='w')
        self.slider_kernel = tk.Scale(frame_controls, from_=1, to=21, orient=tk.HORIZONTAL, bg='lightgray', highlightthickness=0)
        self.slider_kernel.set(5) # Giá trị mặc định
        self.slider_kernel.pack(fill=tk.X, padx=10)

        # Slider cho Diện tích tối thiểu (Min Area)
        lbl_area = tk.Label(frame_controls, text="Min Area (Lọc nhiễu):", bg='lightgray')
        lbl_area.pack(padx=10, anchor='w')
        self.slider_area = tk.Scale(frame_controls, from_=0, to=2000, orient=tk.HORIZONTAL, bg='lightgray', highlightthickness=0) # Tăng max area
        self.slider_area.set(100) # Giá trị mặc định
        self.slider_area.pack(fill=tk.X, padx=10)

        # Nút để chạy ứng dụng (giống nút 5 cũ)
        btn_count = tk.Button(frame_controls, text="5. Ứng dụng: Đếm (Canny)", 
                              command=self.run_counting, bg='orange', font=("Arial", 10, "bold"))
        btn_count.pack(fill=tk.X, padx=10, pady=10)
        
        # Nút đếm nâng cao (Otsu) - Giữ lại để so sánh
        btn_count_adv = tk.Button(frame_controls, text="6. Đếm (So sánh Otsu)", 
                                  command=self.run_counting_advanced, bg='lightblue')
        btn_count_adv.pack(fill=tk.X, padx=10, pady=5)
        
        btn_save = tk.Button(frame_controls, text="Lưu ảnh kết quả", command=self.save_image)
        btn_save.pack(fill=tk.X, padx=10, pady=(10, 5), side=tk.BOTTOM)

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not self.image_path:
            return
            
        try:
            self.original_img, self.gray_img = load_image_grayscale(self.image_path)
            self.display_image(self.original_img, self.panel_original, "Ảnh gốc")
            self.display_image(self.gray_img, self.panel_processed, "Ảnh xám")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải ảnh: {e}")

    def run_sobel(self):
        if self.gray_img is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng tải ảnh trước.")
            return
        self.processed_img = sobel_detector(self.gray_img)
        self.display_image(self.processed_img, self.panel_processed, "Kết quả Sobel")

    def run_laplacian(self):
        if self.gray_img is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng tải ảnh trước.")
            return
        self.processed_img = laplacian_detector(self.gray_img)
        self.display_image(self.processed_img, self.panel_processed, "Kết quả Laplacian")

    def run_canny(self):
        if self.gray_img is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng tải ảnh trước.")
            return
        self.processed_img = canny_detector(self.gray_img, auto_threshold=True)
        self.display_image(self.processed_img, self.panel_processed, "Kết quả Canny (Auto)")

    def run_counting(self):
        """Chạy ứng dụng đếm (Đề tài 2) với tham số từ Sliders."""
        if self.original_img is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng tải ảnh trước.")
            return
            
        # Lấy giá trị từ sliders
        t1 = self.slider_t1.get()
        t2 = self.slider_t2.get()
        k_size = self.slider_kernel.get()
        min_area = self.slider_area.get()

        # Đảm bảo T1 < T2 (Yêu cầu logic của Canny)
        if t1 >= t2:
            t2 = t1 + 1
            self.slider_t2.set(t2) # Tự động cập nhật slider T2

        # Gọi hàm count_objects đã được nâng cấp
        result_img, _, count = count_objects(self.original_img, 
                                             self.gray_img, 
                                             t1, 
                                             t2, 
                                             k_size, 
                                             min_area)
        
        self.processed_img = result_img
        self.display_image(self.processed_img, self.panel_processed, f"Kết quả Đếm: {count} vật thể")

    def run_counting_advanced(self):
        """Chạy phương pháp Otsu để so sánh."""
        if self.original_img is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng tải ảnh trước.")
            return
        result_img, _, count = count_objects_advanced(self.original_img, self.gray_img)
        self.processed_img = result_img
        self.display_image(self.processed_img, self.panel_processed, f"Kết quả Đếm (Otsu): {count} vật thể")

    def save_image(self):
        if self.processed_img is None:
            messagebox.showwarning("Cảnh báo", "Không có ảnh kết quả để lưu.")
            return
        
        save_path = filedialog.asksaveasfilename(defaultextension=".png", 
                                                 filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
        if not save_path:
            return
            
        try:
            cv2.imwrite(save_path, self.processed_img)
            messagebox.showinfo("Thành công", f"Đã lưu ảnh tại: {save_path}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu ảnh: {e}")

    def display_image(self, img, panel, title_text):
        """Hiển thị ảnh (OpenCV) lên Label (Tkinter)."""
        # Thay đổi kích thước ảnh để vừa với cửa sổ
        max_width = 650
        max_height = 750
        h, w = img.shape[:2]
        ratio = min(max_width / w, max_height / h)
        if ratio > 1: # Không phóng to ảnh
            ratio = 1
            
        resized_img = cv2.resize(img, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_AREA)

        # Chuyển đổi màu từ BGR (OpenCV) sang RGB (PIL)
        if len(resized_img.shape) == 3:
            img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        else: # Ảnh xám
            img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)

        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)

        panel.config(image=img_tk, text=title_text, font=("Arial", 12), compound='top')
        panel.image = img_tk

# --- Chạy ứng dụng ---
if __name__ == "__main__":
    root = tk.Tk()
    app = EdgeDetectionApp(root)
    root.mainloop()
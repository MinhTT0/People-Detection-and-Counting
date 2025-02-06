import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
from tkinter import Frame,Label, Tk, Button, Scale, HORIZONTAL, filedialog, messagebox, Radiobutton, StringVar, Entry,ttk
from tkinter.filedialog import asksaveasfilename  # Nhập asksaveasfilename
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
points = []
drawing = False
pause = False  # Biến trạng thái để kiểm tra xem video đang tạm dừng hay không
confidence_threshold=0
detection_mode="vertical"
model_path = "yolo11n.pt"
def update_threshold(value):
    global confidence_threshold
    confidence_threshold = float(value) / 100  # Chuyển đổi từ % sang giá trị thực (0.0 - 1.0)
def update_mode():
    global detection_mode
    detection_mode = mode_var.get()  # Cập nhật chế độ phát hiện
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_entered = 0
    global pause
    count=0
    pause = False
    save_video = False
    previous_time = time.time()  # Thời gian trước đó để tính FPS
    video_name = os.path.basename(video_path).split('.')[0]
    save_dir = os.path.join("output_images", video_name)
    image_dir = os.path.join(save_dir, "image")
    # video_save_path = os.path.join(save_dir, f"{video_name}_output.mp4")
    current_count = 0
    entered_ids = set()
    model =YOLO(model_path)
    os.makedirs(image_dir, exist_ok=True)
    # Function để vẽ vùng đa giác
    def draw_polygon(event, x, y, flags, param):
        global points, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            if len(points) > 0:
                cv2.line(frame, points[-1], (x, y), (0, 255, 0), 1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if len(points) > 1 and cv2.norm(np.array(points[-1]) - np.array(points[0])) < 10:
                points.append(points[0])  # Đóng vòng
    # Xóa vùng vẽ
    def delete_polygon():
        points.clear()
        current_count = 0
    cv2.namedWindow('RGB')
    cv2.setMouseCallback('RGB', draw_polygon)
    if messagebox.askyesno(u"Lưu video", u"Bạn có muốn lưu video không?"):
        frame_width = 1050
        frame_height = 600
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vid_dir = filedialog.askdirectory(title="Chọn thư mục để lưu video")
        if not vid_dir:  # Nếu người dùng không chọn thư mục
            print("Không có thư mục nào được chọn. Thoát chương trình.")
            return
        video_save_path = os.path.join(vid_dir, f"{video_name}_output.mp4")
        out = cv2.VideoWriter(video_save_path, fourcc, fps, (frame_width, frame_height))
        save_video = True
    try:
        while True:
            # Kiểm tra trạng thái tạm dừng
            if not pause:
                ret, frame = cap.read()
                cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                if not ret:
                    break
                count += 1
                if count % 3 != 0:
                    continue
                current_time = time.time()
                frame_time = current_time - previous_time
                previous_time = current_time
                fps_display = 1 / frame_time if frame_time > 0 else 0
                # Resize frame
                frame_riel = cv2.resize(frame, (1050, 600))
                frame_fax = cv2.resize(frame, (1050, 600))
                # Run YOLO tracking
                results = model.track(frame_riel,tracker="botsort.yaml",persist=True, classes=0, imgsz=640)
                # results = model.track(frame_riel,tracker="bytetrack.yaml",persist=True, classes=0, imgsz=640)
                # Vẽ vùng đa giác
                if len(points) > 1:
                    cv2.polylines(frame_riel, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)
                # Kiểm tra người trong vùng vẽ
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.int().cpu().tolist()
                    class_ids = results[0].boxes.cls.int().cpu().tolist()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    confidences = results[0].boxes.conf.float().cpu().tolist()
                    current_count = 0
                    for box, class_id, track_id, confidence in zip(boxes, class_ids, track_ids, confidences):
                        if confidence < confidence_threshold:
                            continue
                        x1, y1, x2, y2 = box
                        if detection_mode == "vertical":  # Chế độ dọc
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        elif detection_mode == "horizontal":  # Chế độ ngang
                            cx, cy = x1, y2
                        if len(points) >= 2:
                            if cv2.pointPolygonTest(np.array(points), (cx, cy), False) >= 0:
                                current_count += 1
                                if track_id not in entered_ids:
                                    total_entered += 1
                                    entered_ids.add(track_id)
                                    person_image = frame_fax[y1:y2, x1:x2]
                                    resize_person_image = cv2.resize(person_image, (548, 548))
                                    person_image_path = os.path.join(image_dir, f"{track_id}.jpg")
                                    cv2.imwrite(person_image_path, resize_person_image)
                        cv2.circle(frame_riel, (cx, cy), 4, (255, 0, 0), -1)
                        cv2.rectangle(frame_riel, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        #Rtext = f'ID: {track_id} Conf: {confidence:.2f}'
                        text = f'ID: {track_id}'
                        cvzone.putTextRect(frame_riel, text, (x1, y1), 1, 1,offset=1)
                # Hiển thị số lượng người trong vùng vẽ
                if len(points) > 2:
                    polygon_center = tuple(np.mean(np.array(points), axis=0).astype(int))
                    cv2.putText(frame_riel, f"{current_count}", polygon_center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # Hiển thị FPS
                cvzone.putTextRect(frame_riel, f'TOTAL: {total_entered}', (50, 50), scale=1, thickness=1)
                cvzone.putTextRect(frame_riel, f'FPS: {fps_display:.2f}', (50, 110), scale=1, thickness=1)
                #print(entered_ids)
                if save_video:
                    out.write(frame_riel)
                cv2.imshow("RGB", frame_riel)   
            # Kiểm tra phím bấm
            key = cv2.waitKey(1)
            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                entered_ids.clear()
                break
            elif key == ord("r"):
                delete_polygon()
            elif key == 32:  # Phím Space để tạm dừng
                pause = not pause

            # Trong khi tạm dừng, tiếp tục chờ phím bấm
            while pause:
                key = cv2.waitKey(1)
                if key == 32:  # Phím Space để tiếp tục
                    pause = not pause
                    break
                elif key == ord("q"):  # Thoát chương trình
                    pause = False
                    break
    finally:
        cap.release()
        delete_polygon()
        if save_video:
            out.release()
        entered_ids.clear()
        cv2.destroyAllWindows()
def process_image(image_path):
    # Đọc ảnh
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (548, 548))
    # Dự đoán kết quả
    model =YOLO(model_path)
    results = model(resized_image)
    count=0
    # Lấy danh sách các bounding box và nhãn
    detections = results[0].boxes  # Chứa bounding boxes
    for box in detections:
        x1, y1, x2, y2 = box.xyxy[0]  # Toạ độ góc trái trên và phải dưới
        conf = box.conf[0]  # Độ tin cậy
        cls = int(box.cls[0])  # Lớp của vật thể
        conf = box.conf[0]  # Độ tin cậy  
        # Lọc lớp 'person' (ID lớp thường là 0 cho người trong COCO dataset)
        if cls == 0:  # 0 là mã cho "person"
            if conf < confidence_threshold:
                continue
            count +=1
            cv2.rectangle(resized_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"Person: {conf:.2f}"  # Tên lớp và độ tin cậy
            cv2.putText(resized_image, label, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cvzone.putTextRect(resized_image, f'COUNT: {count}', (50, 50), scale=1, thickness=1)
    # Hiển thị kết quả
    cv2.imshow("Detected Image", resized_image)
    # Chờ sự kiện nhấn phím và đóng cửa sổ
    key = cv2.waitKey(0)  # Chờ người dùng nhấn phím
    if key == 27:  # 27 là mã ASCII cho phím ESC
        cv2.destroyAllWindows()  # Giải phóng bộ nhớ và đóng tất cả cửa sổ
    cv2.destroyAllWindows() 
def choose_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mkv")])
    if video_path:
        process_video(video_path)

def choose_image():
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if image_path:
        process_image(image_path)
# Tạo giao diện
root = Tk()
# Lấy kích thước màn hình
model_type = StringVar(value="yolo11s")  # Giá trị mặc định là yolo11s
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
# Tính toán vị trí để cửa sổ ở giữa màn hình
window_width = 500
window_height = 400
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
# Đặt vị trí và kích thước cửa sổ
root.geometry(f"{window_width}x{window_height}+{x}+{y}")
root.title(u"Hệ thống Đếm người đi qua khu vực")
# Cài đặt màu nền và font cho cửa sổ chính
root.configure(bg="#f0f0f0")
root.option_add("*Font", "Arial 12")  # Thêm font chữ
# Tạo Frame cho thanh trượt ngưỡng phát hiện và chế độ phát hiện
frame_top = Frame(root, bg="#f0f0f0")
frame_top.pack(pady=15,anchor="w")
frame_bot = Frame(root, bg="#f0f0f0")
frame_bot.pack(pady=15,anchor="w")
# Ngưỡng phát hiện
label_threshold = Label(frame_top, text=u"Ngưỡng phát hiện (%)", bg="#f0f0f0", font=("Arial", 12))
label_threshold.grid(row=0, column=1, padx=10)

slider_threshold = Scale(frame_top, from_=0, to=100, orient=HORIZONTAL, length=200, command=update_threshold)
slider_threshold.set(20)  # Giá trị ban đầu của confidence_threshold là 70%
slider_threshold.grid(row=0, column=2, padx=10)
# Chế độ phát hiện
label_mode = Label(frame_bot, text=u"Chế độ Đếm", bg="#f0f0f0", font=("Arial", 12))
label_mode.grid(row=0, column=2, padx=10)

mode_var = StringVar(value="vertical")  # Giá trị mặc định là "dọc"
radiobutton_vertical = Radiobutton(frame_bot, text="Camera ở Trên", variable=mode_var, value="vertical", command=update_mode, bg="#f0f0f0")
radiobutton_vertical.grid(row=0, column=3, padx=5)

radiobutton_horizontal = Radiobutton(frame_bot, text="Camera ở Dưới", variable=mode_var, value="horizontal", command=update_mode, bg="#f0f0f0")
radiobutton_horizontal.grid(row=0, column=4, padx=5)

# Tạo ComboBox
options = ["yolo11n", "yolo11s", "yolo11m"]
combo = ttk.Combobox(root, values=options)
combo.set(value="yolo11n")  # Giá trị mặc định
combo.pack(pady=20)
def on_select(event):
    global model_path  # Khai báo model là biến toàn cục
    selected_option = combo.get()  # Lấy giá trị được chọn từ ComboBox
    print(f"Selected: {selected_option}")
    # Thay đổi model dựa trên lựa chọn
    if selected_option == "yolo11n":
        model_path = "bestyolo11n.pt"
    elif selected_option == "yolo11s":
        model_path = "yolo11s.pt"
    elif selected_option == "yolo11m":
        model_path = "yolo11m.pt"
combo.bind("<<ComboboxSelected>>", on_select)

# Tạo nút "Chọn ảnh" với thiết kế đẹp hơn
btn_image = Button(root, text=u"Chọn ảnh", command=choose_image, width=20, height=2, bg="#2196F3", fg="white", font=("Arial", 12, "bold"))
btn_image.pack(pady=15)
# Tạo nút "Chọn video" với thiết kế đẹp hơn
btn_video = Button(root, text=u"Chọn video", command=choose_video, width=20, height=2, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
btn_video.pack(pady=15)

# Hiển thị cửa sổ
root.mainloop()

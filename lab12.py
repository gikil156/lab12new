import cv2 as cv
import numpy as np
import sys #tuong tac voi trinh thong dich
import os #tuong tac voi he dieu hanh

# Ép terminal hiển thị tiếng Việt chuẩn UTF-8 (neu khong bi loi co the bo no)
# sys.stdout.reconfigure(encoding='utf-8')

# 1. Đọc ảnh gốc
img_color = cv.imread('v.png')

if img_color is None:
    print("Vui lòng chuẩn bị file ảnh 'v.png'!")
else:
    # 2. Chuyển ảnh về ảnh xám
    img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
    
    # 3. Phân ngưỡng để biến thành ảnh Trắng - Đen nhị phân
    _, img_binary = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY) #co the thay 127 thanh so khac dua vao muc sang cua anh hoac tham chi co the dung otsu hoac thuat toan tu dieu chinh nguong
    
    # 4. TÌM CONTOUR (ĐƯỜNG VIỀN)
    contours, hierarchy = cv.findContours(img_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    print(f"Máy tính đã tìm thấy tổng cộng: {len(contours)} đường viền độc lập!")
    
    # 5. VẼ CONTOUR LÊN ẢNH MÀU GỐC
    cv.drawContours(img_color, contours, -1, (0, 255, 0), 3)
    
    # 6. Hiển thị kết quả (LƯU Ý: Tiêu đề cửa sổ imshow viết KHÔNG DẤU để tránh lỗi hiển thị của OpenCV)
    cv.imshow('Anh nhi phan mang di tim', img_binary)
    cv.imshow('Ket qua ve duong vien mau xanh', img_color)
    
    cv.waitKey(0)
    cv.destroyAllWindows()
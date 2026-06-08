import cv2 as cv
import numpy as np
import sys
import os

# 1. Đọc ảnh gốc
img_color = cv.imread('v.png')

if img_color is None:
    print("Vui lòng chuẩn bị file ảnh 'v.png'!")
else:
    # 2. Chuyển ảnh về ảnh xám
    img_gray = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
    
    # 3. Phân ngưỡng để biến thành ảnh Trắng - Đen nhị phân
    _, img_binary = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY)
    
    # 4. TÌM CONTOUR (ĐƯỜNG VIỀN)
    contours, hierarchy = cv.findContours(img_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    print(f"Máy tính đã tìm thấy tổng cộng: {len(contours)} đường viền độc lập!")
    
    # 5. VẼ CONTOUR LÊN ẢNH MÀU GỐC + ĐÁNH SỐ ID
    for i, cnt in enumerate(contours):
        # Vẽ contour màu xanh
        cv.drawContours(img_color, contours, i, (0, 255, 0), 2)
        
        # Lấy bounding box để đặt số ID
        x, y, w, h = cv.boundingRect(cnt)
        cv.putText(img_color, str(i), (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    
    # 6. Hiển thị kết quả
    cv.imshow('Anh nhi phan mang di tim', img_binary)
    cv.imshow('Ket qua ve duong vien + ID', img_color)
    
    cv.waitKey(0)
    cv.destroyAllWindows()

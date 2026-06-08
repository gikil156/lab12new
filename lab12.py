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
    
    # 👉 In bảng hierarchy
    print("Bảng hierarchy (ID, Next, Previous, First_Child, Parent):")
    for i, h in enumerate(hierarchy[0]):   # hierarchy[0] là mảng 2D (N,4)
        next_id, prev_id, child_id, parent_id = h
        print(f"Contour {i}: Next={next_id}, Previous={prev_id}, First_Child={child_id}, Parent={parent_id}")
    
    # 5. VẼ CONTOUR LÊN ẢNH MÀU GỐC + ĐÁNH SỐ ID
    for i, cnt in enumerate(contours):
        cv.drawContours(img_color, contours, i, (0, 255, 0), 2)
        x, y, w, h = cv.boundingRect(cnt)
        cv.putText(img_color, str(i), (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    
    # 6. Hiển thị kết quả
    cv.imshow('Anh nhi phan mang di tim', img_binary)
    cv.imshow('Ket qua ve duong vien + ID', img_color)
    
    cv.waitKey(0)
    cv.destroyAllWindows()

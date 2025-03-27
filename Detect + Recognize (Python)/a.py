import cv2
import pytesseract
import re
import os
import numpy as np


# 設定 Tesseract 的路徑（如果你在 Windows 上）
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def correct_skew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filter = cv2.medianBlur(gray, 5)
    # 使用 Canny 邊緣檢測
    edges = cv2.Canny(filter, 50, 150, apertureSize=3)
    # cv2.imshow('image.jpg', edges)
    # cv2.waitKey(0)

    # 使用霍夫直線變換
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, None, 50, 10)

    max_length = 0
    longest_line = None

    # 畫出檢測到的直線並找出最長的水平線
    if lines is not None:
        # print("len",len(lines))
        for i in range(0, len(lines)):
            line = lines[i][0]
            x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
            # 計算直線的長度
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            # if x1 != x2 and abs((y2 - y1) / (x2 - x1)) < 1:
            #     cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 更新最長直線
            if x1 != x2 and abs((y2 - y1) / (x2 - x1)) < 1 and length > max_length:
                max_length = length
                longest_line = (x1, y1, x2, y2)

    # 如果找到了最長的直線，畫出來
    if longest_line is not None:
        x1, y1, x2, y2 = longest_line
        horizon = image.copy()
        # cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.imshow('Longest horizontal Line', image)
        # cv2.waitKey(0)

        x1, y1, x2, y2 = longest_line
        if x2 != x1:  # 避免除以零
            print(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
            slope = (y2 - y1) / (x2 - x1)
            theta = np.arctan(slope)
            # 获取图像的尺寸
            (h, w) = image.shape[:2]
            # 计算旋转中心
            center = (w // 2, h // 2)
            # 使用OpenCV的getRotationMatrix2D函数计算旋转矩阵
            rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(theta), 1.0)
            # 进行仿射变换
            rotated_img = cv2.warpAffine(image, rotation_matrix, (w, h))
            # 显示结果
            # cv2.imshow('Rotated Image', rotated_img)
            print(f"最長直線的斜率是: {slope}")
            print(f"最長直線的角度是: {np.degrees(theta)}")
        else:
            rotated_img = image
            print("最長直線是垂直的，斜率無窮大")
    else:
        rotated_img = image
    # cv2.imshow('Rotated Image', rotated_img)
    # cv2.waitKey(0)

    rotated_gray = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
    rotated_filter = cv2.medianBlur(rotated_gray, 5)
    # 使用 Canny 邊緣檢測
    rotated_edges = cv2.Canny(rotated_filter, 50, 150, apertureSize=3)

    rotated_lines = lines = cv2.HoughLinesP(rotated_edges, 1, np.pi / 180, 25, minLineLength=20, maxLineGap= 10)

    vertical_max_length = 0
    vertical_longest_line = None

    if rotated_lines is not None:
        # print("len",len(rotated_lines))
        for i in range(0, len(rotated_lines)):
            line = lines[i][0]
            x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
            # 計算直線的長度
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            # if x1 != x2 and abs((y2 - y1) / (x2 - x1)) >= 1:
                # cv2.line(rotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 更新最長直線
            if x1 != x2 and abs((y2 - y1) / (x2 - x1)) >= 1 and length > vertical_max_length:
                vertical_max_length = length
                vertical_longest_line = (x1, y1, x2, y2)

    if vertical_longest_line is not None:
        x1, y1, x2, y2 = vertical_longest_line
        # cv2.line(rotated_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        # cv2.imshow('Longest vertical Line', rotated_img)
        # cv2.waitKey(0)
    limit_y, limit_x = rotated_img.shape[:2]
    print("limit_x", limit_x, "limit_y", limit_y)

    if vertical_longest_line is not None:
        print("vertical_longest_line", vertical_longest_line)
        x1, y1, x2, y2 = vertical_longest_line
        slope = (y2 - y1) / (x2 - x1)
        print("slope", slope)
        for y in range(limit_y):
            t = int(round(y / slope, 0))
            if(slope > 0):
                for x in range(limit_x):
                    if x + t < limit_x:
                        rotated_img[y, x] = rotated_img[y, x + t]
                    else:
                        rotated_img[y, x] = 0
            else:
                for x in range(limit_x - 1, 0, -1):
                    if x + t >= 0:
                        rotated_img[y, x] = rotated_img[y, x + t]
                    else:
                        rotated_img[y, x] = 0
            # cv2.imshow('Longest Hough Line', rotated_img)
            # cv2.waitKey(0)
        # cv2.line(rotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 顯示結果
    # cv2.imshow('Longest Hough Line', rotated_img)
    # cv2.waitKey(0)

    rotated_gray = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(rotated_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # cv2.imshow('image.jpg', thresh)
    # cv2.waitKey(0)

    card_left = limit_x
    card_right = 0
    for y in range(limit_y):
        for x in range(1, limit_x):
            if thresh[y, x] == 0 and thresh[y, x - 1] == 255:
                if x < card_left:
                    card_left = min(x, card_left)
        for x in range(limit_x - 2, 0, -1):
            if thresh[y, x] == 0 and thresh[y, x + 1] == 255:
                if x > card_right:
                    card_right = max(x, card_right)
    print("card_left", card_left, "card_right", card_right)
    card = rotated_img[0:limit_y, card_left:card_right]
    # cv2.imshow('card.jpg', card)
    # cv2.waitKey(0)
            
    return card

def character_detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filter = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = 255 - cv2.threshold(filter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    vertical_kernel = np.array([[0, 0, 0],
                        [1, 1, 1],
                        [0, 0, 0]], dtype=np.uint8)
    vertical_opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    horizontal_kernel = np.array([[0, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0]], dtype=np.uint8)
    horizontal_opening = cv2.morphologyEx(vertical_opening, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    # cv2.imshow('opening.jpg', horizontal_opening)

    contours, _ = cv2.findContours(horizontal_opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    len = 0
    limit_y, _ = image.shape[:2]
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if h > limit_y * 0.5:
            # cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            len += 1

            # char = gray[y:y + h, x:x + w]
            # cv2.imshow(f'character{idx}.jpg', char)
            # cv2.waitKey(0)
            
            # # 使用 Tesseract 進行 OCR
            # text = pytesseract.image_to_string(char, config='--psm 8').strip()
            # print(f"識別出的字符: {text}")
    print(f"識別出的字符數量: {len}")
    if len >= 4:
        # cv2.imshow('character.jpg', image)
        # cv2.waitKey(0)
        return True
    else:
        return False


def recognize_license_plates_from_video(video_path, filename):
    # 打開影片文件
    cap = cv2.VideoCapture(video_path)

    # 存儲已識別的車牌號碼
    recognized_plates = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 將每一幀轉換為灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray.jpg', gray)

        filter = cv2.medianBlur(gray, 5)
        kernel1 = np.array([[-1, 0, 1],
                            [-1, 0, 1],
                            [-1, 0, 1]])
        kernel2 = np.array([[1, 0, -1], 
                            [1, 0, -1], 
                            [1, 0, -1]])
        s1 = cv2.filter2D(filter, -1, kernel1).astype(np.float32)
        s1 = s1 / 2
        s1 = np.clip(s1, 0, 255)  # 确保值在0-255范围内
        s1 = s1.astype(np.uint8)
        s2 = cv2.filter2D(filter, -1, kernel2).astype(np.float32)
        s2 = s2 / 2
        s2 = np.clip(s2, 0, 255)  # 确保值在0-255范围内
        s2 = s2.astype(np.uint8)
        s = cv2.add(s1, s2)
        
        thresh = cv2.threshold(s, 0, 255, cv2.THRESH_OTSU)[1]
        # cv2.imshow('filter.jpg', thresh)
        # cv2.waitKey(0)

        kernel = np.ones((20, 30), np.uint8)
        ks = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # ks = cv2.morphologyEx(ks, cv2.MORPH_OPEN, kernel)
        # cv2.imshow('ks.jpg', ks)
        # cv2.waitKey(0)


        # 應用邊緣檢測
        # edged = cv2.Canny(filter, 150, 200)
        # cv2.imshow('edged.jpg', edged)
        
        # 找到輪廓
        contours, _ = cv2.findContours(ks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # 獲取輪廓的邊界框
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)

            # 車牌限制條件
            if 2 < aspect_ratio < 6  and character_detect(frame[y:y + h, x:x + w]):
                license_plate = frame[y:y + h, x:x + w]
                corrected_plate = correct_skew(license_plate)
                gray_plate = cv2.cvtColor(corrected_plate, cv2.COLOR_BGR2GRAY)
                # cv2.imshow('license_plate.jpg', corrected_plate)
                # cv2.waitKey(0)

                if character_detect(corrected_plate):

                    # 使用 Tesseract 進行 OCR
                    text = pytesseract.image_to_string(gray_plate, config='--psm 8').strip()
                    filtered_text = re.sub(r'[^A-Z0-9]', '', text)
                    if filtered_text:
                        recognized_plates.append(filtered_text)
                        print("識別出的車牌號碼:", filtered_text)
                        # 寫入識別出的車牌號碼到txt檔案
                        # with open('recognized_plates.txt', 'a') as file:
                        #     file.write(filtered_text + '\n')
                        # 在原圖上畫出車牌邊界框
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, filtered_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # # 顯示處理後的每一幀
        cv2.imshow('Video', frame)
        cv2.waitKey(0)
        # 按 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    if (len(recognized_plates) != 0):
        average = sum(len(s) for s in recognized_plates) / len(recognized_plates)
        print(int(average + 1))
        result = [[0] * 36 for _ in range(int(average + 1))]

        for s in recognized_plates:
            if len(s) > int(average + 1) or len(s) < int(average):
                continue
            for i, a in enumerate(s):
                index = ord('I') - ord('A')
                if 'A' <= a <= 'Z':
                    if a == "I":
                        result[i][27] += 1
                    elif a == "O":
                        result[i][26] += 1
                    else :
                        index = ord(a) - ord('A')
                else:
                    index = ord(a) - ord('0') + 26
                result[i][index] += 1

        output_string_number = []
        for r in result:
            max_num = 0
            max_char = 0
            for char_num in range(36):
                if r[char_num] > max_num:
                    max_num = r[char_num]
                    max_char = char_num
            output_string_number.append(max_char)

        output_string = ''
        for x in output_string_number:
            if x < 26:
                output_string += chr(x + ord('A'))
            else:
                output_string += chr(x - 26 + ord('0'))

        print("識別出的車牌號碼:", output_string)
        with open('output.txt', 'a') as file:
            file.write(filename[:-4] + ' ' + output_string + '\n')
    else:
        print("未識別出車牌號碼")
        with open('output.txt', 'a') as file:
            file.write(filename[:-4]  + '\n')

    # cap.release()
    cv2.destroyAllWindows()

# 測試
for filename in os.listdir('in'):
    recognize_license_plates_from_video('in/' + filename, filename)
import cv2
import numpy as np
import os

def remove_black_borders(image):
    """
    自动裁剪掉图片上下的黑色区域
    """
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 找到非黑色部分
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 找到边界框
    if len(contours) > 0:
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
        cropped = image[y:y+h, x:x+w]
        return cropped
    else:
        return image  # 如果没有找到非黑色部分，返回原图

def denoise_image(image):
    """
    去除图片的噪点
    """
    # 使用高斯模糊去噪
    denoised = cv2.GaussianBlur(image, (5, 5), 0)
    return denoised

def correct_skew(image):
    """
    检测图片倾斜，并旋转到正确的角度
    """
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 边缘检测
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # 霍夫变换找直线
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            angles.append(angle)
        
        # 计算平均角度
        avg_angle = np.mean(angles)
        
        # 旋转图片
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    else:
        return image  # 如果没有检测到直线，返回原图

def process_image(file_path, output_dir):
    """
    处理单张图片
    """
    # 读取图片
    image = cv2.imread(file_path)
    if image is None:
        print(f"无法读取图片：{file_path}")
        return
    
    # 去除黑色边框
    cropped = remove_black_borders(image)
    
    # 去噪
    denoised = denoise_image(cropped)
    
    # 校正倾斜
    corrected = correct_skew(denoised)
    
    # 保存结果
    filename = os.path.basename(file_path)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, corrected)
    print(f"处理完成：{output_path}")

def process_images_in_directory(input_dir, output_dir):
    """
    批量处理文件夹中的所有图片
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            process_image(file_path, output_dir)

if __name__ == "__main__":
    # 输入和输出文件夹
    input_dir = "input_images"  # 替换为你的输入文件夹路径
    output_dir = "output_images"  # 替换为你的输出文件夹路径
    
    # 批量处理图片
    process_images_in_directory(input_dir, output_dir)
    print("所有图片处理完成！")
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import argparse

def clean_document_edges(image_path, output_path=None, bg_color=(255, 255, 255), sensitivity=1.0):
    """
    Clean document image by preserving text regions and making everything else match the background color.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the processed image (if None, will use input_path with '_cleaned' suffix)
        bg_color: Background color to use for non-text regions (default: white)
        sensitivity: Text detection sensitivity (higher values detect more text, default: 1.0)
        
    Returns:
        Path to the processed image
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取图片 {image_path}")
        return None
    
    # Create output path if not provided
    if output_path is None:
        input_path = Path(image_path)
        output_path = str(input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}")
    
    # Store original dimensions
    original_h, original_w = img.shape[:2]
    
    # Resize if image is too large (for faster processing)
    max_dimension = 2000
    scale = 1.0
    if max(original_h, original_w) > max_dimension:
        scale = max_dimension / max(original_h, original_w)
        img = cv2.resize(img, (int(original_w * scale), int(original_h * scale)))
        print(f"图片尺寸过大，调整为: {img.shape[1]}x{img.shape[0]}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Detect text regions using multiple methods for better accuracy
    
    # Method 1: Adaptive thresholding (good for document text with varying background)
    binary_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 11, 10)
    
    # Method 2: Otsu's thresholding (good for clear contrast between text and background)
    _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Method 3: Edge detection to catch text boundaries
    edges = cv2.Canny(gray, 50, 150)
    
    # Combine the methods
    binary = cv2.bitwise_or(binary_adaptive, binary_otsu)
    binary = cv2.bitwise_or(binary, edges)
    
    # Step 2: Use morphological operations to connect nearby text
    # This helps to group characters into words and lines
    kernel_size = max(1, int(3 * sensitivity))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=max(1, int(2 * sensitivity)))
    
    # Step 3: Find contours of text regions
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 4: Create a mask for text regions
    text_mask = np.zeros_like(gray)
    
    # Filter contours to keep only those that are likely text
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h) if h > 0 else 0
        area = w * h
        
        # Text-like characteristics: not too small, not too large, reasonable aspect ratio
        # Adjust thresholds based on sensitivity parameter
        min_area = max(10, int(30 / sensitivity))
        max_aspect = 20 * sensitivity
        min_height = max(2, int(5 / sensitivity))
        
        if (area > min_area and  # Not too small
            aspect_ratio < max_aspect and  # Not extremely wide
            h > min_height):  # Not too short
            
            # Add padding around text regions to ensure we don't cut off parts of characters
            padding = int(min(w, h) * 0.2 * sensitivity)  # Adjustable padding based on sensitivity
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(gray.shape[1], x + w + padding)
            y_end = min(gray.shape[0], y + h + padding)
            
            # Add this region to the text mask
            text_mask[y_start:y_end, x_start:x_end] = 255
    
    # Step 5: Dilate the text mask to ensure we don't miss any text parts
    text_mask = cv2.dilate(text_mask, kernel, iterations=2)
    
    # Step 6: Create a clean image with the specified background color
    clean_img = np.ones_like(img) * np.array(bg_color, dtype=np.uint8)
    
    # Step 7: Copy the original image only where text is detected
    # Invert the mask to get non-text regions
    non_text_mask = cv2.bitwise_not(text_mask)
    
    # 专门处理边缘区域（更强力的清理）
    
    # 1. 先处理顶部黑色区域 - 非常强力的清理
    top_strip_height = int(img.shape[0] * 0.12)  # 顶部区域更大，覆盖所有黑色部分
    top_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    top_mask[:top_strip_height, :] = 255
    
    # 对顶部区域使用非常低的阈值，检测出所有非白区域
    _, top_binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    top_spots = cv2.bitwise_and(top_binary, top_mask)
    
    # 找出顶部区域中的文本保护区
    top_text_protection = cv2.bitwise_and(text_mask, top_mask)
    
    # 扩大文本保护区域，确保不会清除文本
    kernel_text = np.ones((7, 7), np.uint8)
    top_text_protection = cv2.dilate(top_text_protection, kernel_text, iterations=3)
    
    # 从顶部区域中排除文本
    top_spots_final = cv2.bitwise_and(top_spots, cv2.bitwise_not(top_text_protection))
    
    # 2. 处理其他边缘区域
    edge_margin_top = int(img.shape[0] * 0.15)     # 顶部边缘更大
    edge_margin_bottom = int(img.shape[0] * 0.10)  # 底部边缘
    edge_margin_left = int(img.shape[1] * 0.10)    # 左侧边缘
    edge_margin_right = int(img.shape[1] * 0.10)   # 右侧边缘
    
    # 创建边缘掩码
    edge_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    edge_mask[:edge_margin_top, :] = 255           # 顶部边缘
    edge_mask[-edge_margin_bottom:, :] = 255       # 底部边缘
    edge_mask[:, :edge_margin_left] = 255          # 左侧边缘
    edge_mask[:, -edge_margin_right:] = 255        # 右侧边缘
    
    # 在边缘区域使用更严格的阈值检测污点
    _, edge_binary = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)
    edge_spots = cv2.bitwise_and(edge_binary, edge_mask)
    
    # 扩大边缘污点区域以确保完全清除
    kernel_edge = np.ones((7, 7), np.uint8)
    edge_spots_dilated = cv2.dilate(edge_spots, kernel_edge, iterations=3)
    
    # 从边缘污点中排除文本区域
    edge_spots_final = cv2.bitwise_and(edge_spots_dilated, cv2.bitwise_not(text_mask))
    
    # 3. 特殊处理顶部最顶端的黑色区域
    very_top = int(img.shape[0] * 0.05)  # 最顶部区域
    very_top_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    very_top_mask[:very_top, :] = 255
    
    # 将最顶端区域直接加入清理区域，不考虑文本保护
    # 因为这个区域通常是黑色条带
    
    # 4. 合并所有清理区域
    combined_spots = cv2.bitwise_or(top_spots_final, edge_spots_final)
    combined_spots = cv2.bitwise_or(combined_spots, very_top_mask)
    
    # 将边缘污点区域添加到非文本掩码
    non_text_mask = cv2.bitwise_or(non_text_mask, combined_spots)
    
    # Create 3-channel mask for color images
    non_text_mask_3ch = cv2.merge([non_text_mask, non_text_mask, non_text_mask])
    
    # Replace non-text regions with background color
    result = img.copy()
    result[non_text_mask_3ch > 0] = clean_img[non_text_mask_3ch > 0]
    
    # Resize back to original dimensions if needed
    if scale != 1.0:
        result = cv2.resize(result, (original_w, original_h))
    
    # Save the result
    cv2.imwrite(output_path, result)
    print(f"已保存处理后的图片: {output_path}")
    
    return output_path

def batch_process_images(input_folder, output_folder=None, bg_color=(255, 255, 255), sensitivity=1.0):
    """
    Process all images in a folder
    
    Args:
        input_folder: Folder containing images to process
        output_folder: Folder to save processed images (if None, will create a folder named 'output')
        bg_color: Background color to use for non-text regions (default: white)
        sensitivity: Text detection sensitivity (higher values detect more text, default: 1.0)
        
    Returns:
        Number of successfully processed images
    """
    input_path = Path(input_folder)
    
    # Create output folder if not provided
    if output_folder is None:
        output_path = input_path.parent / "output"
    else:
        output_path = Path(output_folder)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Find all image files
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_path.glob(f"*{ext}")))
        image_files.extend(list(input_path.glob(f"*{ext.upper()}")))
    
    if not image_files:
        print(f"在 {input_folder} 中未找到图片文件")
        return 0
    
    print(f"找到 {len(image_files)} 个图片文件，开始处理...")
    
    # Process each image
    successful = 0
    for i, img_path in enumerate(tqdm(image_files, desc="处理图片")):
        output_file = output_path / img_path.name
        try:
            clean_document_edges(str(img_path), str(output_file), bg_color=bg_color, sensitivity=sensitivity)
            successful += 1
        except Exception as e:
            print(f"处理 {img_path} 时出错: {str(e)}")
    
    print(f"成功处理 {successful}/{len(image_files)} 个图片")
    return successful

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="清理文档图片：保留文字区域，清除边缘噪点")
    parser.add_argument("--input", "-i", help="输入图片路径或文件夹路径", required=True)
    parser.add_argument("--output", "-o", help="输出图片路径或文件夹路径")
    parser.add_argument("--batch", "-b", action="store_true", help="批量处理文件夹中的所有图片")
    parser.add_argument("--bg-color", help="背景颜色 (R,G,B 格式, 例如: 255,255,255)", default="255,255,255")
    parser.add_argument("--sensitivity", "-s", type=float, help="文本检测灵敏度 (默认: 1.0, 更高的值会检测更多文本)", default=1.0)
    
    args = parser.parse_args()
    
    # Parse background color
    try:
        bg_color = tuple(map(int, args.bg_color.split(',')))
        if len(bg_color) != 3 or not all(0 <= c <= 255 for c in bg_color):
            raise ValueError("背景颜色必须是三个 0-255 之间的值")
    except Exception as e:
        print(f"背景颜色格式错误: {e}")
        print("使用默认白色背景 (255,255,255)")
        bg_color = (255, 255, 255)
    
    if args.batch:
        batch_process_images(args.input, args.output, bg_color=bg_color, sensitivity=args.sensitivity)
    else:
        if args.output is None:
            input_path = Path(args.input)
            args.output = str(input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}")
        clean_document_edges(args.input, args.output, bg_color=bg_color, sensitivity=args.sensitivity)

import cv2
import numpy as np
import os
import time
import math
import pytesseract
from scipy import ndimage
from pathlib import Path
from tqdm import tqdm

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def clean_document_image(image_path, output_path=None, no_rotate=False, aggressive_clean=False, image_index=None, total_images=None):
    """
    Process a document image:
    1. Remove border artifacts, edge marks, and spots/stains
    2. Correct skewed angles only if needed
    3. Enhance text while preserving the document structure and color
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the processed image (if None, will use input_path with '_cleaned' suffix)
        no_rotate: If True, skips rotation correction
        aggressive_clean: If True, uses more aggressive cleaning settings
        image_index: Current image index (for batch processing)
        total_images: Total number of images (for batch processing)
        
    Returns:
        Path to the processed image and processing time in seconds
    """
    # Start timing
    start_time = time.time()
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取图片 {image_path}")
        return None, 0
        
    # Check if image is too large and resize if necessary
    max_dimension = 4000  # Maximum dimension to process efficiently
    h, w = img.shape[:2]
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        print(f"图片尺寸过大，调整为: {img.shape[1]}x{img.shape[0]}")
    
    # Display progress information if in batch mode
    if image_index is not None and total_images is not None:
        print(f"开始处理图片 [{image_index}/{total_images}]：{image_path}")
    else:
        print(f"开始处理图片：{image_path}")
    
    # Store original color image for later use
    original_img = img.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Step 1: Remove border artifacts, edge marks, and colored areas
    
    # Calculate border removal margins (use percentage of image dimension)
    # Further reduced border margins to prevent removing content text
    top_margin = int(h * 0.01)  # Reduced from 2% to 1% to preserve top text line
    side_margin = int(w * 0.01)  # Keep at 1%
    bottom_margin = int(h * 0.02)  # Keep at 2%
    
    print(f"设置边缘清理区域 - 上: {top_margin}px, 侧边: {side_margin}px, 底部: {bottom_margin}px")
    
    # Create a mask that excludes the borders
    mask = np.ones_like(gray, dtype=np.uint8) * 255
    
    # Mark borders as black (0) in the mask
    mask[0:top_margin, :] = 0  # Top border
    mask[:, 0:side_margin] = 0  # Left border
    mask[:, w-side_margin:w] = 0  # Right border
    mask[h-bottom_margin:h, :] = 0  # Bottom border
    
    # Detect and remove red areas (common in document stamps, marks, etc.)
    print("检测并清除红色区域")
    
    # Split the image into its color channels
    b, g, r = cv2.split(img)
    
    # Create a mask for red areas (where red channel is significantly higher than others)
    red_mask = np.zeros_like(gray, dtype=np.uint8)
    
    # Red areas typically have high red channel and low blue/green channels
    red_dominant = ((r > 150) & (r > g * 1.5) & (r > b * 1.5)).astype(np.uint8) * 255
    
    # Dilate the red mask to ensure complete coverage of red areas
    red_kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.dilate(red_dominant, red_kernel, iterations=1)
    
    # Add red areas to the border mask (marking them for removal)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(red_mask))
    
    # Count red pixels for reporting
    red_pixel_count = np.sum(red_mask > 0)
    if red_pixel_count > 0:
        print(f"检测到 {red_pixel_count} 个红色像素点需要清除")
    
    # Also look for dark horizontal lines at the top which are common artifacts
    # Scan only the very top 5% of the image for horizontal lines to avoid removing text content
    top_scan_area = gray[0:int(h*0.05), :]
    horizontal_kernel = np.ones((1, int(w*0.3)), np.uint8)  # Kernel for detecting horizontal lines
    top_horizontal = cv2.morphologyEx(top_scan_area, cv2.MORPH_OPEN, horizontal_kernel)
    _, top_horiz_thresh = cv2.threshold(top_horizontal, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Identify black regions that are horizontal lines in the top area
    if np.sum(top_horiz_thresh) > 0:
        # Find where these lines are and extend the top border mask if needed
        line_rows = np.where(np.sum(top_horiz_thresh, axis=1) > w*0.7)[0]  # Increased threshold to 70% to only detect clear horizontal lines
        if len(line_rows) > 0:
            extended_top_margin = np.max(line_rows) + int(h*0.01)  # Reduced to 1% to preserve text
            mask[0:min(extended_top_margin, int(h*0.05)), :] = 0  # Reduced scan area to 5%
            print(f"检测到水平线，扩展上边缘清理至: {extended_top_margin}px")
    
    # Use a stronger sharpening to enhance text clarity while keeping processing fast
    # This kernel provides better text definition without excessive processing
    kernel = np.array([[0, -1, 0], [-1, 5.5, -1], [0, -1, 0]], np.float32)  # Enhanced sharpening kernel
    sharpened = cv2.filter2D(gray, -1, kernel)
    
    # Apply direct binary thresholding to preserve text edges
    # Use Otsu's method to find the optimal threshold value automatically
    _, otsu_thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply a very conservative adaptive threshold to maintain text quality
    block_size = 11  # Very small block size to preserve fine details
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, block_size, 5  # Lower constant for better text preservation
    )
    
    print("应用自适应阈值处理以增强文字内容")
    
    # Combine the thresholding methods
    combined_thresh = cv2.bitwise_or(otsu_thresh, adaptive_thresh)
    
    # Create a text mask to preserve original text quality
    # This identifies definite text areas that should be preserved
    text_mask = cv2.bitwise_and(otsu_thresh, adaptive_thresh)
    
    # Apply the border mask - set border areas to white (255)
    with_borders_removed = cv2.bitwise_or(combined_thresh, cv2.bitwise_not(mask))
    
    # Process the resulting image
    processed = with_borders_removed.copy()
    
    # Step 2: Skew detection and correction (only if not disabled)
    if not no_rotate:
        # Function to check if image needs rotation (0, 90, 180, or 270 degrees) using OCR
        def detect_orientation(img):
            # First try OCR-based orientation detection
            try:
                # Create copies of the image rotated in different orientations
                img_original = img.copy()
                img_90 = cv2.rotate(img.copy(), cv2.ROTATE_90_COUNTERCLOCKWISE)
                img_180 = cv2.rotate(img.copy(), cv2.ROTATE_180)
                img_270 = cv2.rotate(img.copy(), cv2.ROTATE_90_CLOCKWISE)
                
                # Function to evaluate text detection confidence for each orientation
                def get_ocr_confidence(image):
                    # Convert to grayscale if not already
                    if len(image.shape) == 3:
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = image
                        
                    # Apply threshold to make text more visible
                    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # Get OCR data with confidence scores
                    ocr_data = pytesseract.image_to_data(thresh, lang='chi_sim+eng', output_type=pytesseract.Output.DICT)
                    
                    # Calculate average confidence and text count
                    confidences = [float(conf) for conf in ocr_data['conf'] if conf != '-1']
                    text_count = len([word for word in ocr_data['text'] if len(word.strip()) > 1])
                    
                    # If no text is detected, return 0
                    if not confidences or text_count == 0:
                        return 0, 0
                        
                    avg_conf = sum(confidences) / len(confidences)
                    return avg_conf, text_count
                
                print("使用OCR检测文本方向...")
                
                # Get confidence scores for each orientation
                conf_0, count_0 = get_ocr_confidence(img_original)
                conf_90, count_90 = get_ocr_confidence(img_90)
                conf_180, count_180 = get_ocr_confidence(img_180)
                conf_270, count_270 = get_ocr_confidence(img_270)
                
                # Weight by both confidence and text count
                score_0 = conf_0 * count_0 if count_0 > 0 else 0
                score_90 = conf_90 * count_90 if count_90 > 0 else 0
                score_180 = conf_180 * count_180 if count_180 > 0 else 0
                score_270 = conf_270 * count_270 if count_270 > 0 else 0
                
                print(f"OCR方向检测结果 - 0°: {score_0:.1f}, 90°: {score_90:.1f}, 180°: {score_180:.1f}, 270°: {score_270:.1f}")
                
                # Find the orientation with the highest score
                scores = [score_0, score_90, score_180, score_270]
                max_score = max(scores)
                
                # Only rotate if the best score is significantly better than original
                if max_score > 0:
                    best_orientation = scores.index(max_score) * 90
                    
                    # If original orientation score is close to the best, keep original
                    if best_orientation != 0 and score_0 > 0 and max_score / score_0 < 1.3:
                        print("原始方向与最佳方向相近，保持原始方向")
                        return 0
                        
                    print(f"OCR检测到最佳文本方向为: {best_orientation}°")
                    return best_orientation
                    
            except Exception as e:
                print(f"OCR方向检测失败: {str(e)}，将使用备用方法")
            
            # Fallback to line-based orientation detection if OCR fails
            print("使用线条分析进行备用方向检测...")
            
            # Look for horizontal and vertical lines to confirm document orientation
            edges = cv2.Canny(img, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=5)
            
            if lines is None or len(lines) < 5:
                return 0  # Not enough lines to determine orientation reliably
                
            # For 180-degree detection, we'll check if most horizontal lines are in the bottom half
            h, w = img.shape[:2]
            top_half_lines = 0
            bottom_half_lines = 0
            left_half_lines = 0
            right_half_lines = 0
            
            horizontal_angles = []
            vertical_angles = []
            upside_down_score = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Count lines in different regions of the image
                if y1 < h/2 and y2 < h/2:
                    top_half_lines += 1
                if y1 > h/2 and y2 > h/2:
                    bottom_half_lines += 1
                if x1 < w/2 and x2 < w/2:
                    left_half_lines += 1
                if x1 > w/2 and x2 > w/2:
                    right_half_lines += 1
                    
                # Calculate angle
                if x2 - x1 == 0:  # Avoid division by zero
                    angle = 90.0
                else:
                    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                
                # Categorize lines as horizontal or vertical
                if abs(angle) < 20 or abs(angle - 180) < 20 or abs(angle + 180) < 20:
                    horizontal_angles.append(angle)
                    # Check for upside-down text (most horizontal lines have negative slope)
                    if abs(angle - 180) < 20 or abs(angle + 180) < 20:
                        upside_down_score += 1
                elif abs(angle - 90) < 20 or abs(angle + 90) < 20:
                    vertical_angles.append(angle)
            
            # Detect 180-degree rotation (upside down)
            if upside_down_score > len(horizontal_angles) * 0.5 and bottom_half_lines > top_half_lines * 1.2:
                print("线条分析检测到图片可能是上下颠倒（180度旋转）")
                return 180
                
            # Detect 90-degree rotation
            if len(vertical_angles) > len(horizontal_angles) * 1.5:
                # Check if it's 90 or 270 degrees by comparing left/right content distribution
                if right_half_lines > left_half_lines * 1.2:
                    print("线条分析检测到图片需要旋转90度")
                    return 90
                else:
                    print("线条分析检测到图片需要旋转270度")
                    return 270
            
            # No significant rotation detected
            return 0
        
        # Detect image orientation and rotate if needed
        rotation_angle = detect_orientation(processed)
        if rotation_angle == 0:
            print("图片已经正直，跳过旋转校正")
            rotated = processed
        elif rotation_angle == 90:
            print("应用90度旋转校正")
            rotated = cv2.rotate(processed, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # Also rotate the original image to match dimensions
            original_img = cv2.rotate(original_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # Update dimensions after rotation
            h, w = rotated.shape
        elif rotation_angle == 180:
            print("应用180度旋转校正")
            rotated = cv2.rotate(processed, cv2.ROTATE_180)
            original_img = cv2.rotate(original_img, cv2.ROTATE_180)
        elif rotation_angle == 270:
            print("应用270度旋转校正")
            rotated = cv2.rotate(processed, cv2.ROTATE_90_CLOCKWISE)
            original_img = cv2.rotate(original_img, cv2.ROTATE_90_CLOCKWISE)
            # Update dimensions after rotation
            h, w = rotated.shape
        else:
            # More accurate skew detection for cases where it's actually needed
            edges = cv2.Canny(processed, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
            
            angle = 0
            if lines is not None:
                # Calculate the average angle of horizontal lines
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    # Only consider nearly horizontal lines
                    if abs(theta - 0) < 0.1 or abs(theta - np.pi) < 0.1:
                        angles.append(theta)
                
                if angles:
                    # Calculate the skew as the average angle
                    mean_angle = np.mean(angles)
                    angle_degrees = np.degrees(mean_angle) - 90  # Adjust by 90 degrees
                    
                    # Normalize the angle
                    if angle_degrees > 45:
                        angle_degrees -= 90
                    elif angle_degrees < -45:
                        angle_degrees += 90
                        
                    angle = angle_degrees
            
            # Only rotate if the angle is significant but not extreme
            if 0.5 < abs(angle) < 20:
                print(f"旋转图片 {angle:.2f} 度")
                (h, w) = processed.shape[:2]
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(processed, rotation_matrix, (w, h),
                                        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            else:
                print(f"角度 ({angle:.2f} 度) 太小或太大，跳过旋转")
                rotated = processed
    else:
        # Skip rotation if disabled
        print("根据设置，跳过旋转校正")
        rotated = processed
    
    # Step 3: Modified stain and spot removal to better preserve all text content
    
    # Use connected component analysis to remove isolated spots and artifacts
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cv2.bitwise_not(rotated), connectivity=8)
    
    # Get the largest component (likely the main text body or background)
    largest_label = 1  # Start from 1 as 0 is typically background
    largest_area = 0
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_label = i
    
    print(f"检测到 {num_labels-1} 个连通区域，最大区域面积: {largest_area}px")
    
    # Create a mask for small, isolated components (likely spots/stains/artifacts)
    # Adjusted thresholds to better clean spots while preserving text
    min_size = 20 if aggressive_clean else 10  # Increased from 10/5 to better remove spots
    aspect_ratio_threshold = 4 if aggressive_clean else 3  # Increased to better distinguish between spots and text
    
    spots_mask = np.zeros(rotated.shape, dtype=np.uint8)
    
    # Special handling for handwritten content - we want to preserve it
    # Assume handwritten text has different characteristics than printed text
    
    print("开始识别和保留文本内容...")
    
    # Skip the background (label 0) and the largest component
    removed_count = 0
    for i in range(1, num_labels):
        # Skip the largest component
        if i == largest_label:
            continue
            
        # Get component stats
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        
        # Skip components that are likely text (based on size and aspect ratio)
        aspect_ratio = max(width, height) / (min(width, height) if min(width, height) > 0 else 1)
        
        # Calculate density of the component (percentage of black pixels)
        component_mask = (labels == i).astype(np.uint8)
        density = np.sum(component_mask) / (width * height) if width * height > 0 else 0
        
        # Check if component is at the edge (these could be border artifacts)
        at_border = (x <= side_margin or y <= top_margin or 
                     x + width >= w - side_margin or 
                     y + height >= h - bottom_margin)
        
        # Special consideration for bottom left region where handwritten text might be
        in_bottom_left = (y > h * 0.7) and (x < w * 0.3)
        
        # Size threshold adjusted to be much smaller for potential text regions
        size_threshold = min_size
        if at_border:
            size_threshold = min_size * 3  # Reduced from 5x to 3x
        if in_bottom_left:
            size_threshold = min_size / 2  # Even smaller threshold for handwritten text area
            
        # Simplified and faster spot detection
        is_spot = False
        
        # Small isolated components are likely spots
        if area < size_threshold and aspect_ratio < 1.5:
            is_spot = True
        # Components at border that aren't too large
        elif at_border and area < h * w * 0.002:
            is_spot = True
        # Special case for bottom area spots (including red-boxed areas)
        elif y > h * 0.7 and (area < 300 or (width > height * 2 and area < 500)):
            is_spot = True
            
        if is_spot:
            # Get the coordinates of this component
            component_mask = (labels == i).astype(np.uint8) * 255
            spots_mask = cv2.bitwise_or(spots_mask, component_mask)
            removed_count += 1
    
    print(f"移除了 {removed_count} 个小型斑点或污渍")
    
    # Remove the identified spots by setting them to white (255)
    cleaned = cv2.bitwise_or(rotated, spots_mask)
    
    # Final cleanup for any remaining thin lines or small spots - more conservative
    if aggressive_clean:
        # Use morphological opening but with smaller kernel to preserve text
        kernel = np.ones((2, 2), np.uint8)  # Using a 2x2 kernel for more effective cleaning
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        print("应用积极清理模式，进行额外处理")
    
    # Simplified and faster bottom area cleanup
    bottom_mask = np.zeros_like(cleaned)
    bottom_mask[int(h*0.7):, :] = 255  # Only process the bottom 30% of the image
    
    # Apply a more aggressive cleaning to the bottom area
    bottom_area = cv2.bitwise_and(cleaned, bottom_mask)
    
    # Use a single morphological operation for speed
    bottom_kernel = np.ones((3, 3), np.uint8)
    bottom_area = cv2.morphologyEx(bottom_area, cv2.MORPH_OPEN, bottom_kernel)
    
    # Apply threshold to clean spots
    _, bottom_thresh = cv2.threshold(bottom_area, 180, 255, cv2.THRESH_BINARY)
    
    # Merge the bottom cleaned area with the rest of the image
    cleaned = cv2.bitwise_and(cleaned, cv2.bitwise_not(bottom_mask))
    cleaned = cv2.bitwise_or(cleaned, bottom_thresh)
    
    # Final step: Create a mask for the cleaned content
    # This mask will identify areas that should be white (background) vs content
    # Black pixels (0) in cleaned = content, White pixels (255) = background
    content_mask = (cleaned < 128).astype(np.uint8)
    
    # Create a color result image starting with all white
    result_img = np.ones_like(original_img) * 255
    
    # Ensure content_mask has the same dimensions as the processed image
    if content_mask.shape != (original_img.shape[0], original_img.shape[1]):
        content_mask = cv2.resize(content_mask, (original_img.shape[1], original_img.shape[0]))
    
    # Additional check for any remaining red areas in the original image
    # This ensures we catch any red marks that might have been missed earlier
    b, g, r = cv2.split(original_img)
    remaining_red = ((r > 150) & (r > g * 1.5) & (r > b * 1.5)).astype(np.uint8)
    
    # Remove remaining red areas from the content mask
    content_mask = cv2.bitwise_and(content_mask, cv2.bitwise_not(remaining_red))
    
    # Faster vectorized operation for color processing
    # Create a darkening factor mask (0.65 = 35% darker) for better text clarity
    darkening = np.ones_like(content_mask, dtype=np.float32) * 0.65
    
    # Apply the mask to all channels at once (much faster)
    for c in range(3):
        channel = original_img[:,:,c].astype(np.float32)
        # Where content_mask is 1, use darkened original, else use white
        result_img[:,:,c] = np.where(content_mask == 1, 
                                    (channel * darkening).astype(np.uint8), 
                                    255)
    
    # Convert back to grayscale for any remaining processing
    cleaned = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
    
    # Determine output path
    if output_path is None:
        path_obj = Path(image_path)
        output_path = str(path_obj.with_stem(path_obj.stem + '_cleaned'))
    
    # Calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Save the processed color image
    cv2.imwrite(output_path, result_img)
    
    # Print completion message with timing information
    if image_index is not None and total_images is not None:
        print(f"图片 [{image_index}/{total_images}] 处理完成，耗时: {processing_time:.2f}秒，已保存到: {output_path}")
    else:
        print(f"处理完成，耗时: {processing_time:.2f}秒，已保存到: {output_path}")
    
    return output_path, processing_time

def batch_process_images(input_folder, output_folder=None, no_rotate=False, aggressive_clean=False):
    """
    Process all images in a folder
    
    Args:
        input_folder: Folder containing images to process
        output_folder: Folder to save processed images (if None, will create a subfolder named 'cleaned')
        no_rotate: If True, skips rotation correction
        aggressive_clean: If True, uses more aggressive cleaning settings
        
    Returns:
        Number of successfully processed images
    """
    # Start timing for the entire batch
    batch_start_time = time.time()
    
    # Create output folder if not specified
    if output_folder is None:
        output_folder = os.path.join(input_folder, 'cleaned')
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(Path(input_folder).glob(f'*{ext}')))
        image_files.extend(list(Path(input_folder).glob(f'*{ext.upper()}')))
    
    if not image_files:
        print(f"在 {input_folder} 中未找到图片文件")
        return 0
        
    total_images = len(image_files)
    print(f"找到 {total_images} 个待处理图片")
    success_count = 0
    total_processing_time = 0
    
    # Process each image with progress bar
    for i, img_path in enumerate(tqdm(image_files, desc="正在处理图片")):
        output_path = os.path.join(output_folder, img_path.name)
        try:
            # Pass image index and total count for progress reporting
            result, processing_time = clean_document_image(
                str(img_path), 
                output_path, 
                no_rotate=no_rotate, 
                aggressive_clean=aggressive_clean,
                image_index=i+1,
                total_images=total_images
            )
            if result:
                success_count += 1
                total_processing_time += processing_time
        except Exception as e:
            print(f"处理 {img_path.name} 时出错: {str(e)}")
    
    # Calculate total batch processing time
    batch_end_time = time.time()
    batch_total_time = batch_end_time - batch_start_time
    
    # Print summary with timing information
    print(f"批处理完成！")
    print(f"成功处理: {success_count}/{total_images} 个图片")
    print(f"总处理时间: {batch_total_time:.2f}秒")
    if success_count > 0:
        print(f"平均每张处理时间: {total_processing_time/success_count:.2f}秒")
    print(f"清理后的图片保存在: {output_folder}")
    return success_count

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="清理文档图片：移除斑点并修正倾斜")
    parser.add_argument("input", help="输入图片路径或包含图片的文件夹")
    parser.add_argument("--output", "-o", help="输出路径或文件夹（可选）")
    parser.add_argument("--batch", "-b", action="store_true", help="处理输入文件夹中的所有图片")
    parser.add_argument("--no-rotate", action="store_true", help="禁用旋转/倾斜校正")
    parser.add_argument("--aggressive-clean", action="store_true", help="使用更积极的污点和边缘清除")
    
    args = parser.parse_args()
    
    try:
        # Start timing for overall execution
        overall_start_time = time.time()
        
        if args.batch:
            print(f"开始批量处理图片文件夹：{args.input}")
            batch_process_images(
                args.input, 
                args.output, 
                no_rotate=args.no_rotate,
                aggressive_clean=args.aggressive_clean
            )
        else:
            print(f"开始处理单个图片：{args.input}")
            result, processing_time = clean_document_image(
                args.input, 
                args.output, 
                no_rotate=args.no_rotate,
                aggressive_clean=args.aggressive_clean
            )
            if result:
                print(f"图片处理成功。总耗时: {processing_time:.2f}秒。输出保存至: {result}")
            else:
                print("处理失败。")
                
        # Calculate and display overall execution time
        overall_end_time = time.time()
        overall_time = overall_end_time - overall_start_time
        print(f"程序总运行时间: {overall_time:.2f}秒")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
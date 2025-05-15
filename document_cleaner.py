import cv2
import numpy as np
import os
import time
import math
from scipy import ndimage
from pathlib import Path
from tqdm import tqdm

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
    
    # Note: Red area detection and removal has been disabled as requested
    # We'll only focus on cleaning borders and enhancing text clarity
    
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
    
    # Preserve original text quality while only removing noise
    # Use a gentler sharpening to maintain text clarity without distortion
    kernel = np.array([[0, -0.3, 0], [-0.3, 2.6, -0.3], [0, -0.3, 0]], np.float32)  # Gentler sharpening kernel
    sharpened = cv2.filter2D(gray, -1, kernel)
    
    # Apply Otsu's thresholding to identify definite text areas
    _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply adaptive thresholding with optimized parameters for Chinese text
    block_size = 25  # Much larger block size to preserve connected Chinese characters
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, block_size, 8  # Adjusted constant for better text preservation
    )
    
    print("应用自适应阈值处理以增强文字内容")
    
    # Apply a more text-preserving approach for Chinese characters
    # Bilateral filter with carefully tuned parameters for text edge preservation
    bilateral = cv2.bilateralFilter(gray, 5, 50, 50)  # Smaller window, more conservative filtering
    
    # Use a more conservative threshold to preserve thin strokes in Chinese characters
    _, bilateral_thresh = cv2.threshold(bilateral, 180, 255, cv2.THRESH_BINARY)
    
    # Combine the thresholding methods with priority to preserve text
    # For Chinese text, we need to be more careful about preserving thin strokes
    combined_thresh = cv2.bitwise_or(bilateral_thresh, adaptive_thresh)
    
    # Create a text mask to preserve original text quality
    # This identifies definite text areas that should be preserved
    text_mask = cv2.bitwise_and(otsu_thresh, adaptive_thresh)
    
    # Apply the border mask - set border areas to white (255)
    with_borders_removed = cv2.bitwise_or(combined_thresh, cv2.bitwise_not(mask))
    
    # Process the resulting image
    processed = with_borders_removed.copy()
    
    # Step 2: Skew detection and correction (only if not disabled)
    if not no_rotate:
        # Enhanced function to check if image needs rotation (0, 90, 180, or 270 degrees)
        def detect_orientation(img):
            print("使用增强的线条和内容分析进行方向检测...")
            
            # First, analyze all four possible orientations
            orientations = [
                ("0度", img.copy()),
                ("90度", cv2.rotate(img.copy(), cv2.ROTATE_90_COUNTERCLOCKWISE)),
                ("180度", cv2.rotate(img.copy(), cv2.ROTATE_180)),
                ("270度", cv2.rotate(img.copy(), cv2.ROTATE_90_CLOCKWISE))
            ]
            
            # Function to analyze content distribution and line patterns
            def analyze_orientation(image, name):
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                
                # Apply adaptive threshold to enhance text and lines
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 11, 2)
                
                # Find edges for line detection
                edges = cv2.Canny(binary, 50, 150, apertureSize=3)
                
                # Detect lines
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=80, maxLineGap=10)
                
                if lines is None or len(lines) < 5:
                    return {"score": 0, "reason": "未检测到足够的线条"}
                
                h, w = gray.shape[:2]
                
                # Content distribution analysis
                # Divide the image into a 4x4 grid and analyze black pixel distribution
                grid_h, grid_w = h // 4, w // 4
                grid_density = []
                
                for i in range(4):
                    for j in range(4):
                        roi = binary[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                        black_pixels = np.sum(roi == 0)
                        grid_density.append(black_pixels)
                
                # Calculate top vs bottom content ratio (for 180 degree detection)
                top_content = sum(grid_density[:8])  # Top half
                bottom_content = sum(grid_density[8:])  # Bottom half
                
                # Calculate left vs right content ratio (for 90/270 degree detection)
                left_content = sum([grid_density[i] for i in range(16) if i % 4 < 2])
                right_content = sum([grid_density[i] for i in range(16) if i % 4 >= 2])
                
                # Analyze lines
                horizontal_lines = 0
                vertical_lines = 0
                diagonal_lines = 0
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Calculate angle
                    if x2 - x1 == 0:  # Vertical line
                        angle = 90.0
                    else:
                        angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                    
                    # Categorize line direction
                    if angle < 20 or angle > 160:
                        horizontal_lines += 1
                    elif 70 < angle < 110:
                        vertical_lines += 1
                    else:
                        diagonal_lines += 1
                
                # Text-like pattern detection (more horizontal than vertical lines typically indicates text)
                text_pattern_score = horizontal_lines - vertical_lines * 0.5
                
                # Calculate a score based on multiple factors
                # 1. Text patterns (horizontal lines are good)
                # 2. Content distribution (more on top is typical for documents)
                # 3. Line alignment (regular spacing of horizontal lines)
                
                # For normal orientation, we expect:
                # - More content in top half than bottom half
                # - More horizontal than vertical lines
                # - Content well-distributed across the page width
                
                # Basic score from line patterns
                score = text_pattern_score
                
                # Adjust score based on content distribution
                if top_content > bottom_content * 1.2:
                    # Typical document has more content at top
                    score += 20
                    reason = "内容主要分布在上半部分，符合正常文档格式"
                elif bottom_content > top_content * 1.2:
                    # Upside down document has more content at bottom
                    score -= 20
                    reason = "内容主要分布在下半部分，可能是上下颠倒"
                else:
                    reason = "内容上下分布均衡"
                
                # Adjust for left-right balance (for 90/270 detection)
                lr_ratio = max(left_content, right_content) / (min(left_content, right_content) if min(left_content, right_content) > 0 else 1)
                if lr_ratio > 2:
                    # Very unbalanced left-right suggests incorrect orientation
                    score -= 10
                
                # Bonus for having a good ratio of horizontal to vertical lines (text-like)
                if horizontal_lines > vertical_lines * 1.5:
                    score += 15
                    reason += ", 水平线条占主导，符合文本特征"
                
                return {
                    "score": score,
                    "h_lines": horizontal_lines,
                    "v_lines": vertical_lines,
                    "top_content": top_content,
                    "bottom_content": bottom_content,
                    "left_content": left_content,
                    "right_content": right_content,
                    "reason": reason
                }
            
            # Analyze each orientation
            results = []
            for name, img_orient in orientations:
                result = analyze_orientation(img_orient, name)
                result["orientation"] = name
                result["angle"] = orientations.index((name, img_orient)) * 90
                results.append(result)
                print(f"方向 {name} 分数: {result['score']:.1f} - {result['reason']}")
            
            # Find the orientation with the highest score
            best_result = max(results, key=lambda x: x["score"])
            
            # Only rotate if the best score is significantly better than original (0 degrees)
            original_score = results[0]["score"]
            best_score = best_result["score"]
            
            # If the best orientation is not 0 degrees and is significantly better
            if best_result["angle"] != 0 and best_score > original_score + 15:
                print(f"检测到最佳方向为 {best_result['orientation']}，分数为 {best_score:.1f}")
                return best_result["angle"]
            else:
                print("保持原始方向，没有检测到显著需要旋转的情况")
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
            
        # Enhanced spot detection with better stain handling
        is_spot = False
        
        # Small isolated components are likely spots
        if area < size_threshold and aspect_ratio < 1.8:
            is_spot = True
        # Components at border that aren't too large
        elif at_border and area < h * w * 0.003:  # Increased threshold to catch more border artifacts
            is_spot = True
        # Special case for bottom area spots (including red-boxed areas)
        elif y > h * 0.7 and (area < 400 or (width > height * 2 and area < 600)):
            is_spot = True
        # Additional check for stains - stains often have irregular shapes and low density
        elif area < 800 and density < 0.4 and aspect_ratio < 2.5:
            is_spot = True
            
        if is_spot:
            # Get the coordinates of this component
            component_mask = (labels == i).astype(np.uint8) * 255
            spots_mask = cv2.bitwise_or(spots_mask, component_mask)
            removed_count += 1
    
    print(f"移除了 {removed_count} 个小型斑点或污渍")
    
    # Remove the identified spots by setting them to white (255)
    cleaned = cv2.bitwise_or(rotated, spots_mask)
    
    # Final cleanup for any remaining thin lines or small spots - more effective but text-preserving
    if aggressive_clean:
        # First identify definite text areas to protect them
        # Use morphological operations only on non-text areas
        
        # Create a dilated text mask to protect text and surrounding areas
        text_protect_mask = cv2.dilate(text_mask, np.ones((5, 5), np.uint8), iterations=1)
        
        # Create a copy for cleaning non-text areas
        cleaned_nontext = cleaned.copy()
        
        # Apply stronger cleaning only to non-text areas
        nontext_areas = cv2.bitwise_not(text_protect_mask)
        cleaned_nontext = cv2.bitwise_and(cleaned_nontext, nontext_areas)
        cleaned_nontext = cv2.medianBlur(cleaned_nontext, 5)  # Stronger blur for non-text
        
        # Merge back with original text areas
        text_areas = cv2.bitwise_and(cleaned, text_protect_mask)
        cleaned = cv2.bitwise_or(text_areas, cleaned_nontext)
        
        print("应用积极清理模式，进行额外处理")
    else:
        # Even in non-aggressive mode, selectively clean non-text areas
        text_protect_mask = cv2.dilate(text_mask, np.ones((3, 3), np.uint8), iterations=1)
        nontext_areas = cv2.bitwise_not(text_protect_mask)
        
        # Apply light cleaning only to non-text areas
        cleaned_nontext = cv2.bitwise_and(cleaned, nontext_areas)
        cleaned_nontext = cv2.medianBlur(cleaned_nontext, 3)
        
        # Merge back with original text areas
        text_areas = cv2.bitwise_and(cleaned, text_protect_mask)
        cleaned = cv2.bitwise_or(text_areas, cleaned_nontext)
    
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
    
    # Note: Red area detection and removal has been disabled as requested
    
    # Faster vectorized operation for color processing
    # No separate darkening mask needed - we'll use a much lighter approach directly in the channel processing
    
    # Direct text extraction approach - no blurring or processing of text
    # Start with a clean white background
    result_img = np.ones_like(original_img) * 255
    
    # Create a mask that captures all text with minimal processing
    # Use multiple thresholding methods to ensure we don't miss any text
    
    # Method 1: Direct binary threshold - good for clear, dark text
    _, binary_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    
    # Method 2: Adaptive threshold - better for varying text darkness
    adaptive_mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 5
    )
    
    # Method 3: Canny edge detection - good for text outlines
    edges = cv2.Canny(gray, 100, 200)
    dilated_edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
    
    # Combine all methods to get comprehensive text mask
    combined_text_mask = cv2.bitwise_or(binary_mask, adaptive_mask)
    combined_text_mask = cv2.bitwise_or(combined_text_mask, dilated_edges)
    
    # Clean up the mask to remove noise while preserving text
    # First close small gaps within characters
    close_kernel = np.ones((2, 2), np.uint8)
    text_mask = cv2.morphologyEx(combined_text_mask, cv2.MORPH_CLOSE, close_kernel)
    
    # Use connected component analysis for more precise text identification
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_mask, connectivity=8)
    
    # Create a refined text mask that only includes components likely to be text
    refined_mask = np.zeros_like(text_mask)
    
    print(f"检测到 {num_labels-1} 个潜在文本区域")
    text_count = 0
    
    # Skip background (label 0)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        
        # Calculate aspect ratio
        aspect_ratio = max(width, height) / (min(width, height) if min(width, height) > 0 else 1)
        
        # Calculate density (percentage of black pixels)
        component_mask = (labels == i).astype(np.uint8)
        density = np.sum(component_mask) / (width * height) if width * height > 0 else 0
        
        # More precise text detection criteria
        is_text = False
        
        # Chinese characters and punctuation
        if (0.3 < aspect_ratio < 3.0) and area > 15 and area < 3000 and density > 0.2:
            is_text = True
        # Long horizontal text (like underlines or dashes)
        elif width > height * 3 and width < w * 0.5 and area > 20 and area < 1000:
            is_text = True
        # Tall vertical text (like exclamation marks or vertical lines in characters)
        elif height > width * 3 and height < h * 0.5 and area > 20 and area < 1000:
            is_text = True
            
        # Exclude very small components that are likely noise
        if area < 10:
            is_text = False
        # Exclude components at the very edge that are likely artifacts
        if x <= 2 or y <= 2 or x + width >= w - 2 or y + height >= h - 2:
            is_text = False
            
        if is_text:
            # Add this component to the refined mask
            component_mask = (labels == i).astype(np.uint8) * 255
            refined_mask = cv2.bitwise_or(refined_mask, component_mask)
            text_count += 1
    
    print(f"保留了 {text_count} 个文本区域")
    
    # Dilate the text mask slightly to ensure we capture all of the text edges
    text_kernel = np.ones((2, 2), np.uint8)
    dilated_text_mask = cv2.dilate(refined_mask, text_kernel, iterations=1)
    
    # Convert mask to 3-channel for processing
    text_mask_3ch = cv2.cvtColor(dilated_text_mask, cv2.COLOR_GRAY2BGR)
    
    # DIRECT COPY approach - absolutely no modification of text pixels
    # Where the mask is white (255), copy the EXACT original pixel values
    # This ensures text is never blurry
    for c in range(3):
        result_img[:,:,c] = np.where(text_mask_3ch[:,:,c] > 0, 
                                    original_img[:,:,c], 
                                    255)
    
    # Convert back to grayscale for any remaining processing
    cleaned = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
    
    # Determine output path
    if output_path is None:
        # When batch processing, we want to keep original filenames
        # For single file processing, add '_cleaned' suffix if not specified
        path_obj = Path(image_path)
        if image_index is not None:
            # In batch mode, use original filename
            output_path = str(path_obj)
        else:
            # In single file mode, add '_cleaned' suffix
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
        output_folder: Folder to save processed images (if None, will create a folder named 'output')
        no_rotate: If True, skips rotation correction
        aggressive_clean: If True, uses more aggressive cleaning settings
        
    Returns:
        Number of successfully processed images
    """
    # Start timing for the entire batch
    batch_start_time = time.time()
    
    # Create output folder if not specified
    if output_folder is None:
        # Create 'output' folder in the same directory as input_folder
        parent_dir = os.path.dirname(os.path.abspath(input_folder))
        output_folder = os.path.join(parent_dir, 'output')
    
    os.makedirs(output_folder, exist_ok=True)
    print(f"将处理后的图片保存到: {output_folder}")
    
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
        # Use the original filename without adding '_cleaned' suffix
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
    parser.add_argument("--output", "-o", help="输出路径或文件夹（可选，默认为'output'文件夹）")
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
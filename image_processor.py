import cv2
import numpy as np
from PIL import Image
import os
import argparse

def enhanced_noise_removal(image):
    """
    Enhanced noise removal function that focuses on cleaning edges and removing outlines
    """
    # Make a copy of the original
    result = image.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a clean white background
    clean_bg = np.ones_like(image) * 255
    
    # 1. Edge cleaning (more aggressive)
    # Define edge regions
    top_margin = int(image.shape[0] * 0.05)  # 5% from top
    bottom_margin = int(image.shape[0] * 0.07)  # 7% from bottom
    left_margin = int(image.shape[1] * 0.05)  # 5% from left
    right_margin = int(image.shape[1] * 0.05)  # 5% from right
    
    # Create an edge mask (1 for edges, 0 for interior)
    edge_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    edge_mask[:top_margin, :] = 255  # Top edge
    edge_mask[-bottom_margin:, :] = 255  # Bottom edge
    edge_mask[:, :left_margin] = 255  # Left edge
    edge_mask[:, -right_margin:] = 255  # Right edge
    
    # 2. Find dark spots within the edge regions
    # Create binary image focusing on dark regions
    _, dark_binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV) 
    
    # Combine with edge mask to focus only on edge artifacts
    edge_dark = cv2.bitwise_and(dark_binary, edge_mask)
    
    # Dilate to connect nearby dark regions on edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edge_dark_dilated = cv2.dilate(edge_dark, kernel, iterations=1)
    
    # 3. Special processing for top outline (the remnant of the black bar)
    # Create a special mask for the top region where we saw the outline
    top_outline_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    top_outline_height = int(image.shape[0] * 0.08)  # 8% from top
    top_outline_mask[:top_outline_height, :] = 255
    
    # Find horizontal lines in top region
    # First create binary image with lower threshold to catch faint lines
    _, top_binary = cv2.threshold(gray[:top_outline_height, :], 220, 255, cv2.THRESH_BINARY_INV)
    
    # Dilate horizontally to connect line segments
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    top_lines = cv2.dilate(top_binary, horiz_kernel, iterations=1)
    
    # Extend top_lines to full image size
    full_top_lines = np.zeros(image.shape[:2], dtype=np.uint8)
    full_top_lines[:top_outline_height, :] = top_lines
    
    # 4. Combine all noise masks
    noise_mask = cv2.bitwise_or(edge_dark_dilated, full_top_lines)
    
    # 5. Apply morphological operations to clean up the mask
    # Dilate to make sure we get all the noise
    noise_mask = cv2.dilate(noise_mask, kernel, iterations=1)
    
    # 6. Find text regions to preserve
    # Use adaptive thresholding which works well for text
    text_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
    
    # Dilate text slightly to make sure we preserve it all
    text_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    text_mask = cv2.dilate(text_thresh, text_kernel, iterations=1)
    
    # Find text contours (these are the regions we want to preserve)
    text_contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create text protection mask
    text_protection = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Draw only contours that are likely text (based on size and aspect ratio)
    for contour in text_contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h) if h > 0 else 0
        area = w * h
        
        # Check if inside the main body of the document (not near edges)
        is_central = (x > left_margin and 
                     x + w < image.shape[1] - right_margin and
                     y > top_margin and 
                     y + h < image.shape[0] - bottom_margin)
        
        # Text-like characteristics
        if (area > 20 and area < 3000 and  # Not too small or too large
            (0.1 < aspect_ratio < 20) and  # Reasonable aspect ratio for text
            (is_central or  # Either in central area
             (area < 200 and 0.2 < aspect_ratio < 5))):  # Or small with text-like ratio
            cv2.drawContours(text_protection, [contour], -1, 255, -1)
    
    # Dilate text protection to ensure we don't touch text
    text_protection = cv2.dilate(text_protection, kernel, iterations=2)
    
    # Remove text protection from noise mask to ensure we don't remove text
    final_noise_mask = cv2.bitwise_and(noise_mask, cv2.bitwise_not(text_protection))
    
    # 7. Special handling for the outline at the top - more aggressive
    top_strip = int(image.shape[0] * 0.025)  # Very top 2.5%
    top_strip_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    top_strip_mask[:top_strip, :] = 255
    
    # Apply top strip mask directly (ignore text protection for very top)
    final_noise_mask = cv2.bitwise_or(final_noise_mask, top_strip_mask)
    
    # Replace noise areas with clean white
    result[final_noise_mask > 0] = clean_bg[final_noise_mask > 0]
    
    # 8. Additional cleaning of the right and bottom edges
    # Create a special mask for these problem areas
    problem_edges = np.zeros(image.shape[:2], dtype=np.uint8)
    problem_edges[-bottom_margin:, :] = 255  # Bottom
    problem_edges[:, -right_margin:] = 255  # Right
    
    # Get gray values in these areas
    edge_gray = gray.copy()
    edge_gray[problem_edges == 0] = 255  # Only consider problem edges
    
    # Threshold to find darker spots in these areas
    _, edge_thresh = cv2.threshold(edge_gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Dilate to connect nearby spots
    edge_spots = cv2.dilate(edge_thresh, kernel, iterations=2)
    
    # Only consider spots in the problem areas
    edge_spots = cv2.bitwise_and(edge_spots, problem_edges)
    
    # Remove these edge spots (don't worry about text preservation here)
    result[edge_spots > 0] = clean_bg[edge_spots > 0]

    # --- 强力清理顶部黑色色块（保留文字）---
    top_height = int(result.shape[0] * 0.12)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    # 检测顶部文字区域（使用自适应阈值更好地检测文字）
    text_thresh = cv2.adaptiveThreshold(gray[:top_height, :], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
    
    # 找到文字轮廓
    text_contours, _ = cv2.findContours(text_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建文字保护掩码
    text_protect = np.zeros((top_height, result.shape[1]), dtype=np.uint8)
    
    # 只保护看起来像文字的区域（基于大小和宽高比）
    for contour in text_contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h) if h > 0 else 0
        area = w * h
        
        # 文字特征：适当的大小和宽高比
        if (area > 20 and area < 1000 and  # 不太大也不太小
            0.1 < aspect_ratio < 10):  # 合理的宽高比
            cv2.drawContours(text_protect, [contour], -1, 255, -1)
    
    # 对非文字区域进行阈值处理
    _, top_mask = cv2.threshold(gray[:top_height, :], 200, 255, cv2.THRESH_BINARY_INV)
    
    # 从清理掩码中排除文字区域
    top_mask = cv2.bitwise_and(top_mask, cv2.bitwise_not(text_protect))
    
    # 应用掩码（只清理非文字区域）
    result_top = result[:top_height, :]
    result_top[top_mask > 0] = 255
    
    # --- 强力清理底部污点（保留文字）---
    bottom_height = int(result.shape[0] * 0.10)
    
    # 检测底部文字
    text_thresh = cv2.adaptiveThreshold(gray[-bottom_height:, :], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
    
    text_contours, _ = cv2.findContours(text_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_protect = np.zeros((bottom_height, result.shape[1]), dtype=np.uint8)
    
    for contour in text_contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h) if h > 0 else 0
        area = w * h
        
        if (area > 20 and area < 1000 and 0.1 < aspect_ratio < 10):
            cv2.drawContours(text_protect, [contour], -1, 255, -1)
    
    _, bottom_mask = cv2.threshold(gray[-bottom_height:, :], 230, 255, cv2.THRESH_BINARY_INV)
    bottom_mask = cv2.bitwise_and(bottom_mask, cv2.bitwise_not(text_protect))
    
    result_bottom = result[-bottom_height:, :]
    result_bottom[bottom_mask > 0] = 255
    
    # --- 强力清理右侧污点（保留文字）---
    right_width = int(result.shape[1] * 0.08)
    
    # 检测右侧文字
    text_thresh = cv2.adaptiveThreshold(gray[:, -right_width:], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
    
    text_contours, _ = cv2.findContours(text_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_protect = np.zeros((result.shape[0], right_width), dtype=np.uint8)
    
    for contour in text_contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h) if h > 0 else 0
        area = w * h
        
        if (area > 20 and area < 1000 and 0.1 < aspect_ratio < 10):
            cv2.drawContours(text_protect, [contour], -1, 255, -1)
    
    _, right_mask = cv2.threshold(gray[:, -right_width:], 230, 255, cv2.THRESH_BINARY_INV)
    right_mask = cv2.bitwise_and(right_mask, cv2.bitwise_not(text_protect))
    
    result_right = result[:, -right_width:]
    result_right[right_mask > 0] = 255

    return result

def improved_deskew(image):
    """
    Improved deskew function with more conservative approach to preserve text
    """
    # Make a copy of the original image
    original = image.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply very light Gaussian blur to reduce noise but preserve text details
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Use adaptive thresholding which works better for text documents
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    if lines is not None and len(lines) > 5:
        # Calculate angles only from lines that are likely to be horizontal text lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line length
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Only consider lines that are reasonably long (likely text lines)
            if length > image.shape[1] / 5:
                # Calculate angle in degrees
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                
                # Filter angles that are close to horizontal
                # (text lines are typically horizontal or close to it)
                if abs(angle) < 5 or abs(angle - 180) < 5 or abs(angle + 180) < 5:
                    # Normalize angle to small value for skew correction
                    if angle > 90:
                        angle -= 180
                    elif angle < -90:
                        angle += 180
                    
                    angles.append(angle)
        
        if angles:
            # Use median for robustness against outliers
            skew_angle = np.median(angles)
            
            # Very conservative approach: only correct if we're confident about skew
            if len(angles) >= 3 and abs(skew_angle) > 0.5:
                # Calculate rotation matrix
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
                
                # Apply rotation with border replication to avoid black borders
                rotated = cv2.warpAffine(original, M, (w, h), 
                                        flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_REPLICATE)
                return rotated
    
    # If skew detection is unreliable or minimal, return original
    return original

def enhanced_process_image(input_path, output_path, debug=False):
    """
    Enhanced image processing pipeline that carefully preserves text
    """
    # Read the image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Could not read image from {input_path}")
    
    # Create debug directory if needed
    debug_dir = os.path.join(os.path.dirname(output_path), "debug")
    if debug and not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    
    # Create a copy of original for comparison
    original = image.copy()
    if debug:
        cv2.imwrite(os.path.join(debug_dir, "0_original.jpg"), original)
    
    # 1. Remove noise while preserving text
    print("Removing noise and edge artifacts...")
    cleaned = enhanced_noise_removal(image)
    if debug:
        cv2.imwrite(os.path.join(debug_dir, "1_cleaned.jpg"), cleaned)
    
    # 2. Apply additional cleaning to right and bottom edges specifically
    print("Extra cleaning of problem edges...")
    
    # Create a mask for the problematic edges
    edge_mask = np.zeros(cleaned.shape[:2], dtype=np.uint8)
    
    # Define problem edge regions
    bottom_margin = int(cleaned.shape[0] * 0.08)  # 8% from bottom
    right_margin = int(cleaned.shape[1] * 0.06)  # 6% from right
    
    # Mark bottom and right margins
    edge_mask[-bottom_margin:, :] = 255  # Bottom margin
    edge_mask[:, -right_margin:] = 255  # Right margin
    
    # Convert to grayscale for thresholding
    edge_gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    
    # Find dark spots in the edge regions
    _, edge_thresh = cv2.threshold(edge_gray, 245, 255, cv2.THRESH_BINARY_INV)
    
    # Restrict to the edge regions
    edge_spots = cv2.bitwise_and(edge_thresh, edge_mask)
    
    # Dilate to cover entire spots
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edge_spots = cv2.dilate(edge_spots, kernel, iterations=2)
    
    # Clean white background 
    white_bg = np.ones_like(cleaned) * 255
    
    # Replace spots with white
    edge_cleaned = cleaned.copy()
    edge_cleaned[edge_spots > 0] = white_bg[edge_spots > 0]
    
    if debug:
        cv2.imwrite(os.path.join(debug_dir, "2_edge_cleaned.jpg"), edge_cleaned)
    
    # 3. Deskew the image
    print("Correcting skew...")
    deskewed = improved_deskew(edge_cleaned)
    if debug:
        cv2.imwrite(os.path.join(debug_dir, "3_deskewed.jpg"), deskewed)
    
    # 4. Final enhancement - careful contrast improvement to preserve text
    print("Enhancing text visibility...")
    
    # Convert to LAB color space for better contrast enhancement
    lab = cv2.cvtColor(deskewed, cv2.COLOR_BGR2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE only to L channel with conservative parameters
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    
    # Merge channels back
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    
    # Convert back to BGR
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # 5. Final clean-up pass for any remaining artifacts
    # Convert to grayscale
    final_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    # Find any isolated small dark spots
    _, final_thresh = cv2.threshold(final_gray, 230, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(final_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out small isolated spots
    spot_mask = np.zeros(result.shape[:2], dtype=np.uint8)
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Small isolated spots not part of text lines
        if area < 100 and w < 20 and h < 20:
            # Check if it's isolated (not part of a text line)
            is_isolated = True
            
            # Check surrounding area for other contours (15 pixel radius)
            for other in contours:
                if contour is other:
                    continue
                    
                x2, y2, w2, h2 = cv2.boundingRect(other)
                # Calculate distance between centers
                dist = np.sqrt((x + w/2 - x2 - w2/2)**2 + (y + h/2 - y2 - h2/2)**2)
                if dist < 15:
                    is_isolated = False
                    break
                    
            if is_isolated:
                cv2.drawContours(spot_mask, [contour], -1, 255, -1)
    
    # Remove any remaining isolated spots
    result[spot_mask > 0] = white_bg[spot_mask > 0]
    
    # Save the final result
    print(f"Saving processed image to {output_path}")
    cv2.imwrite(output_path, result)
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process document images')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('--output', help='Output image path', default=None)
    parser.add_argument('--debug', help='Save debug images', action='store_true')
    
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output if args.output else f"processed_{os.path.basename(input_path)}"
    
    print(f"Processing {input_path}...")
    result = enhanced_process_image(input_path, output_path, args.debug)
    print(f"Processing complete. Result saved to {output_path}")
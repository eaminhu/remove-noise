#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档扫描图片批量清理脚本
功能：去除扫描噪声，保留文字内容和重要标记（如页码等）
Author: Kris Hu eaminhu@gmail.com
Date: 2025-05-09 12:32:58
LastEditors: Kris Hu eaminhu@gmail.com
LastEditTime: 2025-05-28 09:06:47
"""

import os
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance
import argparse
from pathlib import Path
import time
from tqdm import tqdm

# 设置 tesseract 路径
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

class DocumentCleaner:
    def __init__(self, debug=False):
        self.debug = debug
        
    def detect_text_regions_ocr(self, image):
        """使用OCR检测文字区域，生成文字保护掩码"""
        print("    - 使用OCR检测文字区域...")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 创建文字保护掩码
        text_protect_mask = np.zeros_like(gray, dtype=np.uint8)
        
        try:
            # 使用Tesseract检测中文文字
            ocr_data = pytesseract.image_to_data(
                gray, 
                lang='chi_sim+eng',  # 支持中文和英文
                config='--psm 6',    # 假设单一文本块
                output_type=pytesseract.Output.DICT
            )
            
            text_boxes_count = 0
            for i in range(len(ocr_data['text'])):
                # 只保留置信度较高的文字区域
                confidence = int(ocr_data['conf'][i])
                text = ocr_data['text'][i].strip()
                
                if confidence > 30 and len(text) > 0:  # 降低置信度阈值以保护更多文字
                    x = ocr_data['left'][i]
                    y = ocr_data['top'][i]
                    w = ocr_data['width'][i]
                    h = ocr_data['height'][i]
                    
                    # 扩展文字区域边界，确保完整保护
                    padding = 5
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(gray.shape[1] - x, w + 2 * padding)
                    h = min(gray.shape[0] - y, h + 2 * padding)
                    
                    # 在掩码中标记文字区域
                    text_protect_mask[y:y+h, x:x+w] = 255
                    text_boxes_count += 1
            
            print(f"    - OCR检测到 {text_boxes_count} 个文字区域")
            
        except Exception as e:
            print(f"    - OCR检测失败: {e}")
            print("    - 使用备用文字检测方法...")
            
        # 备用方法：基于图像分析的文字检测
        backup_mask = self.detect_text_regions_cv(gray)
        text_protect_mask = cv2.bitwise_or(text_protect_mask, backup_mask)
        
        return text_protect_mask
    
    def detect_text_regions_cv(self, gray):
        """基于OpenCV的文字区域检测（备用方法）"""
        # 使用自适应阈值检测文字
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY_INV, 25, 15
        )
        
        # 连通域分析，筛选文字区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        text_mask = np.zeros_like(gray, dtype=np.uint8)
        
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            
            # 筛选条件：面积、宽高比、位置等
            aspect_ratio = max(w, h) / (min(w, h) if min(w, h) > 0 else 1)
            
            # 保留可能的文字区域
            if (area > 20 and area < gray.shape[0] * gray.shape[1] * 0.1 and  # 面积合理
                h > 8 and w > 3 and  # 最小尺寸
                aspect_ratio < 10):   # 宽高比不要太极端
                
                # 扩展边界
                padding = 3
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(gray.shape[1] - x, w + 2 * padding)
                h = min(gray.shape[0] - y, h + 2 * padding)
                
                text_mask[y:y+h, x:x+w] = 255
        
        return text_mask
    
    def detect_corner_page_numbers(self, image):
        """检测四个角落的页码，生成保护掩码"""
        print("    - 检测角落页码...")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 定义四个角落区域
        corner_size = min(h//8, w//8, 150)  # 角落检测区域大小
        corners = [
            (0, corner_size, 0, corner_size),                    # 左上角
            (0, corner_size, w-corner_size, w),                  # 右上角
            (h-corner_size, h, 0, corner_size),                  # 左下角
            (h-corner_size, h, w-corner_size, w)                 # 右下角
        ]
        
        page_number_mask = np.zeros_like(gray, dtype=np.uint8)
        
        for i, (y1, y2, x1, x2) in enumerate(corners):
            corner_region = gray[y1:y2, x1:x2]
            
            try:
                # 使用OCR检测角落的数字
                ocr_data = pytesseract.image_to_data(
                    corner_region,
                    lang='eng',  # 页码通常是数字，用英文检测
                    config='--psm 8 -c tessedit_char_whitelist=0123456789',  # 只检测数字
                    output_type=pytesseract.Output.DICT
                )
                
                for j in range(len(ocr_data['text'])):
                    confidence = int(ocr_data['conf'][j])
                    text = ocr_data['text'][j].strip()
                    
                    # 检测到数字且置信度较高
                    if confidence > 50 and text.isdigit():
                        x = ocr_data['left'][j] + x1
                        y = ocr_data['top'][j] + y1
                        w_box = ocr_data['width'][j]
                        h_box = ocr_data['height'][j]
                        
                        # 扩展保护区域
                        padding = 10
                        x = max(0, x - padding)
                        y = max(0, y - padding)
                        w_box = min(gray.shape[1] - x, w_box + 2 * padding)
                        h_box = min(gray.shape[0] - y, h_box + 2 * padding)
                        
                        page_number_mask[y:y+h_box, x:x+w_box] = 255
                        print(f"    - 在角落{i+1}检测到页码: {text}")
                        
            except Exception as e:
                # OCR失败时，保护整个角落的小文字区域
                _, thresh = cv2.threshold(corner_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 50 < area < 2000:  # 适合页码的面积范围
                        x, y, w_box, h_box = cv2.boundingRect(contour)
                        # 转换到全图坐标
                        x += x1
                        y += y1
                        page_number_mask[y:y+h_box, x:x+w_box] = 255
        
        return page_number_mask
    
    def detect_noise_regions(self, image, text_protect_mask, page_number_mask):
        """检测需要清除的噪声区域"""
        print("    - 检测噪声区域...")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 合并所有保护区域
        protected_mask = cv2.bitwise_or(text_protect_mask, page_number_mask)
        
        # 1. 检测边缘噪声
        edge_noise_mask = self.detect_edge_noise(gray)
        
        # 2. 检测装订痕迹
        binding_noise_mask = self.detect_binding_marks(gray)
        
        # 3. 检测大块污点
        stain_noise_mask = self.detect_large_stains(gray)
        
        # 4. 检测小斑点
        spot_noise_mask = self.detect_small_spots(gray)
        
        # 合并所有噪声区域
        all_noise_mask = cv2.bitwise_or(edge_noise_mask, binding_noise_mask)
        all_noise_mask = cv2.bitwise_or(all_noise_mask, stain_noise_mask)
        all_noise_mask = cv2.bitwise_or(all_noise_mask, spot_noise_mask)
        
        # 从噪声掩码中排除保护区域
        final_noise_mask = cv2.bitwise_and(all_noise_mask, cv2.bitwise_not(protected_mask))
        
        return final_noise_mask
    
    def detect_edge_noise(self, gray):
        """检测边缘噪声"""
        h, w = gray.shape
        edge_mask = np.zeros_like(gray, dtype=np.uint8)
        
        # 边缘区域大小
        border_size = min(h//20, w//20, 30)
        
        # 检测四个边缘
        edges = [
            (0, border_size, 0, w),          # 上边缘
            (h-border_size, h, 0, w),        # 下边缘
            (0, h, 0, border_size),          # 左边缘
            (0, h, w-border_size, w)         # 右边缘
        ]
        
        for y1, y2, x1, x2 in edges:
            edge_region = gray[y1:y2, x1:x2]
            mean_val = np.mean(edge_region)
            
            # 如果边缘区域整体偏暗，标记为噪声
            if mean_val < 200:
                threshold = min(mean_val + np.std(edge_region) * 1.5, 180)
                noise_pixels = (edge_region < threshold).astype(np.uint8) * 255
                edge_mask[y1:y2, x1:x2] = np.maximum(edge_mask[y1:y2, x1:x2], noise_pixels)
        
        return edge_mask
    
    def detect_binding_marks(self, gray):
        """检测装订痕迹（通常在左边缘）"""
        h, w = gray.shape
        binding_mask = np.zeros_like(gray, dtype=np.uint8)
        
        # 检测左边缘的垂直线条（装订痕迹）
        left_region = gray[:, :w//10]  # 左边10%区域
        
        # 使用形态学操作检测垂直线条
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h//20))
        vertical_lines = cv2.morphologyEx(left_region, cv2.MORPH_OPEN, vertical_kernel)
        
        # 阈值处理
        _, vertical_thresh = cv2.threshold(vertical_lines, 100, 255, cv2.THRESH_BINARY_INV)
        
        # 将检测结果映射到全图
        binding_mask[:, :w//10] = vertical_thresh
        
        return binding_mask
    
    def detect_large_stains(self, gray):
        """检测大块污点"""
        # 使用形态学操作检测大块暗区域
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        
        # 检测暗区域
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # 形态学闭运算连接相近的暗点
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # 连通域分析，只保留大面积的污点
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
        stain_mask = np.zeros_like(gray, dtype=np.uint8)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # 只标记大面积的污点（避免误删文字）
            if area > 1000 and area < gray.shape[0] * gray.shape[1] * 0.05:
                stain_mask[labels == i] = 255
        
        return stain_mask
    
    def detect_small_spots(self, gray):
        """检测小斑点"""
        # 使用形态学操作检测小的孤立暗点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # 检测小暗点
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
        
        # 连通域分析
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)
        spot_mask = np.zeros_like(gray, dtype=np.uint8)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # 只标记小面积的斑点
            if 5 < area < 200:
                spot_mask[labels == i] = 255
        
        return spot_mask
    
    def clean_document(self, image_path, output_path):
        """主要的文档清理函数"""
        print(f"处理文件: {os.path.basename(image_path)}")
        
        # 检查文件
        if not os.path.exists(image_path):
            print(f"  ❌ 文件不存在: {image_path}")
            return False
        
        # 读取图像
        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                print(f"  ❌ 无法读取图像: {image_path}")
                return False
            
            h, w = image.shape[:2]
            print(f"  📐 图像尺寸: {w} x {h}")
            
            # 如果图像太大，适当缩放以提高处理速度
            max_dimension = 3000
            if max(h, w) > max_dimension:
                scale = max_dimension / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h))
                print(f"  🔄 图像已缩放至: {new_w} x {new_h}")
            
        except Exception as e:
            print(f"  ❌ 读取图像失败: {e}")
            return False
        
        try:
            # 保存原始图像
            original_img = image.copy()
            
            # 1. 检测文字区域（OCR + 图像分析）
            print("  🔍 检测文字区域...")
            text_protect_mask = self.detect_text_regions_ocr(image)
            
            # 2. 检测角落页码
            print("  🔍 检测页码...")
            page_number_mask = self.detect_corner_page_numbers(image)
            
            # 3. 检测噪声区域
            print("  🔍 检测噪声区域...")
            noise_mask = self.detect_noise_regions(image, text_protect_mask, page_number_mask)
            
            # 4. 应用清理
            print("  🧹 清理图像...")
            result = self.apply_cleaning(original_img, noise_mask, text_protect_mask, page_number_mask)
            
            # 5. 图像增强
            print("  ✨ 图像增强...")
            result = self.enhance_image(result)
            
            # 保存结果
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            success = cv2.imwrite(output_path, result)
            
            if success:
                print(f"  ✅ 已保存到: {os.path.basename(output_path)}")
                
                # 保存调试图像
                if self.debug:
                    self.save_debug_images(image, text_protect_mask, page_number_mask, 
                                         noise_mask, result, output_path)
                
                return True
            else:
                print(f"  ❌ 保存失败")
                return False
                
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def apply_cleaning(self, image, noise_mask, text_protect_mask, page_number_mask):
        """应用清理操作"""
        result = image.copy()
        
        # 合并所有保护区域
        protected_mask = cv2.bitwise_or(text_protect_mask, page_number_mask)
        
        # 确保噪声掩码不包含保护区域
        final_noise_mask = cv2.bitwise_and(noise_mask, cv2.bitwise_not(protected_mask))
        
        # 方法1：直接将噪声区域设为白色
        result[final_noise_mask > 0] = [255, 255, 255]
        
        # 方法2：对于较大的噪声区域，使用修复算法
        if np.sum(final_noise_mask) > 0:
            # 膨胀噪声掩码以获得更好的修复效果
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated_mask = cv2.dilate(final_noise_mask, kernel, iterations=1)
            
            # 确保膨胀后的掩码不覆盖保护区域
            dilated_mask = cv2.bitwise_and(dilated_mask, cv2.bitwise_not(protected_mask))
            
            # 使用修复算法
            if np.sum(dilated_mask) > 0:
                result = cv2.inpaint(result, dilated_mask, 3, cv2.INPAINT_TELEA)
        
        # 最后，确保所有保护区域使用原始像素
        for c in range(3):
            result[:,:,c] = np.where(protected_mask == 255, 
                                   image[:,:,c], 
                                   result[:,:,c])
        
        return result
    
    def enhance_image(self, image):
        """图像增强"""
        # 转换为PIL图像
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 轻微增强对比度
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.1)
        
        # 轻微增强清晰度
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.05)
        
        # 转换回OpenCV格式
        enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return enhanced
    
    def save_debug_images(self, original, text_mask, page_mask, noise_mask, result, output_path):
        """保存调试图像"""
        debug_dir = os.path.join(os.path.dirname(output_path), 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        
        cv2.imwrite(os.path.join(debug_dir, f"{base_name}_01_original.jpg"), original)
        cv2.imwrite(os.path.join(debug_dir, f"{base_name}_02_text_mask.jpg"), text_mask)
        cv2.imwrite(os.path.join(debug_dir, f"{base_name}_03_page_mask.jpg"), page_mask)
        cv2.imwrite(os.path.join(debug_dir, f"{base_name}_04_noise_mask.jpg"), noise_mask)
        cv2.imwrite(os.path.join(debug_dir, f"{base_name}_05_result.jpg"), result)
        
        print(f"  🔍 调试图像已保存到: {debug_dir}")
    
    def process_folder(self, input_folder, output_folder):
        """批量处理文件夹"""
        # 支持的图像格式
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        # 获取所有图像文件
        image_files = []
        for ext in extensions:
            image_files.extend(Path(input_folder).glob(f'*{ext}'))
            image_files.extend(Path(input_folder).glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"❌ 在 {input_folder} 中没有找到图像文件")
            return
        
        print(f"📁 找到 {len(image_files)} 个图像文件")
        print("=" * 60)
        
        # 创建输出目录
        os.makedirs(output_folder, exist_ok=True)
        
        # 处理每个文件
        success_count = 0
        start_time = time.time()
        
        for i, image_file in enumerate(tqdm(image_files, desc="处理进度")):
            try:
                filename = image_file.name
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_folder, f"{name}_cleaned{ext}")
                
                if self.clean_document(str(image_file), output_path):
                    success_count += 1
                    
            except Exception as e:
                print(f"❌ 处理文件 {image_file.name} 时出错: {e}")
        
        # 处理完成统计
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "=" * 60)
        print(f"📊 处理完成！")
        print(f"✅ 成功处理: {success_count}/{len(image_files)} 个文件")
        print(f"⏱️  总耗时: {total_time:.2f} 秒")
        if success_count > 0:
            print(f"⚡ 平均每张: {total_time/success_count:.2f} 秒")
        print(f"📂 输出目录: {output_folder}")

def main():
    parser = argparse.ArgumentParser(description='批量清理扫描文档图片，保留文字和页码，去除污点噪声')
    parser.add_argument('input_folder', help='输入文件夹路径')
    parser.add_argument('output_folder', help='输出文件夹路径')
    parser.add_argument('--debug', action='store_true', help='启用调试模式，保存中间处理结果')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input_folder):
        print(f"❌ 输入目录不存在: {args.input_folder}")
        return
    
    # 创建清理器
    cleaner = DocumentCleaner(debug=args.debug)
    
    print(f"🚀 开始处理文档...")
    print(f"📂 输入目录: {args.input_folder}")
    print(f"📂 输出目录: {args.output_folder}")
    print(f"🔧 调试模式: {'开启' if args.debug else '关闭'}")
    print("-" * 60)
    
    # 开始处理
    cleaner.process_folder(args.input_folder, args.output_folder)

if __name__ == "__main__":
    # 如果直接运行脚本，使用默认路径进行测试
    if len(os.sys.argv) == 1:
        # 测试模式
        input_directory = "input_images"
        output_directory = "output_images"
        
        if os.path.exists(input_directory):
            print("🧪 测试模式")
            cleaner = DocumentCleaner(debug=True)
            cleaner.process_folder(input_directory, output_directory)
        else:
            print("请使用命令行参数运行脚本:")
            print("python index.py <输入目录> <输出目录> [--debug]")
            print("\n示例:")
            print("python index.py ./input_images ./output_images --debug")
    else:
        main()
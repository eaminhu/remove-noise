# Image Processing Tool

一个基于 Python 的图片批量处理工具，用于去除图片噪点和自动校正图片倾斜角度。

## 功能特点

- 🔨 批量处理多个图片文件
- 🎯 自动去除图片噪点
- 📐 自动检测和校正图片倾斜角度
- 💾 保留原图，处理后的图片单独保存
- 📊 支持多种图片格式 (PNG, JPG, JPEG)

## 环境要求

- Python 3.7+
- OpenCV (cv2)
- NumPy
- scikit-image

## 安装步骤

1. 克隆仓库到本地：
```bash
git clone https://github.com/eaminhu/image-processing-tool.git
cd image-processing-tool
```

2. 创建虚拟环境（推荐）：
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/MacOS
python3 -m venv venv
source venv/bin/activate
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 命令行使用

```bash
python process_images.py --input_dir "input_folder" --output_dir "output_folder" [options]

python3 image_processor.py index.jpg --output processed_index.jpg --debug
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --input_dir | 输入图片文件夹路径 | ./input |
| --output_dir | 处理后图片保存路径 | ./output |
| --denoise_strength | 降噪强度 (1-10) | 5 |
| --auto_rotate | 是否自动校正倾斜 | True |
| --format | 输出图片格式 | jpg |

### 示例

1. 基本使用：
```bash
python process_images.py --input_dir "my_images"
```

2. 指定参数：
```bash
python process_images.py --input_dir "my_images" --output_dir "processed" --denoise_strength 7
```

## 处理流程

1. 图片预处理
   - 读取图片
   - 转换色彩空间
   - 调整图片大小（如需要）

2. 噪点去除
   - 使用高斯模糊
   - 应用非局部均值去噪算法
   - 保持图片细节

3. 倾斜校正
   - 检测图片中的直线
   - 计算倾斜角度
   - 旋转校正

4. 后处理
   - 优化对比度
   - 保存处理后的图片

## 配置文件说明

可以通过 `config.yaml` 文件修改默认配置：

```yaml
processing:
  denoise_strength: 5
  auto_rotate: true
  preserve_metadata: true

output:
  format: jpg
  quality: 95
  prefix: "processed_"
```

## 常见问题

1. **Q: 程序运行时报 "ImportError: No module named cv2"**
   - A: 请确认是否已安装 OpenCV：`pip install opencv-python`

2. **Q: 处理后的图片质量下降**
   - A: 调整 denoise_strength 参数，使用较小的值

3. **Q: 校正后图片出现黑边**
   - A: 这是正常现象，可以通过裁剪参数来去除黑边

## 注意事项

- 处理大量图片时，请确保有足够的磁盘空间
- 建议先处理少量图片测试效果
- 原始图片会被保留，处理后的图片将保存在输出目录

## 贡献指南

欢迎提交 Pull Request 或创建 Issue!

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 许可证

该项目使用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 联系方式

作者：eaminhu
邮箱：eaminhu@gmail.com

## 更新日志

### v1.0.0 (2025-05-13)
- 初始版本发布
- 实现基础的降噪功能
- 实现自动校正倾斜功能

---
*最后更新时间：2025-05-13*
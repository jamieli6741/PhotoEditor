#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
证件照自动生成器 - 简化版
支持人脸检测、背景替换、尺寸调整、批量处理等功能
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import argparse
import os
import json
from typing import Tuple, Optional, List, Dict
import logging
from pathlib import Path
import concurrent.futures
from functools import lru_cache

# 尝试导入MediaPipe和其他可选依赖
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

# 预设证件照规格
PHOTO_SPECS = {
    "1寸": (295, 413),
    "2寸": (413, 579),
    "小1寸": (260, 378),
    "小2寸": (378, 522),
    "护照": (390, 567),
    "签证": (390, 567),
    "驾照": (260, 378),
    "身份证": (358, 441),
    "社保卡": (358, 441),
    "工作证": (413, 579),
    "美国签证": (600, 600)  # 2×2英寸，51×51mm，正方形
}

# 常用背景颜色
BG_COLORS = {
    "white": (255, 255, 255),
    "blue": (255, 0, 0),
    "red": (0, 0, 255),
    "light_blue": (255, 191, 0),
    "gray": (128, 128, 128)
}

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class IDPhotoGenerator:
    def __init__(self, config_path: Optional[str] = None):
        """初始化证件照生成器"""
        self.config = self._load_config(config_path)
        self._init_detectors()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置文件"""
        default_config = {
            "face_ratio": 0.75,
            "face_position_y": 0.35,
            "enhance_params": {
                "sharpen_radius": 1,
                "sharpen_percent": 150,
                "contrast_factor": 1.1,
                "brightness_offset": 10
            },
            "background_removal": {
                "method": "grabcut",
                "grabcut_iterations": 5,
                "morphology_kernel_size": 3
            },
            "face_detection": {
                "mediapipe_confidence": 0.5,
                "opencv_scale_factor": 1.1,
                "opencv_min_neighbors": 5
            },
            "us_visa": {
                "face_ratio": 0.60,  # 美国签证要求50%-69%，设置为60%
                "face_position_y": 0.30
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config

    def _init_detectors(self):
        """初始化检测器"""
        # OpenCV人脸检测器
        self.opencv_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # MediaPipe检测器
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_mesh = mp.solutions.face_mesh
            self.use_mediapipe = True
            logger.info("MediaPipe人脸检测器初始化成功")
        else:
            self.use_mediapipe = False
            logger.info("MediaPipe不可用，使用OpenCV进行人脸检测")

    @lru_cache(maxsize=10)
    def _get_face_detector_config(self) -> Dict:
        """获取人脸检测配置（缓存）"""
        return self.config.get("face_detection", {})

    def detect_face_with_landmarks(self, image: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]:
        """
        检测人脸并返回关键点信息
        
        Returns:
            (face_bbox, landmarks) - 人脸边界框和关键点
        """
        if self.use_mediapipe:
            return self._detect_face_mediapipe_advanced(image)
        else:
            face_bbox = self.detect_face_opencv(image)
            return face_bbox, None

    def _detect_face_mediapipe_advanced(self, image: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]:
        """增强版MediaPipe人脸检测"""
        config = self._get_face_detector_config()
        confidence = config.get("mediapipe_confidence", 0.5)
        
        # 人脸检测
        with self.mp_face_detection.FaceDetection(
            model_selection=1, 
            min_detection_confidence=confidence
        ) as face_detection:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_image)
            
            face_bbox = None
            if results.detections:
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                face_bbox = (x, y, width, height)
        
        # 获取面部关键点
        landmarks = None
        if face_bbox:
            with self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=confidence
            ) as face_mesh:
                results = face_mesh.process(rgb_image)
                if results.multi_face_landmarks:
                    landmarks_3d = results.multi_face_landmarks[0]
                    h, w, _ = image.shape
                    landmarks = []
                    for landmark in landmarks_3d.landmark:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        landmarks.append([x, y])
                    landmarks = np.array(landmarks)
        
        return face_bbox, landmarks

    def detect_face_opencv(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """OpenCV人脸检测（改进版）"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 直方图均衡化提高检测效果
        gray = cv2.equalizeHist(gray)
        
        config = self._get_face_detector_config()
        scale_factor = config.get("opencv_scale_factor", 1.1)
        min_neighbors = config.get("opencv_min_neighbors", 5)
        
        # 多尺度检测
        faces = self.opencv_detector.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) > 0:
            # 选择最大且最居中的人脸
            h, w = gray.shape
            center_x, center_y = w // 2, h // 2
            
            best_face = None
            best_score = -1
            
            for face in faces:
                x, y, fw, fh = face
                face_center_x = x + fw // 2
                face_center_y = y + fh // 2
                
                # 计算综合得分：尺寸 + 位置
                size_score = (fw * fh) / (w * h)  # 相对大小
                distance = np.sqrt((face_center_x - center_x)**2 + (face_center_y - center_y)**2)
                position_score = 1 / (1 + distance / min(w, h))  # 距离中心的远近
                
                total_score = size_score * 0.7 + position_score * 0.3
                
                if total_score > best_score:
                    best_score = total_score
                    best_face = face
            
            return tuple(best_face)
        
        return None

    def remove_background_advanced(self, image: np.ndarray, method: str = "auto") -> np.ndarray:
        """
        增强版背景移除
        
        Args:
            image: 输入图像
            method: 移除方法 ("auto", "rembg", "grabcut", "threshold")
        """
        if method == "auto":
            # 自动选择最佳方法
            if REMBG_AVAILABLE:
                method = "rembg"
            else:
                method = "grabcut"
        
        if method == "rembg" and REMBG_AVAILABLE:
            return self._remove_background_rembg(image)
        elif method == "grabcut":
            return self._remove_background_grabcut_enhanced(image)
        else:
            return self._remove_background_threshold(image)

    def _remove_background_rembg(self, image: np.ndarray) -> np.ndarray:
        """使用rembg库移除背景（AI方法，效果最好）"""
        # 转换为PIL图像
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 使用rembg移除背景
        result_pil = remove(pil_image)
        
        # 转换回OpenCV格式
        result_array = np.array(result_pil)
        
        if result_array.shape[2] == 4:  # 已有alpha通道
            return cv2.cvtColor(result_array, cv2.COLOR_RGBA2BGRA)
        else:
            # 添加alpha通道
            height, width = result_array.shape[:2]
            result_bgra = np.zeros((height, width, 4), dtype=np.uint8)
            result_bgra[:, :, :3] = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
            result_bgra[:, :, 3] = 255
            return result_bgra

    def _remove_background_grabcut_enhanced(self, image: np.ndarray) -> np.ndarray:
        """增强版GrabCut背景移除"""
        height, width = image.shape[:2]
        mask = np.zeros((height, width), np.uint8)
        
        # 改进的前景区域估计
        padding_x = max(width // 8, 20)
        padding_y = max(height // 10, 20)
        rect = (padding_x, padding_y, width - 2*padding_x, height - 2*padding_y)
        
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        config = self.config.get("background_removal", {})
        iterations = config.get("grabcut_iterations", 5)
        
        # 应用GrabCut
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
        
        # 创建二值掩码
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # 形态学操作优化掩码
        kernel_size = config.get("morphology_kernel_size", 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)
        
        # 高斯模糊平滑边缘
        mask2 = cv2.GaussianBlur(mask2.astype(np.float32), (3, 3), 1)
        mask2 = (mask2 * 255).astype(np.uint8)
        
        # 创建结果图像
        result = np.zeros((height, width, 4), dtype=np.uint8)
        result[:, :, :3] = image
        result[:, :, 3] = mask2
        
        return result

    def _remove_background_threshold(self, image: np.ndarray) -> np.ndarray:
        """阈值方法背景移除（简单方法）"""
        # 转换为HSV进行更好的颜色分割
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 创建掩码（这里假设背景是相对均匀的）
        # 可以根据具体需求调整阈值
        lower = np.array([0, 0, 200])  # 假设背景是亮色
        upper = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower, upper)
        
        # 反转掩码（前景为白，背景为黑）
        mask = cv2.bitwise_not(mask)
        
        # 形态学操作清理掩码
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 创建带alpha通道的结果
        result = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        result[:, :, :3] = image
        result[:, :, 3] = mask
        
        return result

    def add_background(self, image_with_alpha: np.ndarray, bg_color: Tuple[int, int, int]) -> np.ndarray:
        """
        为带有alpha通道的图像添加背景色
        
        Args:
            image_with_alpha: 带有alpha通道的图像 (BGRA格式)
            bg_color: 背景颜色 (B, G, R)
        
        Returns:
            添加背景后的图像 (BGR格式)
        """
        if image_with_alpha.shape[2] != 4:
            # 如果没有alpha通道，直接返回原图
            return image_with_alpha
        
        height, width = image_with_alpha.shape[:2]
        
        # 分离RGB和Alpha通道
        bgr_image = image_with_alpha[:, :, :3]
        alpha_channel = image_with_alpha[:, :, 3]
        
        # 创建背景图像
        background = np.full((height, width, 3), bg_color, dtype=np.uint8)
        
        # 将alpha通道归一化到0-1范围
        alpha_normalized = alpha_channel.astype(np.float32) / 255.0
        
        # 扩展alpha通道维度以便广播
        alpha_3d = np.stack([alpha_normalized] * 3, axis=2)
        
        # 执行alpha混合
        # result = foreground * alpha + background * (1 - alpha)
        result = (bgr_image.astype(np.float32) * alpha_3d + 
                  background.astype(np.float32) * (1 - alpha_3d))
        
        return result.astype(np.uint8)

    def enhance_image_advanced(self, image: np.ndarray) -> np.ndarray:
        """
        高级图像增强
        """
        # 转换为PIL进行处理
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        config = self.config.get("enhance_params", {})
        
        # 锐化
        radius = config.get("sharpen_radius", 1)
        percent = config.get("sharpen_percent", 150)
        pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=3))
        
        # 对比度增强
        contrast_factor = config.get("contrast_factor", 1.1)
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(contrast_factor)
        
        # 亮度调整
        brightness_factor = 1.0 + config.get("brightness_offset", 10) / 255.0
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(brightness_factor)
        
        # 色彩饱和度微调
        enhancer = ImageEnhance.Color(pil_image)
        pil_image = enhancer.enhance(1.05)
        
        # 转换回OpenCV格式
        enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # 降噪处理
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 3, 3, 7, 21)
        
        return enhanced

    def adjust_face_position_advanced(self, image: np.ndarray, face_bbox: Tuple[int, int, int, int],
                                    landmarks: Optional[np.ndarray] = None,
                                    target_ratio: float = 0.75, 
                                    target_size: Tuple[int, int] = (295, 413)) -> np.ndarray:
        """
        高级人脸位置调整（考虑关键点信息）
        """
        x, y, w, h = face_bbox
        
        # 如果有关键点，使用眼部位置进行更精确的定位
        if landmarks is not None and len(landmarks) > 468:  # MediaPipe面部网格有468个点
            # 获取眼部关键点（简化版）
            left_eye = landmarks[33:43].mean(axis=0)  # 左眼区域
            right_eye = landmarks[362:372].mean(axis=0)  # 右眼区域
            eye_center = ((left_eye + right_eye) / 2).astype(int)
            
            # 使用眼部中心作为参考点
            face_center_x, face_center_y = eye_center
            
            # 根据眼部位置调整人脸框
            face_height = max(h, int(abs(landmarks[10][1] - landmarks[152][1]) * 1.3))  # 下巴到额头
        else:
            # 使用原有的人脸中心
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            face_height = h
        
        # 计算缩放比例
        target_face_height = int(target_size[1] * target_ratio)
        scale_factor = target_face_height / face_height
        
        # 缩放图像
        new_height, new_width = image.shape[:2]
        new_height = int(new_height * scale_factor)
        new_width = int(new_width * scale_factor)
        
        scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # 重新计算人脸中心
        new_face_center_x = int(face_center_x * scale_factor)
        new_face_center_y = int(face_center_y * scale_factor)
        
        # 创建画布
        canvas = np.full((target_size[1], target_size[0], 3), 255, dtype=np.uint8)
        
        # 计算粘贴位置
        face_position_y = self.config.get("face_position_y", 0.35)
        paste_x = target_size[0] // 2 - new_face_center_x
        paste_y = int(target_size[1] * face_position_y) - new_face_center_y
        
        # 执行粘贴
        self._paste_image_to_canvas(scaled_image, canvas, paste_x, paste_y)
        
        return canvas

    def _paste_image_to_canvas(self, src_image: np.ndarray, canvas: np.ndarray, 
                             paste_x: int, paste_y: int):
        """将源图像粘贴到画布上"""
        src_h, src_w = src_image.shape[:2]
        canvas_h, canvas_w = canvas.shape[:2]
        
        # 计算实际粘贴区域
        src_x1 = max(0, -paste_x)
        src_y1 = max(0, -paste_y)
        src_x2 = min(src_w, src_x1 + canvas_w - max(0, paste_x))
        src_y2 = min(src_h, src_y1 + canvas_h - max(0, paste_y))
        
        dst_x1 = max(0, paste_x)
        dst_y1 = max(0, paste_y)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        
        if src_x2 > src_x1 and src_y2 > src_y1:
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = src_image[src_y1:src_y2, src_x1:src_x2]

    def generate_id_photo_advanced(self, input_path: str, output_path: str,
                                 spec_name: str = "1寸", 
                                 bg_color_name: str = "white",
                                 face_ratio: Optional[float] = None,
                                 remove_bg_method: str = "auto") -> Dict:
        """
        高级证件照生成
        
        Returns:
            处理结果信息字典
        """
        result = {
            "success": False,
            "message": "",
            "processing_time": 0,
            "face_detected": False,
            "output_size": None
        }
        
        import time
        start_time = time.time()
        
        try:
            # 读取图像
            image = cv2.imread(input_path)
            if image is None:
                result["message"] = f"无法读取图像: {input_path}"
                return result
            
            logger.info(f"开始处理图像: {input_path}")
            
            # 获取规格和颜色
            target_size = PHOTO_SPECS.get(spec_name, PHOTO_SPECS["1寸"])
            bg_color = BG_COLORS.get(bg_color_name, BG_COLORS["white"])
            
            # 如果是美国签证且没有指定face_ratio，使用特殊配置
            if spec_name == "美国签证" and face_ratio is None:
                face_ratio = self.config.get("us_visa", {}).get("face_ratio", 0.60)
                # 同时更新face_position_y
                self.config["face_position_y"] = self.config.get("us_visa", {}).get("face_position_y", 0.30)
            elif face_ratio is None:
                face_ratio = self.config["face_ratio"]
            
            # 检测人脸和关键点
            face_bbox, landmarks = self.detect_face_with_landmarks(image)
            if face_bbox is None:
                result["message"] = "未检测到人脸"
                return result
            
            result["face_detected"] = True
            logger.info(f"检测到人脸: {face_bbox}")
            
            processed_image = image.copy()
            
            # 背景移除和替换
            if remove_bg_method != "none":
                logger.info("处理背景...")
                image_with_alpha = self.remove_background_advanced(processed_image, remove_bg_method)
                processed_image = self.add_background(image_with_alpha, bg_color)
            
            # 调整人脸位置
            logger.info("调整人脸位置和比例...")
            final_image = self.adjust_face_position_advanced(
                processed_image, face_bbox, landmarks, face_ratio, target_size
            )
            
            # 图像增强
            logger.info("增强图像质量...")
            final_image = self.enhance_image_advanced(final_image)
            
            # 保存结果
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            success = cv2.imwrite(output_path, final_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if success:
                result["success"] = True
                result["message"] = "证件照生成成功"
                result["output_size"] = target_size
                logger.info(f"证件照已保存到: {output_path}")
            else:
                result["message"] = "保存图像失败"
            
        except Exception as e:
            result["message"] = f"处理过程中出错: {str(e)}"
            logger.error(result["message"])
        
        finally:
            result["processing_time"] = time.time() - start_time
        
        return result

    def batch_process(self, input_dir: str, output_dir: str, **kwargs) -> List[Dict]:
        """批量处理证件照"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        results = []
        
        # 多线程处理
        max_workers = min(4, len(image_files))  # 限制线程数
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {}
            
            for image_file in image_files:
                output_file = output_path / f"{image_file.stem}_id{image_file.suffix}"
                future = executor.submit(
                    self.generate_id_photo_advanced,
                    str(image_file),
                    str(output_file),
                    **kwargs
                )
                future_to_file[future] = image_file
            
            for future in concurrent.futures.as_completed(future_to_file):
                image_file = future_to_file[future]
                try:
                    result = future.result()
                    result["input_file"] = str(image_file)
                    results.append(result)
                    
                    status = "✅" if result["success"] else "❌"
                    print(f"{status} {image_file.name}: {result['message']}")
                    
                except Exception as e:
                    results.append({
                        "input_file": str(image_file),
                        "success": False,
                        "message": f"处理异常: {str(e)}"
                    })
                    print(f"❌ {image_file.name}: 处理异常")
        
        return results


def create_default_config(config_path: str):
    """创建默认配置文件"""
    default_config = {
        "face_ratio": 0.75,
        "face_position_y": 0.35,
        "us_visa": {
            "face_ratio": 0.60,
            "face_position_y": 0.30
        },
        "enhance_params": {
            "sharpen_radius": 1,
            "sharpen_percent": 150,
            "contrast_factor": 1.1,
            "brightness_offset": 10
        },
        "background_removal": {
            "method": "grabcut",
            "grabcut_iterations": 5,
            "morphology_kernel_size": 3
        },
        "face_detection": {
            "mediapipe_confidence": 0.5,
            "opencv_scale_factor": 1.1,
            "opencv_min_neighbors": 5
        }
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, ensure_ascii=False, indent=2)
    
    print(f"默认配置文件已创建: {config_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='证件照自动生成器')
    parser.add_argument('input', help='输入图片路径或目录')
    parser.add_argument('output', help='输出图片路径或目录')
    
    # 基本参数
    parser.add_argument('--spec', choices=list(PHOTO_SPECS.keys()), default='1寸',
                        help='证件照规格')
    parser.add_argument('--bg-color', choices=list(BG_COLORS.keys()), default='white',
                        help='背景颜色')
    parser.add_argument('--face-ratio', type=float, help='人脸占画面高度的比例')
    parser.add_argument('--bg-method', choices=['auto', 'rembg', 'grabcut', 'threshold', 'none'],
                        default='auto', help='背景移除方法')
    
    # 高级选项
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--batch', action='store_true', help='批量处理模式')
    parser.add_argument('--create-config', help='创建默认配置文件')
    
    args = parser.parse_args()
    
    # 创建配置文件
    if args.create_config:
        create_default_config(args.create_config)
        return
    
    # 创建生成器
    generator = IDPhotoGenerator(args.config)
    
    if args.batch:
        # 批量处理
        if not os.path.isdir(args.input):
            print("❌ 批量模式需要输入目录")
            return
        
        results = generator.batch_process(
            args.input, args.output,
            spec_name=args.spec,
            bg_color_name=args.bg_color,
            face_ratio=args.face_ratio,
            remove_bg_method=args.bg_method
        )
        
        # 输出批量处理统计
        success_count = sum(1 for r in results if r["success"])
        total_count = len(results)
        print(f"\n📊 批量处理完成: {success_count}/{total_count} 成功")
        
        # 保存处理报告
        report_path = os.path.join(args.output, "processing_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"📋 处理报告已保存: {report_path}")
        
    else:
        # 单张处理
        if not os.path.exists(args.input):
            print(f"❌ 输入文件不存在: {args.input}")
            return
        
        # 确保输出目录存在
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 处理单张图片
        result = generator.generate_id_photo_advanced(
            input_path=args.input,
            output_path=args.output,
            spec_name=args.spec,
            bg_color_name=args.bg_color,
            face_ratio=args.face_ratio,
            remove_bg_method=args.bg_method
        )
        
        # 输出结果
        status = "✅" if result["success"] else "❌"
        print(f"{status} {result['message']}")
        
        if result["success"]:
            print(f"📏 输出尺寸: {result['output_size'][0]}×{result['output_size'][1]}")
            print(f"⏱️  处理时间: {result['processing_time']:.2f}秒")
            
            # 美国签证特殊提示
            if args.spec == "美国签证":
                print("🇺🇸 美国签证照片要求:")
                print("   - 头部应占照片高度的50%-69%")
                print("   - 眼部位置在照片中心线上方")
                print("   - 保持中性表情，嘴巴闭合")
                print("   - 直视相机，眼睛清晰可见")
        
        # 显示可用选项提示
        if not result["success"] and not result["face_detected"]:
            print("\n💡 提示:")
            print("1. 确保照片中有清晰可见的人脸")
            print("2. 人脸应该正面朝向相机")
            print("3. 光线充足，避免阴影遮挡")
            print("4. 可以尝试不同的检测方法（安装 mediapipe: pip install mediapipe）")


def print_usage_examples():
    """打印使用示例"""
    examples = """
🚀 证件照生成器使用示例:

基本使用:
  python id_photo_generator.py input.jpg output.jpg

指定证件照规格:
  python id_photo_generator.py input.jpg output.jpg --spec 2寸
  python id_photo_generator.py input.jpg output.jpg --spec 美国签证

指定背景颜色:
  python id_photo_generator.py input.jpg output.jpg --bg-color blue

美国签证照片生成:
  python id_photo_generator.py input.jpg us_visa.jpg --spec 美国签证
  python id_photo_generator.py input.jpg us_visa.jpg --spec 美国签证 --face-ratio 0.60

使用AI背景移除 (需要安装 rembg):
  python id_photo_generator.py input.jpg output.jpg --bg-method rembg

批量处理:
  python id_photo_generator.py input_dir/ output_dir/ --batch --spec 美国签证

创建配置文件:
  python id_photo_generator.py --create-config config.json

使用自定义配置:
  python id_photo_generator.py input.jpg output.jpg --config config.json

📏 支持的证件照规格:
  1寸 (295×413), 2寸 (413×579), 小1寸 (260×378)
  小2寸 (378×522), 护照 (390×567), 签证 (390×567)
  驾照 (260×378), 身份证 (358×441), 社保卡 (358×441)
  美国签证 (600×600) - 2×2英寸正方形格式

🎨 支持的背景颜色:
  white (白色), blue (蓝色), red (红色)
  light_blue (浅蓝色), gray (灰色)

⚙️ 背景移除方法:
  auto (自动选择), rembg (AI方法，效果最好)
  grabcut (传统方法), threshold (阈值方法)
  none (保留原背景)

🇺🇸 美国签证照片特殊要求:
  - 尺寸: 2×2英寸 (600×600像素)
  - 头部比例: 50%-69% (默认60%)
  - 背景: 白色
  - 表情: 自然，嘴巴闭合
  - 眼睛: 直视相机，清晰可见
  - 头饰: 除宗教原因外不允许
  - 眼镜: 可以佩戴，但不能反光
"""
    print(examples)


if __name__ == "__main__":
    import sys
    
    # 如果没有参数，显示帮助信息
    if len(sys.argv) == 1:
        print_usage_examples()
    else:
        main()
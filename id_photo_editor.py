#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
证件照自动生成器
支持人脸检测、背景替换、尺寸调整等功能
使用OpenCV + MediaPipe替代dlib
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter
import argparse
import os
from typing import Tuple, Optional
import logging

# 尝试导入MediaPipe（推荐，精度更高）
try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe未安装，将使用OpenCV进行人脸检测")
    print("推荐安装: pip install mediapipe")

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IDPhotoGenerator:
    def __init__(self):
        """初始化证件照生成器"""
        # 初始化OpenCV人脸检测器
        self.opencv_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # 初始化MediaPipe人脸检测器（如果可用）
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_face_mesh = mp.solutions.face_mesh
            self.use_mediapipe = True
            logger.info("MediaPipe人脸检测器初始化成功")
        else:
            self.use_mediapipe = False
            logger.info("使用OpenCV进行人脸检测")

    def detect_face_mediapipe(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        使用MediaPipe检测人脸位置

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            人脸边界框 (x, y, w, h) 或 None
        """
        with self.mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
        ) as face_detection:
            # 转换颜色格式
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_image)

            if results.detections:
                # 取第一个检测到的人脸
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box

                # 转换为绝对坐标
                h, w, _ = image.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)

                return (x, y, width, height)

        return None

    def detect_face_opencv(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        使用OpenCV检测人脸位置

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            人脸边界框 (x, y, w, h) 或 None
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 使用多种参数进行检测，提高检测率
        faces = self.opencv_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) > 0:
            # 如果检测到多个人脸，选择最大的那个
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            return tuple(largest_face)

        # 如果没检测到，尝试更宽松的参数
        faces = self.opencv_detector.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20)
        )

        if len(faces) > 0:
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            return tuple(largest_face)

        return None

    def detect_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        检测人脸位置（优先使用MediaPipe，后备使用OpenCV）

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            人脸边界框 (x, y, w, h) 或 None
        """
        # 优先使用MediaPipe
        if self.use_mediapipe:
            face_bbox = self.detect_face_mediapipe(image)
            if face_bbox is not None:
                return face_bbox
            logger.info("MediaPipe未检测到人脸，尝试使用OpenCV")

        # 使用OpenCV作为后备
        return self.detect_face_opencv(image)

    def get_face_landmarks_mediapipe(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        使用MediaPipe获取人脸关键点（可用于更精确的人脸定位）

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            关键点数组或None
        """
        if not self.use_mediapipe:
            return None

        with self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
        ) as face_mesh:

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                h, w, _ = image.shape

                # 转换为像素坐标
                points = []
                for landmark in landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    points.append([x, y])

                return np.array(points)

        return None

    def remove_background(self, image: np.ndarray, method: str = 'grabcut') -> np.ndarray:
        """
        移除背景

        Args:
            image: 输入图像
            method: 背景移除方法 ('grabcut' 或 'threshold')

        Returns:
            带alpha通道的图像
        """
        if method == 'grabcut':
            return self._remove_background_grabcut(image)
        else:
            return self._remove_background_threshold(image)

    def _remove_background_grabcut(self, image: np.ndarray) -> np.ndarray:
        """使用GrabCut算法移除背景"""
        height, width = image.shape[:2]

        # 创建掩码
        mask = np.zeros((height, width), np.uint8)

        # 定义前景区域（中心区域作为可能的前景）
        rect = (width // 6, height // 8, width * 2 // 3, height * 7 // 8)

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # 应用GrabCut算法
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

        # 创建最终掩码
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # 平滑掩码边缘
        mask2 = cv2.medianBlur(mask2, 5)

        # 创建带alpha通道的图像
        result = np.zeros((height, width, 4), dtype=np.uint8)
        result[:, :, :3] = image
        result[:, :, 3] = mask2 * 255

        return result

    def _remove_background_threshold(self, image: np.ndarray) -> np.ndarray:
        """使用简单阈值方法移除背景（适用于纯色背景）"""
        # 转换为HSV色彩空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 定义背景颜色范围（假设为浅色背景）
        lower_bg = np.array([0, 0, 200])
        upper_bg = np.array([180, 30, 255])

        # 创建掩码
        mask = cv2.inRange(hsv, lower_bg, upper_bg)
        mask = cv2.bitwise_not(mask)  # 反转掩码

        # 形态学操作清理掩码
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 创建带alpha通道的图像
        height, width = image.shape[:2]
        result = np.zeros((height, width, 4), dtype=np.uint8)
        result[:, :, :3] = image
        result[:, :, 3] = mask

        return result

    def add_background(self, image_with_alpha: np.ndarray,
                       bg_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """
        添加背景色

        Args:
            image_with_alpha: 带alpha通道的图像
            bg_color: 背景颜色 (B, G, R)

        Returns:
            BGR图像
        """
        height, width = image_with_alpha.shape[:2]

        # 创建背景
        background = np.full((height, width, 3), bg_color, dtype=np.uint8)

        # 提取前景和alpha通道
        foreground = image_with_alpha[:, :, :3]
        alpha = image_with_alpha[:, :, 3] / 255.0

        # 混合前景和背景
        result = np.zeros((height, width, 3), dtype=np.uint8)
        for c in range(3):
            result[:, :, c] = (alpha * foreground[:, :, c] +
                               (1 - alpha) * background[:, :, c])

        return result

    def adjust_face_ratio(self, image: np.ndarray, face_bbox: Tuple[int, int, int, int],
                          target_ratio: float = 0.75, target_size: Tuple[int, int] = (295, 413)) -> np.ndarray:
        """
        调整人脸在画面中的比例

        Args:
            image: 输入图像
            face_bbox: 人脸边界框 (x, y, w, h)
            target_ratio: 目标人脸占画面高度的比例
            target_size: 目标尺寸 (width, height)

        Returns:
            调整后的图像
        """
        x, y, w, h = face_bbox

        # 计算人脸中心
        face_center_x = x + w // 2
        face_center_y = y + h // 2

        # 根据目标比例计算新的画面大小
        target_face_height = int(target_size[1] * target_ratio)
        scale_factor = target_face_height / h

        # 计算缩放后的图像尺寸
        new_height, new_width = image.shape[:2]
        new_height = int(new_height * scale_factor)
        new_width = int(new_width * scale_factor)

        # 缩放图像
        scaled_image = cv2.resize(image, (new_width, new_height))

        # 重新计算人脸中心位置
        new_face_center_x = int(face_center_x * scale_factor)
        new_face_center_y = int(face_center_y * scale_factor)

        # 创建目标尺寸的画布
        canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        canvas.fill(255)  # 白色背景

        # 计算放置位置（人脸居中，稍微偏上）
        paste_x = target_size[0] // 2 - new_face_center_x
        paste_y = int(target_size[1] * 0.35) - new_face_center_y  # 人脸中心位于画面上部1/3处

        # 计算实际粘贴区域
        src_x1 = max(0, -paste_x)
        src_y1 = max(0, -paste_y)
        src_x2 = min(new_width, src_x1 + target_size[0] - max(0, paste_x))
        src_y2 = min(new_height, src_y1 + target_size[1] - max(0, paste_y))

        dst_x1 = max(0, paste_x)
        dst_y1 = max(0, paste_y)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        # 粘贴图像
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = scaled_image[src_y1:src_y2, src_x1:src_x2]

        return canvas

    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        增强图像质量

        Args:
            image: 输入图像

        Returns:
            增强后的图像
        """
        # 转换为PIL图像进行处理
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # 轻微锐化
        pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))

        # 转换回OpenCV格式
        enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # 色彩平衡
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=10)

        return enhanced

    def generate_id_photo(self, input_path: str, output_path: str,
                          face_ratio: float = 0.75, bg_color: Tuple[int, int, int] = (255, 255, 255),
                          size: Tuple[int, int] = (295, 413), remove_bg: bool = True) -> bool:
        """
        生成证件照

        Args:
            input_path: 输入图片路径
            output_path: 输出图片路径
            face_ratio: 人脸占画面高度的比例
            bg_color: 背景颜色 (B, G, R)
            size: 输出尺寸 (width, height)
            remove_bg: 是否移除原背景

        Returns:
            是否成功生成
        """
        try:
            # 读取图像
            image = cv2.imread(input_path)
            if image is None:
                logger.error(f"无法读取图像: {input_path}")
                return False

            logger.info(f"处理图像: {input_path}")

            # 检测人脸
            face_bbox = self.detect_face(image)
            if face_bbox is None:
                logger.error("未检测到人脸")
                return False

            logger.info(f"检测到人脸: {face_bbox}")

            processed_image = image.copy()

            # 移除背景（可选）
            if remove_bg:
                logger.info("移除背景中...")
                image_with_alpha = self.remove_background(processed_image)
                processed_image = self.add_background(image_with_alpha, bg_color)

            # 调整人脸比例和尺寸
            logger.info("调整人脸比例...")
            final_image = self.adjust_face_ratio(processed_image, face_bbox, face_ratio, size)

            # 图像增强
            logger.info("增强图像质量...")
            final_image = self.enhance_image(final_image)

            # 保存结果
            success = cv2.imwrite(output_path, final_image)
            if success:
                logger.info(f"证件照已保存到: {output_path}")
                return True
            else:
                logger.error("保存图像失败")
                return False

        except Exception as e:
            logger.error(f"处理过程中出错: {str(e)}")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='证件照自动生成器')
    parser.add_argument('input', help='输入图片路径')
    parser.add_argument('output', help='输出图片路径')
    parser.add_argument('--face-ratio', type=float, default=0.75,
                        help='人脸占画面高度的比例 (默认: 0.75)')
    parser.add_argument('--bg-color', nargs=3, type=int, default=[255, 255, 255],
                        help='背景颜色 BGR值 (默认: 255 255 255 白色)')
    parser.add_argument('--size', nargs=2, type=int, default=[295, 413],
                        help='输出尺寸 宽x高 (默认: 295 413)')
    parser.add_argument('--keep-bg', action='store_true',
                        help='保留原背景，不进行背景替换')

    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        return

    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 创建生成器并处理图像
    generator = IDPhotoGenerator()

    success = generator.generate_id_photo(
        input_path=args.input,
        output_path=args.output,
        face_ratio=args.face_ratio,
        bg_color=tuple(args.bg_color),
        size=tuple(args.size),
        remove_bg=not args.keep_bg
    )

    if success:
        print("✅ 证件照生成成功！")
    else:
        print("❌ 证件照生成失败，请检查输入图像和参数。")


if __name__ == "__main__":
    main()
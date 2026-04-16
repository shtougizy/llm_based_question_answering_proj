"""
图像预处理模块
增强对纸张蜷曲、圈画标记、低质量图片的识别鲁棒性
"""
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import logging

logger = logging.getLogger(__name__)


def preprocess_image(image_path: str) -> str:
    """
    对题目图片进行预处理，返回处理后的临时文件路径
    处理流程：
    1. 透视校正（去除纸张蜷曲）
    2. 自适应去噪
    3. 对比度增强
    4. 圈画区域弱化（减少红色笔迹干扰）
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"OpenCV 无法读取图片，跳过预处理: {image_path}")
            return image_path

        original_h, original_w = img.shape[:2]

        # Step 1: 透视校正（处理纸张蜷曲）
        img = _deskew(img)

        # Step 2: 弱化圈画标记（红色/蓝色笔迹）
        img = _weaken_annotations(img)

        # Step 3: 自适应对比度增强（CLAHE）
        img = _enhance_contrast(img)

        # Step 4: 去噪
        img = _denoise(img)

        # 保存预处理结果
        out_path = image_path.replace('.', '_processed.')
        if out_path == image_path:
            out_path = image_path + '_processed.jpg'
        cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        logger.info(f"图像预处理完成: {out_path}")
        return out_path

    except Exception as e:
        logger.warning(f"图像预处理失败，使用原图: {e}")
        return image_path


def _deskew(img: np.ndarray) -> np.ndarray:
    """
    透视校正：检测文档边缘并矫正倾斜/蜷曲
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 高斯模糊去除噪点
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Canny 边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        # 膨胀边缘
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # 寻找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img

        # 找最大轮廓（文档边界）
        max_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(max_contour)
        img_area = img.shape[0] * img.shape[1]

        # 只有当最大轮廓面积 > 图片面积的 20% 时才做透视校正
        if area < img_area * 0.2:
            return img

        # 多边形近似
        peri = cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, 0.02 * peri, True)

        # 只处理四边形（标准文档）
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            # 排序：左上、右上、右下、左下
            rect = _order_points(pts)
            # 计算目标尺寸
            (tl, tr, br, bl) = rect
            w = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
            h = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
            dst = np.array([
                [0, 0], [w-1, 0], [w-1, h-1], [0, h-1]
            ], dtype=np.float32)
            M = cv2.getPerspectiveTransform(rect, dst)
            img = cv2.warpPerspective(img, M, (int(w), int(h)))

        return img
    except Exception as e:
        logger.debug(f"透视校正跳过: {e}")
        return img


def _order_points(pts: np.ndarray) -> np.ndarray:
    """将四个点排序为 左上、右上、右下、左下"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # 左上（x+y最小）
    rect[2] = pts[np.argmax(s)]   # 右下（x+y最大）
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上（y-x最小）
    rect[3] = pts[np.argmax(diff)]  # 左下（y-x最大）
    return rect


def _weaken_annotations(img: np.ndarray) -> np.ndarray:
    """
    弱化圈画标记：将红色/蓝色笔迹区域替换为接近背景的颜色
    使 InternVL 专注于印刷文字
    """
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 红色范围（圆圈、划线常用红笔）
        red_lower1 = np.array([0, 80, 80])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 80, 80])
        red_upper2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)

        # 蓝色范围（蓝笔圈画）
        blue_lower = np.array([100, 80, 80])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

        # 合并掩码，膨胀使覆盖范围稍大
        annotation_mask = red_mask | blue_mask
        kernel = np.ones((3, 3), np.uint8)
        annotation_mask = cv2.dilate(annotation_mask, kernel, iterations=1)

        # 用 inpaint 修复（用周围像素填充笔迹区域）
        if annotation_mask.sum() > 0:
            img = cv2.inpaint(img, annotation_mask, 3, cv2.INPAINT_TELEA)

        return img
    except Exception as e:
        logger.debug(f"圈画弱化跳过: {e}")
        return img


def _enhance_contrast(img: np.ndarray) -> np.ndarray:
    """
    自适应直方图均衡化（CLAHE）提升对比度
    对低光照、纸张泛黄的图片效果显著
    """
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return img
    except Exception as e:
        logger.debug(f"对比度增强跳过: {e}")
        return img


def _denoise(img: np.ndarray) -> np.ndarray:
    """
    非局部均值去噪（保留边缘细节）
    处理拍摄噪点，使文字更清晰
    """
    try:
        img = cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)
        return img
    except Exception as e:
        logger.debug(f"去噪跳过: {e}")
        return img
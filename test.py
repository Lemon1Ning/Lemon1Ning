import cv2
import numpy as np
from glob import glob
import os

def preprocess_template(template_path):
    """
    模板预处理：增强边缘特征，降噪，统一灰度化
    """
    # 读取模板并转为灰度图
    template = cv2.imread(template_path, 0)
    if template is None:
        print(f"警告：模板 {template_path} 读取失败，已跳过")
        return None
    # 高斯降噪：消除模板中的噪点
    template = cv2.GaussianBlur(template, (3, 3), 0)
    # 边缘检测：突出电机轮廓特征（核心优化）
    template = cv2.Canny(template, 50, 150)
    return template

def multi_scale_template_match(target_gray, template, threshold=0.7):
    """
    改进的多尺度匹配：缩放模板而非目标图像，效率更高
    返回：匹配到的所有有效框 [x1, y1, x2, y2]
    """
    h, w = template.shape[:2]
    target_h, target_w = target_gray.shape[:2]
    match_boxes = []
    # 缩放范围：0.5~2.0倍（覆盖电机常见尺寸变化），步长0.1更精细
    scales = np.arange(0.5, 2.1, 0.1)

    for scale in scales:
        # 缩放模板（而非目标图），减少计算量
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)
        # 跳过模板缩放后超过目标图像的情况
        if scaled_w > target_w or scaled_h > target_h:
            continue
        scaled_template = cv2.resize(template, (scaled_w, scaled_h))

        # 模板匹配：使用归一化相关系数（对亮度变化鲁棒）
        result = cv2.matchTemplate(target_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
        # 提取匹配得分≥阈值的位置
        loc = np.where(result >= threshold)

        # 遍历匹配位置，生成框并过滤无效区域
        for pt in zip(*loc[::-1]):  # pt为匹配框左上角坐标
            x1, y1 = pt[0], pt[1]
            x2, y2 = x1 + scaled_w, y1 + scaled_h
            # 过滤条件1：面积过滤（排除过小的误匹配，根据电机实际大小调整）
            area = (x2 - x1) * (y2 - y1)
            if area < 200:  # 最小面积阈值，可根据实际场景调整
                continue
            # 过滤条件2：宽高比过滤（电机多为方形/圆柱形，排除过扁/过长的框）
            aspect_ratio = (x2 - x1) / (y2 - y1)
            if not 0.3 < aspect_ratio < 3.0:  # 宽高比范围，可调整
                continue
            match_boxes.append([x1, y1, x2, y2])

    return match_boxes

def optimized_nms(boxes, iou_threshold=0.2):
    """
    优化的非极大值抑制：修复原代码排序逻辑，优先保留高匹配度的框（按面积排序）
    """
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    # 计算每个框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 按面积降序排序（大框优先，电机通常有固定大小，大框更可能是真实目标）
    indices = np.argsort(areas)[::-1]
    keep = []

    while len(indices) > 0:
        # 保留当前面积最大的框
        current_idx = indices[0]
        keep.append(current_idx)
        # 计算当前框与其他框的IOU
        xx1 = np.maximum(x1[current_idx], x1[indices[1:]])
        yy1 = np.maximum(y1[current_idx], y1[indices[1:]])
        xx2 = np.minimum(x2[current_idx], x2[indices[1:]])
        yy2 = np.minimum(y2[current_idx], y2[indices[1:]])
        # 计算交叠区域的宽高
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算IOU
        iou = inter / (areas[current_idx] + areas[indices[1:]] - inter)
        # 保留IOU小于阈值的框
        indices = indices[1:][iou <= iou_threshold]

    # 返回去重后的框，转为整数
    return [boxes[i].astype(np.int32) for i in keep]

def count_motors_template_matching(target_path, template_dir, match_threshold=0.7, iou_threshold=0.2):
    """
    主函数：模板匹配电机计数
    参数：
        target_path: 目标图像路径
        template_dir: 模板文件夹路径
        match_threshold: 匹配得分阈值（0~1，越大越严格）
        iou_threshold: NMS的IOU阈值（0~1，越小去重越严格）
    返回：电机数量，标注后的图像
    """
    # 读取目标图像
    target_img = cv2.imread(target_path)
    if target_img is None:
        print(f"错误：目标图像 {target_path} 读取失败")
        return 0, None
    # 目标图像预处理：灰度化+降噪
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.GaussianBlur(target_gray, (3, 3), 0)

    # 1. 加载所有模板并预处理
    template_paths = glob(os.path.join(template_dir, "*.png"))  # 仅支持PNG模板
    if not template_paths:
        print(f"警告：模板文件夹 {template_dir} 中未找到PNG模板")
        return 0, target_img.copy()
    templates = []
    for path in template_paths:
        temp = preprocess_template(path)
        if temp is not None:
            templates.append(temp)

    # 2. 多模板+多尺度匹配，收集所有框
    all_boxes = []
    for template in templates:
        boxes = multi_scale_template_match(target_gray, template, match_threshold)
        all_boxes.extend(boxes)
    if not all_boxes:
        print("警告：未匹配到任何电机区域")
        return 0, target_img.copy()

    # 3. 非极大值抑制去重
    final_boxes = optimized_nms(all_boxes, iou_threshold)
    motor_count = len(final_boxes)

    # 4. 绘制标注结果
    result_img = target_img.copy()
    for (x1, y1, x2, y2) in final_boxes:
        # 绘制绿色框（BGR：0,255,0），线宽2
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # 绘制计数文本（红色）
    cv2.putText(result_img, f"Motor Count: {motor_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    return motor_count, result_img

# 程序入口
if __name__ == "__main__":
    # -------------------------- 需手动调整的参数 --------------------------
    TARGET_IMAGE_PATH = "motors.png"  # 你的目标图像路径
    TEMPLATE_DIR = "templates"        # 你的模板文件夹路径
    MATCH_THRESHOLD = 0.65            # 匹配阈值（可根据效果调整：0.5~0.8）
    IOU_THRESHOLD = 0.2               # NMS去重阈值（可根据效果调整：0.1~0.3）
    # ---------------------------------------------------------------------

    # 执行计数
    count, result_img = count_motors_template_matching(
        TARGET_IMAGE_PATH,
        TEMPLATE_DIR,
        MATCH_THRESHOLD,
        IOU_THRESHOLD
    )

    # 保存并显示结果
    if result_img is not None:
        cv2.imwrite("motor_count_result.png", result_img)
        cv2.imshow("Motor Count Result", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print(f"最终检测到的电机数量：{count}")
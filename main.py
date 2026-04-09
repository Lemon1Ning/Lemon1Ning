import cv2  # 导入OpenCV库，用于图像处理、模板匹配等核心操作
import numpy as np  # 导入numpy库，用于数组运算和数值处理
from glob import glob  # 导入glob模块，用于批量获取模板文件路径


def global_non_max_suppression(all_boxes, iou_threshold=0.15):
    """
    全局非极大值抑制（NMS）：去除所有匹配结果中的重复框，确保每个电机只被标注一次
    参数说明：
        all_boxes: 所有模板匹配到的原始框列表，格式为[[x1, y1, x2, y2], ...]
                   其中(x1,y1)是框的左上角坐标，(x2,y2)是右下角坐标
        iou_threshold: 交并比（IOU）阈值（范围0~1）
                       - 作用：判断两个框是否为重复框（重叠程度）
                       - 数值越小：去重越严格（即使轻微重叠也会被过滤）
                       - 默认0.15：适合电机排列较密集的场景
    返回值：
        去重后的框列表，格式与输入一致
    """
    # 若没有匹配框，直接返回空列表
    if len(all_boxes) == 0:
        return []

    # 将框列表转换为numpy数组，方便批量计算
    boxes = np.array(all_boxes, dtype=np.float32)
    # 提取所有框的坐标（x1左上角x，y1左上角y，x2右下角x，y2右下角y）
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # 计算每个框的面积（用于后续交并比计算）
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 按框的右下角y坐标排序（用于迭代筛选，确保从下到上处理框）
    indices = np.argsort(y2)
    keep = []  # 存储最终保留的框的索引

    # 迭代筛选非重复框
    while len(indices) > 0:
        # 保留当前最底部的框（最后一个索引，视为可信度较高的框）
        last = len(indices) - 1
        i = indices[last]
        keep.append(i)

        # 计算当前框与其他所有框的交叠区域坐标
        xx1 = np.maximum(x1[i], x1[indices[:last]])  # 交叠区域左上角x
        yy1 = np.maximum(y1[i], y1[indices[:last]])  # 交叠区域左上角y
        xx2 = np.minimum(x2[i], x2[indices[:last]])  # 交叠区域右下角x
        yy2 = np.minimum(y2[i], y2[indices[:last]])  # 交叠区域右下角y

        # 计算交叠区域的宽和高（确保非负，避免无效值）
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h  # 交叠区域的面积

        # 计算交并比（IOU）：交叠面积 / 并集面积（并集=两框面积和-交叠面积）
        iou = inter / (areas[i] + areas[indices[:last]] - inter)

        # 保留交并比小于阈值的框（移除与当前框重叠严重的框）
        indices = indices[:last][iou <= iou_threshold]

    # 将保留的框转换为整数坐标（图像坐标为整数）并返回
    return [boxes[i].astype(np.int32) for i in keep]




def match_template_single(target_img, template_path, threshold=0.75):
    """
    单模板多尺度匹配：用单个电机模板在目标图像中匹配同类电机，支持不同尺寸的电机
    参数说明：
        target_img: 目标图像（BGR格式，即原始彩色图）
        template_path: 单个电机模板的路径（需为.png格式灰度图，建议仅包含电机主体）
        threshold: 匹配阈值（范围0~1）
                   - 作用：筛选匹配度高的区域（得分≥阈值才视为有效匹配）
                   - 数值越大：匹配越严格（仅保留高度相似的区域，减少误检但可能漏检）
                   - 数值越小：匹配越宽松（保留更多潜在目标，可能增加误检）
    返回值：
        该模板匹配到的所有框列表，格式为[[x1, y1, x2, y2], ...]
    """
    # 读取模板图像（以灰度图方式，减少计算量并突出轮廓特征）
    template = cv2.imread(template_path, 0)  # 0表示强制转为灰度图
    if template is None:  # 若模板读取失败（路径错误或文件损坏），跳过该模板
        print(f"警告：模板 {template_path} 读取失败，已自动跳过")
        return []
    h, w = template.shape[:2]  # 获取模板的高度和宽度（用于计算匹配框尺寸）

    # 将目标图像转为灰度图（消除颜色干扰，加快匹配速度）
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    match_boxes = []  # 存储当前模板匹配到的所有框

    # 多尺度匹配：在不同缩放比例下搜索电机（处理电机大小差异）
    # np.linspace(0.8, 1.6, 16)：生成16个从0.8到1.6的缩放比例（覆盖电机可能的尺寸范围）
    # [::-1]：反转顺序，从大到小匹配（优先处理大尺寸，减少无效计算）
    for scale in np.linspace(0.8, 1.6, 16)[::-1]:
        # 按当前比例缩放目标图像（模拟模板放大/缩小，避免模板变形）
        resized = cv2.resize(
            target_gray,
            (int(target_gray.shape[1] * scale),  # 缩放后的图像宽度
             int(target_gray.shape[0] * scale))   # 缩放后的图像高度
        )
        # 计算缩放比例（用于将匹配结果还原到原始图像尺寸）
        r = target_gray.shape[1] / resized.shape[1]

        # 若缩放后的图像小于模板，停止该尺度匹配（避免无效计算）
        if resized.shape[0] < h or resized.shape[1] < w:
            break

        # 模板匹配：使用归一化相关系数法（TM_CCOEFF_NORMED）
        # 特点：对亮度变化有一定鲁棒性，匹配得分范围为0~1（1表示完全匹配）
        result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
        # 提取匹配得分≥阈值的位置（满足条件的潜在电机区域）
        loc = np.where(result >= threshold)

        # 遍历所有符合条件的匹配位置，计算并记录匹配框（还原到原图尺寸）
        for pt in zip(*loc[::-1]):  # pt为缩放后图像中的匹配框左上角坐标
            x1 = int(pt[0] * r)  # 转换为原始图像的左上角x坐标
            y1 = int(pt[1] * r)  # 转换为原始图像的左上角y坐标
            x2 = int(x1 + w * r)  # 转换为原始图像的右下角x坐标（模板宽度×缩放比例）
            y2 = int(y1 + h * r)  # 转换为原始图像的右下角y坐标（模板高度×缩放比例）
            match_boxes.append([x1, y1, x2, y2])  # 保存当前匹配框

    return match_boxes  # 返回该模板匹配到的所有框


def count_motors_multi_template(target_path, template_dir):
    """
    多模板匹配计数主函数：整合所有模板的匹配结果，去重后统计电机数量并标注
    参数说明：
        target_path: 目标图像路径（包含多个电机的原始图像，如"motors.png"）
        template_dir: 模板文件夹路径（存放所有电机模板的文件夹，如"templates"）
    返回值：
        电机数量（整数），标注后的结果图像（BGR格式）
    """
    # 读取目标图像（BGR格式，用于后续标注结果）
    target_img = cv2.imread(target_path)
    if target_img is None:  # 若目标图像读取失败，返回0和空
        print(f"错误：目标图像 {target_path} 读取失败，请检查路径是否正确")
        return 0, target_img

    # 1. 收集所有模板的匹配框（未去重）
    all_raw_boxes = []  # 存储所有模板匹配到的原始框（可能存在重复）
    # 获取模板文件夹中所有.png格式的模板路径
    template_paths = glob(f"{template_dir}/*.png")
    if not template_paths:  # 若模板文件夹为空，返回0和原图
        print(f"警告：模板文件夹 {template_dir} 中未找到.png格式的模板文件")
        return 0, target_img

    # 遍历每个模板，调用单模板匹配函数，汇总所有匹配框
    for template_path in template_paths:
        # 调用单模板匹配（当前阈值0.7：中等严格度，平衡检出率和误检率）
        boxes = match_template_single(target_img, template_path, threshold=0.71)
        all_raw_boxes.extend(boxes)  # 将当前模板的匹配框添加到总列表

    # 2. 全局非极大值抑制（去重所有重复框）
    # iou_threshold=0.15：严格去重，避免同一电机被多个框标注
    final_boxes = global_non_max_suppression(all_raw_boxes, iou_threshold=0.15)

    # 3. 绘制结果并计数
    result_img = target_img.copy()  # 复制原图用于标注（避免修改原始图像）
    # 遍历去重后的框，绘制绿色边框（绿色：(0,255,0)，线宽2）
    for (x1, y1, x2, y2) in final_boxes:
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # 在图像左上角添加计数文本（红色：(0,0,255)，字体大小1，线宽2）
    cv2.putText(result_img, f"Motor Count: {len(final_boxes)}",
                (50, 50),  # 文本左上角坐标（距离图像左上角50像素）
                cv2.FONT_HERSHEY_SIMPLEX,  # 字体类型
                1,  # 字体大小
                (0, 0, 255),  # 文本颜色（BGR格式，红色）
                2  # 文本线条厚度
                )

    return len(final_boxes), result_img  # 返回最终电机数量和标注图


# ------------------- 程序运行入口 -------------------
if __name__ == "__main__":
    # 目标图像路径（需替换为你的图像实际路径）
    target_path = "motors.png"
    # 模板文件夹路径（需替换为你的模板文件夹实际路径）
    template_dir = "templates"

    # 调用多模板匹配计数函数，获取电机数量和结果图
    motor_count, result_img = count_motors_multi_template(target_path, template_dir)

    # 保存结果图到当前目录（文件名为result.jpg）
    cv2.imwrite("result.jpg", result_img)
    # 显示结果图（窗口标题为"Result"）
    cv2.imshow("Result", result_img)
    cv2.waitKey(0)  # 等待用户按下任意键（按ESC或关闭窗口可退出）
    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口

    # 打印最终的电机数量
    print(f"去重后电机数量：{motor_count}")

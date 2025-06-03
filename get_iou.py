import os
import tensorflow as tf
import numpy as np
from PIL import Image

def load_mask_as_binary_tensor(path):
    img = Image.open(path).convert('L')
    arr = np.array(img).astype(np.float32)

    # 自动处理归一化或整数型图像
    if arr.max() <= 1.0:
        binary = (arr > 0.5).astype(np.int32)
    else:
        binary = (arr > 127).astype(np.int32)

    return tf.convert_to_tensor(binary)


def compute_iou_per_image(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.int32)
    
    intersection = tf.reduce_sum(tf.cast(tf.logical_and(y_true == 1, y_pred == 1), tf.float32))
    union = tf.reduce_sum(tf.cast(tf.logical_or(y_true == 1, y_pred == 1), tf.float32))
    iou = tf.math.divide_no_nan(intersection, union)
    return iou.numpy()

def get_image_files(directory):
    return sorted([
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

def main(pred_dir, true_dir, output_file):
    pred_files = get_image_files(pred_dir)
    true_files = get_image_files(true_dir)

    assert pred_files == true_files, "预测和真实标签文件名不匹配"

    ious = []
    with open(output_file, "w") as f:
        for fname in pred_files:
            pred_path = os.path.join(pred_dir, fname)
            true_path = os.path.join(true_dir, fname)

            y_pred = load_mask_as_binary_tensor(pred_path)
            y_true = load_mask_as_binary_tensor(true_path)

            iou = compute_iou_per_image(y_true, y_pred)
            line = f"{fname} IoU: {iou:.4f}"
            print(line)
            f.write(line + "\n")
            ious.append(iou)

        mean_iou = np.mean(ious)
        mean_line = f"Mean IoU over {len(ious)} images: {mean_iou:.4f}"
        print(mean_line)
        f.write(mean_line + "\n")

if __name__ == "__main__":
    pred_dir = "/hy-tmp/unet3p/predictions"
    true_dir = "/hy-tmp/unet3p/data/val/mask"
    output_file = "./iou_results.txt"
    main(pred_dir, true_dir, output_file)

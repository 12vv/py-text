import torch
import os
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
import pdb

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def split_feature_by_token_length(feature, tokenizer, max_token_len=512):
    max_token_len -= 10
    tokens = tokenizer.tokenize(feature)
    total_tokens = len(tokens)

    # split tokenized feature
    splitted_features = []
    start_idx = 0
    while start_idx < total_tokens:
        end_idx = min(start_idx + max_token_len, total_tokens) # sliding window
        splitted_tokens = tokens[start_idx:end_idx]
        splitted_text = tokenizer.convert_tokens_to_string(splitted_tokens)
        splitted_features.append(splitted_text)
        start_idx += max_token_len // 2
        if end_idx == total_tokens:
            break

    if len(splitted_features) == 0:
        splitted_features = [feature]
        print(f"!!!!!!!!!!!!!!!!!!!feature!!!!!!!!!!!!!!!!: {feature}")
    return splitted_features

def get_image_paths(folder_path):
    image_paths = []
    # 遍历文件夹及其子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 判断文件是否为图片文件
            if file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                # 构建图片文件的完整路径
                image_path = os.path.join(root, file)
                image_paths.append(image_path)
    return image_paths


def find_bounding_box(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    
    # 初始化 bounding box 的边界坐标
    top = float('inf')
    bottom = 0
    left = float('inf')
    right = 0
    
    # 遍历矩阵，更新 bounding box 的边界坐标
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j]:
                top = min(top, i)
                bottom = max(bottom, i)
                left = min(left, j)
                right = max(right, j)
    
    # 返回 bounding box 的坐标
    return top, bottom, left, right


def cut_image(user_image, select_mask):
    top, bottom, left, right = find_bounding_box(select_mask)
    cut_user_image = user_image[:,:,:3].copy()
    cut_user_image = cut_user_image[top:bottom, left:right]
    true_area_mask = select_mask[top:bottom, left:right]
    cut_user_image[~true_area_mask,:]= [255, 255, 255]
    return cut_user_image

def save_image_with_overlay(image_path, mask, output_path, mask_color=(255, 0, 0), alpha=0.5, max_dimension=0):
    """
    将图像与mask叠加显示并保存，并根据最大长宽缩放图像。

    :param image_path: 原始图像的路径
    :param mask: 布尔mask数组
    :param output_path: 保存输出图像的路径
    :param mask_color: mask叠加颜色 (默认是红色)
    :param alpha: 叠加透明度 (0.0 完全透明到 1.0 完全不透明)
    :param max_dimension: 图像最大长宽 (大于0时生效)
    """
    # 打开原始图像
    image = Image.open(image_path).convert("RGBA")
    
    cut_user_image = cut_image(user_image=np.asarray(image), select_mask=mask)

    # 如果最大长宽大于0，进行缩放
    if max_dimension > 0:
        ratio = min(max_dimension / image.width, max_dimension / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.LANCZOS)  # Use LANCZOS for high-quality downsampling
        
        # Resize the mask to the new size
        mask_resized = np.zeros((new_size[1], new_size[0]), dtype=bool)
        mask_ratio = (new_size[1] / mask.shape[0], new_size[0] / mask.shape[1])
        resized_mask_shape = (int(mask.shape[0] * mask_ratio[0]), int(mask.shape[1] * mask_ratio[1]))
        mask_resized[:resized_mask_shape[0], :resized_mask_shape[1]] = np.array(Image.fromarray(mask.astype('uint8') * 255).resize((resized_mask_shape[1], resized_mask_shape[0]), Image.NEAREST)) > 0

        # 创建一个正方形的白色背景
        square_image = Image.new("RGBA", (max_dimension, max_dimension), (255, 255, 255, 255))
        
        # 将缩放后的图像粘贴到正方形背景中央
        offset = ((max_dimension - new_size[0]) // 2, (max_dimension - new_size[1]) // 2)
        square_image.paste(image, offset)
        
        image = square_image
        
        # Create a new square mask
        square_mask = np.zeros((max_dimension, max_dimension), dtype=bool)
        square_mask[offset[1]:offset[1]+new_size[1], offset[0]:offset[0]+new_size[0]] = mask_resized
        
    else:
        square_mask = mask
    
    # 创建一个新的图像，用于叠加mask
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_data = overlay.load()
    
    # 将mask颜色应用到叠加图像中
    for y in range(image.height):
        for x in range(image.width):
            if square_mask[y, x]:
                overlay_data[x, y] = mask_color + (int(255 * alpha),)

    # 将叠加图像应用到原始图像
    combined = Image.alpha_composite(image, overlay)
    
    # 转换为RGB格式并保存为JPEG
    combined = combined.convert("RGB")
    combined.save(output_path, format='JPEG')

    return np.array(combined), cut_user_image
    










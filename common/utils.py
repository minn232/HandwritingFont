# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import os
import glob
import imageio
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def draw_example(char, font, canvas_size):
    """
    주어진 문자와 폰트를 사용해 캔버스에 이미지를 그립니다.
    """
    try:
        # 흰 배경으로 새 이미지 생성
        image = Image.new("L", (canvas_size, canvas_size), "white")
        draw = ImageDraw.Draw(image)

        # 텍스트 크기 계산
        bbox = draw.textbbox((0, 0), char, font=font)  # 텍스트 바운딩 박스
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]  # 너비와 높이 계산

        # 텍스트 위치 조정
        x, y = (canvas_size - w) // 2, (canvas_size - h) // 2
        draw.text((x, y), char, font=font, fill="black")
        return np.array(image)  # NumPy 배열로 반환
    except Exception as e:
        print(f"Error generating image for character {char}: {e}")
        return None


def pad_seq(seq, batch_size):
    # pad the sequence to be the multiples of batch_size
    seq_len = len(seq)
    if seq_len % batch_size == 0:
        return seq
    padded = batch_size - (seq_len % batch_size)
    seq.extend(seq[:padded])
    return seq


def bytes_to_file(bytes_img):
    return BytesIO(bytes_img)


def normalize_image(img):
    """
    Make image zero centered and in between (-1, 1)
    """
    normalized = (img / 127.5) - 1.
    return normalized


def denorm_image(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def read_split_image(img_path):
    """
    이미지를 읽어 절반으로 나누어 반환합니다.
    img_A: 왼쪽 절반 (target)
    img_B: 오른쪽 절반 (source)
    """
    try:
        # 이미지 열기
        img = Image.open(img_path).convert("RGB")  # RGB 형식으로 변환
        mat = np.array(img).astype(np.float32)  # NumPy 배열로 변환
        
        # 이미지 절반으로 나누기
        side = mat.shape[1] // 2
        assert side * 2 == mat.shape[1], "이미지 너비는 2의 배수여야 합니다."
        img_A = mat[:, :side, :]  # 왼쪽 절반
        img_B = mat[:, side:, :]  # 오른쪽 절반

        return img_A, img_B
    except Exception as e:
        print(f"[Error] Failed to read and split image: {e}")
        return None, None


def shift_and_resize_image(img, shift_x, shift_y, nw, nh):
    try:
        img_pil = Image.fromarray(img)  # NumPy 배열을 PIL 이미지로 변환
        resized_img = img_pil.resize((nw, nh), Image.LANCZOS)  # Pillow의 resize 함수 사용
        resized_array = np.array(resized_img)  # PIL 이미지를 다시 NumPy 배열로 변환
        cropped_img = resized_array[shift_x:shift_x + img.shape[0], shift_y:shift_y + img.shape[1]]
        return cropped_img
    except Exception as e:
        print(f"Error in shift_and_resize_image: {e}")
        return img

def scale_back(images):
    return (images + 1.) / 2.


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img

def save_concat_images(imgs, img_path):
    """
    여러 이미지를 가로로 결합하여 저장합니다.
    """
    try:
        # 이미지를 NumPy 배열에서 Pillow 이미지로 변환
        concated = np.concatenate(imgs, axis=1)  # 이미지를 가로로 결합
        result_img = Image.fromarray(concated.astype(np.uint8))  # Pillow 이미지 생성

        # 결과 이미지 저장
        result_img.save(img_path)
        print(f"[INFO] Saved concatenated image to {img_path}")
    except Exception as e:
        print(f"[Error] Failed to save concatenated images: {e}")


def save_gif(gif_path, image_path, file_name):
    filenames = sorted(glob.glob(os.path.join(image_path, "*.png")))
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(os.path.join(gif_path, file_name), images)


def show_comparison(font_num, real_targets, fake_targets, show_num=8):
    plt.figure(figsize=(14, show_num//2+1))
    for idx in range(show_num):
        plt.subplot(show_num//4, 8, 2*idx+1)
        plt.imshow(real_targets[font_num][idx].reshape(128, 128), cmap='gray')
        plt.title("Real [%d]" % font_num)
        plt.axis('off')

        plt.subplot(show_num//4, 8, 2*idx+2)
        plt.imshow(fake_targets[font_num][idx].reshape(128, 128), cmap='gray')
        plt.title("Fake [%d]" % font_num)
        plt.axis('off')
    plt.show()
    
    
def tight_crop_image(img, verbose=False, resize_fix=False):
    """
    이미지를 타이트하게 크롭합니다.
    """
    full_white = 255  # 흰색 픽셀 값
    # tolerance = 5  # 허용 오차 (흰색에서 약간 벗어난 값까지 감지)
    # col_sum = np.where(np.sum(img, axis=0) < (full_white - tolerance) * img.shape[0])
    # row_sum = np.where(np.sum(img, axis=1) < (full_white - tolerance) * img.shape[1])
    
    col_sum = np.where(np.sum(img, axis=0) < full_white * img.shape[0])
    row_sum = np.where(np.sum(img, axis=1) < full_white * img.shape[1])

    if col_sum[0].size > 0 and row_sum[0].size > 0:
        y1, y2 = row_sum[0][0], row_sum[0][-1] + 1  # y2를 +1하여 포함
        x1, x2 = col_sum[0][0], col_sum[0][-1] + 1  # x2를 +1하여 포함
        cropped_image = img[y1:y2, x1:x2]
    else:
        cropped_image = img  # 감지 실패 시 원본 반환

    if verbose:
        print(f"Cropping bounds: ({y1}, {y2}), ({x1}, {x2}) -> Cropped size: {cropped_image.shape}")

    return cropped_image


def add_padding(img, image_size=128, verbose=False, pad_value=255):
    """
    이미지에 패딩을 추가하여 정사각형으로 만듭니다.
    """
    height, width = img.shape
    pad_value = pad_value or img[0][0]
    pad_x_width = (image_size - width) // 2
    pad_y_height = (image_size - height) // 2

    pad_x = np.full((height, pad_x_width), pad_value, dtype=np.float32)
    pad_y = np.full((pad_y_height, width + 2 * pad_x_width), pad_value, dtype=np.float32)

    img = np.concatenate((pad_x, img, pad_x), axis=1)
    img = np.concatenate((pad_y, img, pad_y), axis=0)

    if verbose:
        print(f"Final image size: {img.shape}")

    return img


def centering_image(img, image_size=128, verbose=False, resize_fix=False, pad_value=None):
    if not pad_value:
        pad_value = img[0][0]
    cropped_image = tight_crop_image(img, verbose=verbose, resize_fix=resize_fix)
    centered_image = add_padding(cropped_image, image_size=image_size, verbose=verbose, pad_value=pad_value)
    
    return centered_image


def chars_to_ids(sentence):
    charset = []
    for i in range(0xac00,0xd7a4):
        charset.append(chr(i))

    fixed_char_ids = []
    for char in sentence:
        fixed_char_ids.append(charset.index(char))
        
    return fixed_char_ids


def round_function(i):
    if i < -0.95:
        return -1
    elif i > 0.95:
        return 1
    else:
        return i
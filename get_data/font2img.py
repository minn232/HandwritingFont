# https://github.com/kaonashi-tyc/zi2zi

# -*- coding: utf-8 -*-

import argparse
import sys
import glob
import numpy as np
import io, os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import collections
import random

SRC_PATH = './get_data/fonts/source/'
TRG_PATH = './get_data/fonts/target/'
OUTPUT_PATH = './get_data/dataset-11172/'


def draw_single_char(ch, font, canvas_size):
    image = Image.new('L', (canvas_size, canvas_size), color=255)
    drawing = ImageDraw.Draw(image)
    w, h = drawing.textsize(ch, font=font)
    drawing.text(
        ((canvas_size-w)/2, (canvas_size-h)/2),
        ch,
        fill=(0),
        font=font
    )
    flag = np.sum(np.array(image))
    
    # 해당 font에 글자가 없으면 return None
    if flag == 255 * 128 * 128:
        return None
    
    return image


def draw_example(ch, src_font, dst_font, canvas_size):
    dst_img = draw_single_char(ch, dst_font, canvas_size)
    
    # 해당 font에 글자가 없으면 return None
    if not dst_img:
        return None
    
    src_img = draw_single_char(ch, src_font, canvas_size)
    example_img = Image.new("RGB", (canvas_size * 2, canvas_size), (255, 255, 255)).convert('L')
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))   
    return example_img


def draw_handwriting(ch, src_font, canvas_size, dst_folder, label, count):
    dst_path = dst_folder + "%d_%04d" % (label, count) + ".png"
    dst_img = Image.open(dst_path)
    src_img = draw_single_char(ch, src_font, canvas_size)
    example_img = Image.new("RGB", (canvas_size * 2, canvas_size), (255, 255, 255)).convert('L')
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))
    return example_img

if __name__ == "__main__":
    CANVAS_SIZE = 128  # 캔버스 크기 설정
    SRC_FONT_PATH = os.path.join(SRC_PATH, 'source_font.ttf')  # 예: 단일 소스 폰트
    # 한글 유니코드 범위
    HANGUL_START = 0xAC00
    HANGUL_END = 0xD7A3
    HANGUL_COUNT = 3000  # 글자 수

    # 단일 소스 폰트 불러오기
    src_font = ImageFont.truetype(SRC_FONT_PATH, size=CANVAS_SIZE)

    # 대상 폰트 디렉토리 순회
    target_fonts = [f for f in os.listdir(TRG_PATH) if f.endswith('.ttf')]

    count = 0
    for target_font_file in target_fonts:
        try:
            target_font = ImageFont.truetype(os.path.join(TRG_PATH, target_font_file), size=CANVAS_SIZE)
        
            # 한글 글자 3000자 랜덤 선택
            random_chars = random.sample(range(HANGUL_START, HANGUL_END + 1), HANGUL_COUNT)

            for char_code in random_chars:
                ch = chr(char_code)
                example_img = draw_example(ch, src_font, target_font, CANVAS_SIZE)
                if example_img:
                    output_file = os.path.join(OUTPUT_PATH, f"{target_font_file}_{count:04d}.png")
                    example_img.save(output_file)
                    count += 1

        except Exception as e:
            print(f"Error processing font {target_font_file}: {e}")
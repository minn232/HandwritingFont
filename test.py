from common import utils
from get_data import font2img_copy
from PIL import ImageFont, Image
import numpy as np
import os
from common.utils import tight_crop_image, add_padding  # utils 파일에서 함수 가져오기

# 경로 설정
SRC_PATH = './get_data/fonts/source/'
TRG_PATH = './get_data/fonts/target/'
OUTPUT_PATH = './get_data/dataset-11172/'  # 원본 이미지가 저장된 경로
CROPPED_OUTPUT_PATH = './get_data/cropped/'  # 크롭 후 저장할 경로
CHARSET_FILE = './get_data/2350-common-hangul.txt'  # 문자 집합 파일 경로
os.makedirs(CROPPED_OUTPUT_PATH, exist_ok=True)  # 크롭 경로가 없으면 생성

def load_charset(file_path):
    """파일에서 문자 집합을 로드"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            charset = f.read().splitlines()
        return charset
    except Exception as e:
        print(f"Error reading charset file {file_path}: {e}")
        return []


if __name__ == "__main__":
    CANVAS_SIZE = 128  # 캔버스 크기 설정
    SRC_FONT_PATH = os.path.join(SRC_PATH, 'source_font.ttf')  # 소스 폰트 경로

    # 문자 집합 로드
    charset = load_charset(CHARSET_FILE)
    if not charset:
        print("Character set is empty. Exiting.")
        exit(1)

    # 소스 폰트 불러오기
    try:
        src_font = ImageFont.truetype(SRC_FONT_PATH, size=CANVAS_SIZE)
    except Exception as e:
        print(f"Error loading source font: {e}")
        exit(1)

    # 대상 폰트 디렉토리 순회
    target_fonts = [f for f in os.listdir(TRG_PATH) if f.endswith('.ttf')]

    count = 0
    for target_font_file in target_fonts:
        try:
            target_font = ImageFont.truetype(os.path.join(TRG_PATH, target_font_file), size=CANVAS_SIZE)

            for ch in charset:
                # 문자 이미지 생성
                src_img = font2img_copy.draw_single_char(ch, src_font, CANVAS_SIZE)
                target_img = font2img_copy.draw_single_char(ch, target_font, CANVAS_SIZE)

                if src_img is None or target_img is None:
                    print(f"Skipping character {ch}: Image is None")
                    continue

                # numpy 배열로 전환
                src_img_array = np.array(src_img)
                target_img_array = np.array(target_img)
                
               # 1. Crop 단계
                src_cropped_img = tight_crop_image(src_img_array, verbose=True)
                target_cropped_img = tight_crop_image(target_img_array, verbose=True)

                # 크롭된 이미지 시각화 및 저장
                Image.fromarray(src_cropped_img.astype(np.uint8)).save(f"src_cropped_{count}.png")
                Image.fromarray(target_cropped_img.astype(np.uint8)).save(f"target_cropped_{count}.png")

                # 2. Resize 및 Padding 단계
                src_processed_img = add_padding(src_cropped_img, image_size=CANVAS_SIZE, verbose=True)
                target_processed_img = add_padding(target_cropped_img, image_size=CANVAS_SIZE, verbose=True)

                # 패딩된 이미지 시각화 및 저장
                Image.fromarray(src_processed_img.astype(np.uint8)).save(f"src_padded_{count}.png")
                Image.fromarray(target_processed_img.astype(np.uint8)).save(f"target_padded_{count}.png")

                # 결합 이미지 생성
                example_img = Image.new("RGB", (CANVAS_SIZE * 2, CANVAS_SIZE), (255, 255, 255)).convert('L')
                example_img.paste(Image.fromarray(target_processed_img.astype(np.uint8)), (0, 0))
                example_img.paste(Image.fromarray(src_processed_img.astype(np.uint8)), (CANVAS_SIZE, 0))

                # 저장
                output_file = os.path.join(CROPPED_OUTPUT_PATH, f"{target_font_file}_{count:04d}.png")
                example_img.save(output_file)
                print(f"Processed and saved: {output_file}")
                count += 1

        except Exception as e:
            print(f"Error processing font {target_font_file}: {e}")

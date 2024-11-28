from get_data import font2img
from PIL import ImageFont, Image
import numpy as np
import os
from common.utils import tight_crop_image, add_padding  # utils 파일에서 함수 가져오기
import random

# 경로 설정
SRC_PATH = './get_data/fonts/source/'
TRG_PATH = './get_data/fonts/target/'
OUTPUT_PATH = './get_data/cropped_dataset/'  # 원본 이미지가 저장된 경로


# TODO) 한글 글자 중 랜덤으로 3000개 뽑아서 charset 만들기
CHARSET_FILE = './get_data/2350-common-hangul.txt'  # 문자 집합 파일 경로
def make_charset():

    # Define the range of Unicode for Hangul syllables
    hangul_start = 0xAC00
    hangul_end = 0xD7A3

    # Generate all possible Hangul syllables
    all_hangul_chars = [chr(i) for i in range(hangul_start, hangul_end + 1)]

    # Randomly select 3000 unique Hangul characters
    return random.sample(all_hangul_chars, 3000)

def is_character_supported(font, character):
    try:
        # getmask()가 빈 마스크를 반환하는지 확인
        mask = font.getmask(character)
        return mask.size != (0, 0)  # 비어 있지 않으면 지원됨
    except Exception:
        return False


if __name__ == "__main__":
    CANVAS_SIZE = 128   # 캔버스 크기 설정
    FONT_SIZE = 100      # 폰트 크기 설정
    SRC_FONT_PATH = os.path.join(SRC_PATH, 'source_font.ttf')  # 소스 폰트 경로

    # 문자 집합 로드
    charset = make_charset()
    if not charset:
        print("Character set is empty. Exiting.")
        exit(1)

    # 소스 폰트 불러오기
    try:
        src_font = ImageFont.truetype(SRC_FONT_PATH, size=FONT_SIZE)
    except Exception as e:
        print(f"Error loading source font: {e}")
        exit(1)

    # 대상 폰트 디렉토리 순회
    target_fonts = [f for f in os.listdir(TRG_PATH) if f.endswith('.ttf')]

    count = 0
    for target_font_file in target_fonts:
        try:
            target_font = ImageFont.truetype(os.path.join(TRG_PATH, target_font_file), size=FONT_SIZE)

            for ch in charset:
                # 문자 이미지 생성
                src_img = font2img.draw_single_char(ch, src_font, CANVAS_SIZE)
                target_img = font2img.draw_single_char(ch, target_font, CANVAS_SIZE)

                if src_img is None or target_img is None:
                    print(f"Skipping character {ch}: Image is None")
                    continue
                
                if not is_character_supported(src_font, ch) or not is_character_supported(target_font, ch):
                    print(f"Skipping character {ch}: Not supported by one of the fonts")
                    continue

                # numpy 배열로 전환
                src_img_array = np.array(src_img)
                target_img_array = np.array(target_img)
                
               # 1. Crop 단계
                src_cropped_img = tight_crop_image(src_img_array, verbose=False)
                target_cropped_img = tight_crop_image(target_img_array, verbose=False)

                # 2. Resize 및 Padding 단계
                src_processed_img = add_padding(src_cropped_img, image_size=CANVAS_SIZE, verbose=False)
                target_processed_img = add_padding(target_cropped_img, image_size=CANVAS_SIZE, verbose=False)

                # 결합 이미지 생성
                example_img = Image.new("RGB", (CANVAS_SIZE * 2, CANVAS_SIZE), (255, 255, 255)).convert('L')
                example_img.paste(Image.fromarray(target_processed_img.astype(np.uint8)), (0, 0))
                example_img.paste(Image.fromarray(src_processed_img.astype(np.uint8)), (CANVAS_SIZE, 0))

                # 저장
                output_file = os.path.join(OUTPUT_PATH, f"{target_font_file}_{count:04d}.png")
                example_img.save(output_file)
                print(f"Processed and saved: {output_file}")
                count += 1

        except Exception as e:
            print(f"Error processing font {target_font_file}: {e}")

from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np


SRC_PATH = './get_data/fonts/source/'
TRG_PATH = './get_data/fonts/target/'
OUTPUT_PATH = './get_data/dataset-11172/'
CHARSET_FILE = './2350-common-hangul.txt'



def load_charset(file_path):
    """파일에서 문자 집합을 로드"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            charset = f.read().splitlines()
        return charset
    except Exception as e:
        print(f"Error reading charset file {file_path}: {e}")
        return []

def draw_single_char(ch, font, canvas_size):
    """특정 문자를 캔버스에 그립니다."""
    image = Image.new('L', (canvas_size, canvas_size), color=255)
    draw = ImageDraw.Draw(image)

    try:
        # 텍스트 바운딩 박스를 계산하여 너비와 높이 얻기
        left, top, right, bottom = draw.textbbox((0, 0), ch, font=font)
        w, h = right - left, bottom - top

        # 문자 중앙 배치
        x = (canvas_size - w) // 2
        y = (canvas_size - h) // 2
        draw.text((x, y), ch, font=font, fill=0)

        # 이미지 픽셀 합계 확인 (거의 흰색이면 제외)
        if np.sum(np.array(image)) >= 255 * canvas_size * canvas_size * 0.99:
            print(f"Character '{ch}' is not visible in font.")
            return None

        return image
    except Exception as e:
        print(f"Error drawing character '{ch}': {e}")
        return None


def process_font(font_path, canvas_size, charset):
    """폰트 파일을 처리하여 문자 이미지를 생성합니다."""
    try:
        font = ImageFont.truetype(font_path, size=canvas_size)
    except Exception as e:
        print(f"Error loading font {font_path}: {e}")
        return

    count = 0
    for ch in charset:
        img = draw_single_char(ch, font, canvas_size)
        if img:
            output_file = os.path.join(OUTPUT_PATH, f"{os.path.basename(font_path)}_{count:04d}.png")
            img.save(output_file)
            print(f"Saved: {output_file}")
            count += 1
        else:
            print(f"Skipped character: {ch}")

    print(f"Processed {count}/{len(charset)} characters for font {font_path}")


if __name__ == '__main__':
    import os
    from PIL import ImageFont
    from font2img import draw_example  # draw_example 함수 가져오기

    # 테스트 설정
    SRC_FONT_PATH = './get_data/fonts/source/source_font.ttf'  # 소스 폰트 경로
    TRG_FONT_PATH = './get_data/fonts/target/01.ttf'  # 대상 폰트 경로
    OUTPUT_PATH = './test_output/'  # 테스트 결과 저장 폴더
    os.makedirs(OUTPUT_PATH, exist_ok=True)  # 출력 폴더 생성

    CANVAS_SIZE = 128  # 캔버스 크기
    TEST_CHAR = '가'  # 테스트할 문자

    # 폰트 로드
    try:
        src_font = ImageFont.truetype(SRC_FONT_PATH, size=CANVAS_SIZE)
        trg_font = ImageFont.truetype(TRG_FONT_PATH, size=CANVAS_SIZE)
    except Exception as e:
        print(f"Error loading fonts: {e}")
        exit(1)

    # draw_example 함수 테스트
    try:
        example_image = draw_example(TEST_CHAR, src_font, trg_font, CANVAS_SIZE)

        if example_image:
            # 결과 저장
            output_file = os.path.join(OUTPUT_PATH, f"example_{TEST_CHAR}.png")
            example_image.save(output_file)
            print(f"Example image saved to {output_file}")
            example_image.show()  # 결과 이미지를 화면에 표시
        else:
            print("draw_example returned None. Possibly unsupported character or font.")
    except Exception as e:
        print(f"Error during draw_example execution: {e}")
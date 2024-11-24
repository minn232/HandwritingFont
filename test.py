from common import utils
import importlib.util
import os

# 상위 디렉토리의 파일 경로 설정
file_path = os.path.abspath(os.path.join(os.path.dirname("common"), "../utils.py"))

# 모듈 spec 생성 및 로드
spec = importlib.util.spec_from_file_location("utils", file_path)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


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
            random_chars = random.sample(range(HANGUL_START, HANGUL_END + 1), HANGUL_COUNT)

            for char_code in random_chars:
                ch = chr(char_code)
                example_img = draw_example(ch, src_font, target_font, CANVAS_SIZE)
                if example_img:
                    # numpy 배열로 변환
                    img_array = np.array(example_img)

                    # 1. Crop 단계
                    cropped_img = utils.tight_crop_image(img_array, verbose=False)

                    # 2. Resize 및 Padding 단계
                    processed_img = utils.add_padding(cropped_img, image_size=CANVAS_SIZE)

                    # PIL 이미지로 변환 후 저장
                    output_file = os.path.join(OUTPUT_PATH, f"{target_font_file}_{count:04d}.png")
                    Image.fromarray(processed_img).convert('L').save(output_file)
                    count += 1

        except Exception as e:
            print(f"Error processing font {target_font_file}: {e}")
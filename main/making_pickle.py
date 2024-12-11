import pickle
import os
import numpy as np
from PIL import Image

def save_data_to_pickle(data, file_path):
    """Pickle 데이터를 파일로 저장하는 함수"""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def process_image(image_path):
    """이미지 파일을 열어서 numpy 배열로 변환하는 함수"""
    img = Image.open(image_path).convert('L')  # 흑백으로 변환
    img_array = np.array(img)  # numpy 배열로 변환
    return img_array

def create_dataset(data_dir, train_file, val_file, split_ratio=0.8):
    """PNG 이미지를 읽어서 Pickle 파일로 저장하는 함수"""
    image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.png')]  # PNG 파일만 선택
    total_images = len(image_paths)  # 전체 이미지 개수
    
    dataset = []
    for i, img_path in enumerate(image_paths):
        # 파일 이름에서 레이블 추출 (예: '01.ttf_0000.png' -> '0000')
        try:
            font_name = os.path.basename(img_path).split("_")[0]  # 폰트 이름 (예: '01.ttf')
            charid = os.path.basename(img_path).split("_")[1].split(".")[0]  # 문자 ID 추출 (예: '0000')
            label = int(charid)  # 문자 ID를 레이블로 사용
            image_bytes = process_image(img_path)
            dataset.append((label, charid, image_bytes))
        except ValueError:
            print(f"Skipping file {img_path}: Invalid label format.")
            continue  # 레이블 추출 오류 시 건너뛰기
        
        # 진행 상황 출력 (몇 번째 이미지가 처리 중인지, 전체 대비 진행률)
        if (i + 1) % 1000 == 0:  # 1000개 처리마다 출력
            print(f"Processed {i + 1}/{total_images} images ({(i + 1) / total_images * 100:.2f}%)")

    # 데이터셋을 훈련과 검증 데이터로 나눔
    np.random.shuffle(dataset)
    split_index = int(len(dataset) * split_ratio)
    train_data = dataset[:split_index]
    val_data = dataset[split_index:]

    # Pickle 파일로 저장
    save_data_to_pickle(train_data, train_file)
    save_data_to_pickle(val_data, val_file)
    print(f"Train data saved to {train_file}")
    print(f"Val data saved to {val_file}")

# 예시 실행
data_dir = './get_data/cropped_dataset'  # PNG 파일이 저장된 디렉토리
train_file = './get_data/cropped_dataset/train.obj'  # 학습 데이터 파일
val_file = './get_data/cropped_dataset/val.obj'  # 검증 데이터 파일

create_dataset(data_dir, train_file, val_file)

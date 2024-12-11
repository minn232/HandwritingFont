import torch
from common.train import Trainer
from common.models import Encoder, Decoder, Discriminator

# GPU 설정
GPU = torch.cuda.is_available()

# 데이터 경로
data_dir = './get_data/cropped_dataset'
fixed_dir = './fixed_dir'  # 임베딩 등 고정 데이터 저장 경로
save_path = './results'  # 샘플 결과 저장 경로
to_model_path = './models'  # 학습된 모델 저장 경로

# 하이퍼파라미터
batch_size = 32
img_size = 128
fonts_num = 100  # 사용된 글꼴 수 (필요에 따라 수정)
max_epoch = 20
schedule = 5  # 학습률 스케줄링 주기

# Trainer 생성
trainer = Trainer(
    GPU=GPU,
    data_dir=data_dir,
    fixed_dir=fixed_dir,
    fonts_num=fonts_num,
    batch_size=batch_size,
    img_size=img_size,
)

# 학습 실행
trainer.train(
    max_epoch=max_epoch,
    schedule=schedule,
    save_path=save_path,
    to_model_path=to_model_path,
    lr=0.0002,
    log_step=50,
    sample_step=200,
    model_save_step=5,
)

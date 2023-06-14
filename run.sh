
pip install transformers
pip install wandb
# 허깅페이스 로그인 토큰
# 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
from huggingface_hub import login
import wandb
login(token='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')
# inisw (중요)
# 2be184e31a96c722bfebdfe35f726042eb8e526c
# 현욱
# 122f007f67ba33fd04a03ee9b81489dfb42264a6
wandb login --relogin '122f007f67ba33fd04a03ee9b81489dfb42264a6'





python main.py --batch_size 32 \
  --learning_rate 5e-5 \
  --optimizer 'adam' \
  --epochs 3 \
  --run_name 'wow_project' \
  --project_name 'final_project' \
  --entity_name 'ugiugi' \

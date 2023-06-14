
pip install transformers
pip install wandb
# 허깅페이스 로그인 토큰
# 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


huggingface-cli login --token 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
# inisw (중요)
# 2be184e31a96c722bfebdfe35f726042eb8e526c
# 현욱
# 122f007f67ba33fd04a03ee9b81489dfb42264a6
wandb login --relogin '122f007f67ba33fd04a03ee9b81489dfb42264a6'



# 주의!!! lr 은 바꾸지 마세요! .sh 로 실행하니까, 문제 생겨요. main.py 는 상관없어요!

python main.py \
    --batch_size 32 \
    --optimizer 'adam' \
    --epochs 3 \
    --run_name 'wow_project' \
    --project_name 'final_project' \
    --entity_name 'ugiugi'

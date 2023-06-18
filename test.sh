
pip install transformers
pip install wandb
# 허깅페이스 로그인 토큰
# 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


huggingface-cli login --token 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
# inisw (중요)
# 2be184e31a96c722bfebdfe35f726042eb8e526c
# 현욱
# 122f007f67ba33fd04a03ee9b81489dfb42264a6
wandb login --relogin '2be184e31a96c722bfebdfe35f726042eb8e526c'





python main.py \
    --drive \
    --data 'result_all' \
    --batch_size 256 \
    --optimizer 'sgd' \
    --learning_rate 5.999e-5 \
    --epochs 30 \
    --run_name 'inisw08-twitter-sentiment-analysis-roberta_ep30_bs256_all' \
    --project_name 'inisw08-twitter-sentiment-analysis-roberta' \
    --entity_name 'inisw08' \
    --model_fold_name 'inisw08-twitter-sentiment-analysis-roberta' \
    --test \
    --test_data 'test_final' \
    --test_model_name 'ugiugi/inisw08-twitter-sentiment-analysis-roberta-ep30-bs256-all' \
    --project_test_name 'which-SA-model-is-test'
    

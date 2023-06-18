import argparse
from train import train 
from etc.utils import setup
from test import test




if __name__ == '__main__':

  p = argparse.ArgumentParser()
  p.add_argument("--data", type=str, default='train_val', help="data name")
  p.add_argument("--drive", action='store_true', help="drive를 연결하고 싶다면, --drive 만 해주세요!")
  p.add_argument("--batch_size", type=int, default=32, help="batch_size")
  p.add_argument("--learning_rate", type=float, default=5e-5, help="learning_rate")
  p.add_argument("--optimizer", type=str, default='adam', help="['adam', 'sgd', 'rmsprop', 'adagrad'] 중 선택")
  p.add_argument("--epochs", type=int, default=3, help="epochs")
  p.add_argument("--run_name", type=str, default='wow_project', help='wandb, run_name')
  p.add_argument("--project_name", type=str, default='final_project', help='wandb, project_name')
  p.add_argument("--entity_name", type=str, default='ugiugi', help='wandb, entity_name')
  p.add_argument("--model_fold_name", type=str, default='inisw_tweet_robert', help='wandb, model_fold_name')
  p.add_argument("--test", action='store_true',  help='test 하고 싶다면, --test, 만 입력해주세요')
  p.add_argument("--test_data", type=str, default='test', help='wandb, model_fold_name')
  p.add_argument("--test_model_name", type=str, default='ugiugi/inisw08-twitter-sentiment-analysis-roberta-ep30-bs256-all', help='wandb, model_fold_name')
  p.add_argument("--project_test_name", type=str, default='test_project', help='wandb, project_name')
  
  
  
  
  
  args = p.parse_args()
  setup(args)
  if args.test:
    test(args)
  else:
    train(args)
  
  
  

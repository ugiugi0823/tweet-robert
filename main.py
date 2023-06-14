import argparse
from train import train 



if __name__ == '__main__':

  p = argparse.ArgumentParser()
  p.add_argument("--batch_size", type=int, default=32, help="batch_size")
  p.add_argument("--learning_rate", default=5e-5, help="learning_rate")
  p.add_argument("--optimizer", type=str, default='adam', help="['adam', 'sgd', 'rmsprop', 'adagrad'] 중 선택")
  p.add_argument("--epochs", type=int, default=3, help="epochs")

  args = p.parse_args()
  train(args)
  
  

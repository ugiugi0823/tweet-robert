# utils
import os
import torch
import pandas as pd

from google.colab import drive
from util.models import ret_tokenizer



# For every sentence...
def get_input_mask_label(args):
  # args
  model_fold_name = args.model_fold_name
  run_name = args.run_name
  data_name = args.data
  print(args.test)
  if args.test:
    print('test set 로드')
    df = pd.read_csv(f'./data/{args.test_data}.csv')
    
  else:
    print('train set 로드')
    df = pd.read_csv(f'./data/{data_name}.csv')
    
    
  
  sentences = df.sentence.values
  labels = df.label.astype(int).values
  # Tokenize all of the sentences and map the tokens to thier word IDs.
  input_ids = []
  attention_masks = []
  tokenizer = ret_tokenizer(args)
  if args.drive:
    tokenizer.save_pretrained(f"/content/drive/MyDrive/inisw08/tweet-sa-roberta/{model_fold_name}/{run_name}")
  else:
    tokenizer.save_pretrained(f"/content/inisw08/tweet-sa-roberta/{model_fold_name}/{run_name}")
    

  for sent in sentences:
      # `encode_plus` will:
      #   (1) Tokenize the sentence.
      #   (2) Prepend the `[CLS]` token to the start.
      #   (3) Append the `[SEP]` token to the end.
      #   (4) Map tokens to their IDs.
      #   (5) Pad or truncate the sentence to `max_length`
      #   (6) Create attention masks for [PAD] tokens.
      encoded_dict = tokenizer.encode_plus(
                          sent,                      # Sentence to encode.
                          add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                          max_length = 64,           # Pad & truncate all sentences.
                          pad_to_max_length = True,
                          return_attention_mask = True,   # Construct attn. masks.
                          return_tensors = 'pt',     # Return pytorch tensors.
                    )

      # Add the encoded sentence to the list.
      input_ids.append(encoded_dict['input_ids'])

      # And its attention mask (simply differentiates padding from non-padding).
      attention_masks.append(encoded_dict['attention_mask'])

  # Convert the lists into tensors.
  input_ids = torch.cat(input_ids, dim=0)
  attention_masks = torch.cat(attention_masks, dim=0)
  labels = torch.tensor(labels)

  # Print sentence 0, now as a list of IDs.
  print('Original: ', sentences[0])
  print('Token IDs:', input_ids[0])
  return input_ids, attention_masks, labels





# utils
# args
import torch
def ret_optim(model, args):
    print('Learning_rate = ',args.learning_rate )
    if args.optimizer == "sgd":
      optimizer = torch.optim.SGD(model.parameters(),
                      lr = args.learning_rate,
                      momentum=0.9
                      )
    elif args.optimizer == "adam":
      optimizer = torch.optim.AdamW(model.parameters(),
                      lr = args.learning_rate,
                      eps = 1e-8
                      )
    elif args.optimizer == "adagrad":
      optimizer = torch.optim.AdamW(model.parameters(),
                      lr = args.learning_rate,
                      )
    elif args.optimizer == "rmsprop":
      optimizer = torch.optim.RMSprop(model.parameters(),
                      lr = args.learning_rate,
                      )


    return optimizer
  
# utils
from transformers import get_linear_schedule_with_warmup

def ret_scheduler(train_dataloader,optimizer, args):
    epochs = args.epochs
    print('epochs =>', epochs)
    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    return scheduler
  
  
  
  # utils
import time
import numpy as np
import datetime

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



def get_arr(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    
    return pred_flat, labels_flat



def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
                                                



def setup(args):
  print('setup 구간')
  print('drive connect',args.drive)
  
  model_fold_name = args.model_fold_name
  run_name = args.run_name

  if args.drive:
    assert 'drive' in os.listdir('/content') # 당황하지 마세요! 드라이브 연결을 안해놓았어요! 코랩 드라이브 연결해주세요!
    print('구글 코랩 드라이브로 시작합니다.')
    os.makedirs("/content/drive/MyDrive/inisw08", exist_ok=True)
    os.makedirs(f"/content/drive/MyDrive/inisw08/tweet-sa-roberta/{model_fold_name}", exist_ok=True)
    os.makedirs(f"/content/drive/MyDrive/inisw08/tweet-sa-roberta/{model_fold_name}/{run_name}", exist_ok=True)
    
  else:
    print('로컬로 시작합니다!')
    os.makedirs(f"/content/inisw08", exist_ok=True)
    os.makedirs(f"/content/inisw08/tweet-sa-roberta/{model_fold_name}", exist_ok=True)
    os.makedirs(f"/content/inisw08/tweet-sa-roberta/{model_fold_name}/{run_name}", exist_ok=True)


  

                                                
  

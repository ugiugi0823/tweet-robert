# test
import time
import torch
import wandb
import random
import datetime


import numpy as np
import pandas as pd

from sklearn import metrics

from util.models import ret_model
from etc.utils import format_time, flat_accuracy, get_arr
from util.dataloader import ret_dataloader


def test(args):
  # args
  run_name = args.run_name
  project_name = args.project_test_name
  entity_name = args.entity_name
  model_fold_name = args.model_fold_name


  # wandb
  # wandb.init(project=project_name, entity=entity_name)
  # wandb.run.name = run_name




  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(device)
  model = ret_model(args)
  model.to(device)
  #wandb.init(config=sweep_defaults)
  train_dataloader,test_dataloader, test_dataloader = ret_dataloader(args)


  #print("config ",wandb.config.learning_rate, "\n",wandb.config)
  seed_val = 42

  random.seed(seed_val)
  np.random.seed(seed_val)
  torch.manual_seed(seed_val)




  # Put the model in evaluation mode--the dropout layers behave differently
  # during evaluation.
  model.eval()

  # Tracking variables
  total_eval_accuracy = 0




  y_an = []
  y_pred = []
  print('test 데이터 로더 길이', len(test_dataloader))
  for batch in test_dataloader:
    


    # Unpack this training batch from our dataloader.
    #
    # As we unpack the batch, we'll also copy each tensor to the GPU using
    # the `to` method.
    #
    # `batch` contains three pytorch tensors:
    #   [0]: input ids
    #   [1]: attention masks
    #   [2]: labels
    b_input_ids = batch[0].cuda()
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)

    # Tell pytorch not to bother with constructing the compute graph during
    # the forward pass, since this is only needed for backprop (training).
    with torch.no_grad():

        # Forward pass, calculate logit predictions.
        # token_type_ids is the same as the "segment ids", which
        # differentiates sentence 1 and 2 in 2-sentence tasks.
        # The documentation for this `model` function is here:
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        outputs = model(b_input_ids,
                              token_type_ids=None,
                              attention_mask=b_input_mask,
                              labels=b_labels)
        loss, logits = outputs['loss'], outputs['logits']

    # Accumulate the test loss.
    # total_eval_loss += loss.item()

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    # Calculate the accuracy for this batch of test sentences, and
    # accumulate it over all batches.
    
    total_eval_accuracy += flat_accuracy(logits, label_ids)
    pred_flat, labels_flat = get_arr(logits, label_ids)
    y_an.append(labels_flat.item())
    y_pred.append(pred_flat.item())
    # print(len(y_an))




  print('test 완료!!!') 
  confusion_matrix = metrics.confusion_matrix(y_an, y_pred)
  # print(confusion_matrix.shape)
  print("+++++++++++ confusion_matrix +++++++++++++")
  
  print(confusion_matrix)
  print("++++++++++++++++++++++++++++++++")
  
  # y_an_v = np.array(y_an)
  # y_an_v_t = np.transpose(y_an_v)

  # y_pred_v = np.array(y_pred)
  # y_pred_v_t = np.transpose(y_pred_v)
  # combined_array = np.vstack((y_pred_v, y_an_v_t))
  # df = pd.DataFrame(combined_array)
  # df.to_csv('shape.csv', index=False)
  
  # Report the final accuracy for this test run.
  avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
  # wandb.log({'val_accuracy':avg_val_accuracy})
  print("  Accuracy: {0:.2f}".format(avg_val_accuracy))









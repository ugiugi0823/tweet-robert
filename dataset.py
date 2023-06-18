# dataset
from torch.utils.data import TensorDataset, random_split
from etc.utils import get_input_mask_label

def get_dataset(args):
  input_ids, attention_masks, labels = get_input_mask_label(args)
  # Combine the training inputs into a TensorDataset.
  dataset = TensorDataset(input_ids, attention_masks, labels)
  test_dataset = TensorDataset(input_ids, attention_masks, labels)

  # Create a 90-10 train-validation split.

  # Calculate the number of samples to include in each set.
  train_size = int(0.9 * len(dataset))
  val_size = len(dataset) - train_size
  test_size = len(test_dataset) 

  # Divide the dataset by randomly selecting samples.
  train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

  print('{:>5,} training samples'.format(train_size))
  print('{:>5,} validation samples'.format(val_size))
  print('{:>5,} test samples'.format(test_size))

  return train_dataset, val_dataset, test_dataset

# dataloader
import wandb
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from util.dataset import get_dataset

# WANDB PARAMETER
def ret_dataloader(args):
    train_dataset, val_dataset, test_dataset = get_dataset(args)
    batch_size = args.batch_size
    
    print('batch_size = ', batch_size)
    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )

    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )
    
    test_dataloader = DataLoader(
                test_dataset, # The validation samples.
                sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
                batch_size = 1 # Evaluate with this batch size.
            )

    
    return train_dataloader,validation_dataloader, test_dataloader

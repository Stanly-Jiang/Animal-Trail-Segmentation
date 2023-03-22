import copy
import csv
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


def train_model(model, criterion, dataloaders, optimizer, metrics=None, bpath=None,
                num_epochs=None, train_transforms=None, test_transforms=None,
                save_best_model=False):
    
# def train_model(model, criterion, dataloaders, optimizer, metrics=None, bpath=None,
#                 num_epochs=None, save_best_model=False):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + \
        [f'Train_{m}' for m in metrics.keys()] + \
        [f'Test_{m}' for m in metrics.keys()]
    with open(os.path.join(bpath, 'log.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    if train_transforms is None:
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(360, expand=True, p=0.5),
            transforms.ToTensor()
        ])
    if test_transforms is None:
        test_transforms = transforms.Compose([
            transforms.ToTensor()
        ])

    train_dataset = dataloaders['Train'].dataset
    train_dataset.transform = train_transforms

    test_dataset = dataloaders['Test'].dataset
    test_dataset.transform = test_transforms

    train_dataloader = DataLoader(train_dataset, batch_size=dataloaders['Train'].batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=dataloaders['Test'].batch_size, shuffle=False)

    dataloaders = {'Train': train_dataloader, 'Test': test_dataloader}

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)
        # Initialize batch summary
        batchsummary = {a: [0] for a in fieldnames}

        for phase in ['Train', 'Test']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data.
            for sample in tqdm(iter(dataloaders[phase])):
            # for sample in iter(dataloaders[phase]):
                inputs = sample['image'].to(device)
                masks = sample['mask'].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    loss = criterion(outputs['out'], masks)
                    y_pred = outputs['out'].data.cpu().numpy().ravel()
                    y_true = masks.data.cpu().numpy().ravel()
                    for name, metric in metrics.items():
                        if name == 'f1_score':
                            # Use a classification threshold of 0.1
                            batchsummary[f'{phase}_{name}'].append(
                                metric(y_true > 0, y_pred > 0.1))
                        else:
                            batchsummary[f'{phase}_{name}'].append(
                                metric(y_true.astype('uint8'), y_pred))

                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
            batchsummary['epoch'] = epoch
            epoch_loss = loss

            # deep copy the model
            if loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())

            batchsummary[f'{phase}_loss'] = epoch_loss.item()
            print('{} Loss: {:.4f}'.format(phase, loss))

        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        print(batchsummary)
        with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
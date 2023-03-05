from modelling.model import createDeepLabv3
import modelling.datahandler as datahandler
from modelling.trainer import train_model
#import all modules from /d/Storage/Random_Stuff/Stanl/DigitizationWork/Animal-Trail-Segmentation/modelling folder including model.py and trainer.py

import torch
from sklearn.metrics import f1_score, mean_squared_error

DATA_DIR = 'D:/Storage/Random_Stuff/Stanl/DigitizationWork/Animal-Trail-Segmentation/data/train_data'
EXP_DIR = 'D:/Storage/Random_Stuff/Stanl/DigitizationWork/Animal-Trail-Segmentation/experiments'
EPOCHS = 25
BATCH_SIZE = 2
def main():
    # Create the deeplabv3 resnet101 model which is pretrained on a subset
    # of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
    model = createDeepLabv3()
    model.train()

    # Specify the loss function
    criterion = torch.nn.MSELoss(reduction='mean')
    # Specify the optimizer with a lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Specify the evaluation metrics
    metrics = {'f1_score': f1_score, 'mse': mean_squared_error}
    #metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}

    # Create the dataloader
    dataloaders = datahandler.get_dataloader_single_folder(
        DATA_DIR, batch_size=BATCH_SIZE)
    _ = train_model(model,
                    criterion,
                    dataloaders,
                    optimizer,
                    bpath=EXP_DIR,
                    metrics=metrics,
                    num_epochs=EPOCHS)

    # Save the trained model
    torch.save(model, EXP_DIR / 'weights.pt')

if __name__ == "__main__":
    main()

from modelling.model import createDeepLabv3
import modelling.datahandler as datahandler
from modelling.trainer import train_model

import torch
from sklearn.metrics import f1_score, mean_squared_error, roc_auc_score

DATA_DIR = 'data/train_data'
EXP_DIR = 'experiments'
EPOCHS = 1
BATCH_SIZE = 2

def main(): 
    # Create the deeplabv3 resnet101 model which is pretrained on a subset
    # of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
    model = createDeepLabv3()

    model.train()

    # Specify the loss function 
    # past this point is where vsc code is transferable up until dataloader
    #criterion = torch.nn.MSELoss(reduction='mean') #changed to binary cross entropy loss for old unet model comparison
    criterion = torch.nn.BCEWithLogitsLoss()

    # Specify the optimizer with a lower learning rate
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    #freeze the layers up to conv5
    for name, param in model.named_parameters():
        if 'layer3' not in name and 'layer4' not in name:
            param.requires_grad = False

    #unfreeze the rest of the layers
    for name, param in model.named_parameters():
        if 'layer3' in name or 'layer4' in name or 'classifier' in name or 'aux_classifier' in name:
            param.requires_grad = True

    # Specify the evaluation metrics
    # metrics = {'f1_score': f1_score, 'mse': mean_squared_error}
    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}

    # Create the dataloader & vsc code transferable up until this point
    dataloaders = datahandler.get_dataloader_single_folder(
        DATA_DIR, batch_size=BATCH_SIZE)
    
    _ = train_model(model,
                    criterion,
                    dataloaders,
                    optimizer,
                    bpath=EXP_DIR,
                    metrics=metrics,
                    num_epochs=EPOCHS,
                    train_transforms=None,
                    test_transforms=None)
                      # the underscore in _ = train_model() is used to ignore the output of the function 

    # Save the trained model
    torch.save(model, f"{EXP_DIR}/weights.pt")

if __name__ == "__main__":
    main()

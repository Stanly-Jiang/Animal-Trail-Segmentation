import datahandler
data_dir = "D:/Storage/Random_Stuff/Stanl/DigitizationWork/AFSCRunningModel/dev_workflow/DeepLabV3/DeepLabv3FineTuning-master/data"
dataloaders = get_dataloader_single_folder(data_dir, fraction=0.2, batch_size=4)

train_dataloader = dataloaders['Train']
test_dataloader = dataloaders['Test']

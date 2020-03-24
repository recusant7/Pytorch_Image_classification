import copy
import time
import torch
import argparse
from models.models import *
from torch.utils.data import DataLoader
from utils.load_data import ListDataSet
from utils.transforms import *
from torch import nn,optim

def train_model(model, device,dataloaders, criterion, optimizer, num_epochs, is_inception=False):
    since = time.time()

    val_acc_history = []
    train_acc_history=[]
    
    val_loss_history=[]
    train_loss_history=[]
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)

        print()
    model.history={"train_acc":train_acc_history,"train_loss":train_loss_history,"val_acc":val_acc_history,"val_loss":val_loss_history}
    time_elapsed = time.time() - since
    print('{} Training complete in {:.0f}m {:.0f}s'.format(model_name,time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model,"output/{}_{:.2f}.pkl".format(model_name,best_acc))
    return model

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="size of each image batch")
    parser.add_argument("--model_name", type=str, default="resnet50",nargs='+',help="model to train")
    parser.add_argument("--pretrained", type=bool, default=False,help="if use pretrained weights")
    parser.add_argument("--feature_extract", type=bool, default=False,help="if freeze some layers")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
    # parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    # parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    opt = parser.parse_args()
    print(opt)
    
    # load data
    data_transforms=get_transforms()
    train_data=ListDataSet("train","data/custom/train.txt",transforms=data_transforms["train"],)
    val_data=ListDataSet("val","data/custom/val.txt",data_transforms["val"])
    
    trainloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True,num_workers=opt.n_cpu)
    valloader=DataLoader(val_data, batch_size=opt.batch_size, shuffle=True,num_workers=4)
    dataloader={"train":trainloader,"val":valloader}
    his=[]
    for model_name in opt.model_name:
        # release gpu's memory
        torch.cuda.empty_cache()
        
        # initialize model
        model=None
        input_size=0
        model,input_size=initialize_model(
            model_name,
            num_classes=9,
            feature_extract=opt.feature_extract, 
            use_pretrained=opt.pretrained
        )
        
        # train model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        params_to_update = model.parameters()
        criterion = nn.CrossEntropyLoss()
        # optimizer=optim.SGD(params_to_update, lr=0.01, momentum=0.9)
        optimizer = optim.Adam(params_to_update,lr=0.001)
        model=train_model(model, device,dataloader, criterion, optimizer, num_epochs=opt.epochs, is_inception=(model_name=="inception"))
        his.append(model.history)
        
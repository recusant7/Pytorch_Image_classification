import torch
import tqdm
import argparse
import time
from utils.load_data import ListDataSet
from torch.utils.data import DataLoader
from utils.transforms import get_transforms
def evaluation(model,device,dataloader):
    c1=time.time()
    corrects=0
    model.eval()
    print("start to evalue")
    for batch_id,(inputs,labels) in enumerate(tqdm.tqdm(dataloader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs=model(inputs)
        _,preds=torch.max(outputs,1)
        corrects += torch.sum(preds == labels.data)
    acc=corrects.double() / len(dataloader.dataset)
    c2=time.time()
    print("evaluation complete in {:.0f}m {:.0f}s".format((c2-c1)//60,(c2-c1)%60))
    print(f"acc:{acc}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="size of each image batch")
    parser.add_argument("--check_point", type=str,default=None,help="position of check_point")
    parser.add_argument("--dataset", type=str,default="data/custom/test.txt", help="position of dataset")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)
    # load weights
    model=torch.load(opt.check_point)
    data_transforms=get_transforms()
    # load data
    dataset=ListDataSet("test",opt.dataset,data_transforms["test"])
    testdata=DataLoader(dataset,batch_size=opt.batch_size, shuffle=False,num_workers=opt.n_cpu)
    print(dataset.__len__())
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    evaluation(model,device,testdata)
    
    
    
    
    

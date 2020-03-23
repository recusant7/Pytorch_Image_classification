# PyTorch Image Classification

## Requirements
- Python >= 3.6
- PyTorch >= 1.0.0

## implementation of models 
- VGG-16
- Alexnet
- ResNet-18
- DenseNet
- SqueezeNet

## Getting Started
### Clone the project
```
git clone https://github.com/recusant7/Pytorch_Image_classification.git
```

### Prepare for the dataset with this directory structure
```
|-- data
    |--custom
        |-- images
            |-- 0
                |-- xxx.png
                |-- xxx.png
                ···
            |-- 1
                |-- xxx.png
                |-- xxx.png
            |-- 2
                |-- xxx.png
                |-- xxx.png
            ···
        |-- get_txt.py

```
Divide dataset into training set, val set and test set.
```
python3 get_txt.py
```
Then you can get `train.txt`, `val.txt` and `test.txt`,or produce the `.txt` in your ways.The `.txt` includes the position of images.

## Training
```
usage: train.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                [--model_name MODEL_NAME [MODEL_NAME ...]]
                [--pretrained PRETRAINED] [--feature_extract FEATURE_EXTRACT]
                [--n_cpu N_CPU] [--img_size IMG_SIZE]
```

### Example
To use Resnet-18 and VGG:
```
python3 --epochs 20 --batch_size 32 --model_name resnet18 vgg
```


### Log
```
Namespace(batch_size=32, epochs=100, feature_extract=False, img_size=224, model_name=['resnet18','vgg'], n_cpu=8, pretrained=False)
Epoch 0/99
----------
train Loss: 2.1876 Acc: 0.1495
val Loss: 2.1728 Acc: 0.1343

Epoch 1/99
----------
train Loss: 2.1625 Acc: 0.1562
val Loss: 2.0345 Acc: 0.2038

Epoch 2/99
----------
train Loss: 1.9810 Acc: 0.2344
val Loss: 1.7873 Acc: 0.3285


Epoch 99/99
----------
train Loss: 0.4022 Acc: 0.8722
val Loss: 0.4377 Acc: 0.8657

vgg Training complete in 32m 22s
Best val Acc: 0.865707
```
## Test
```
root@Ubuntu# python3 test.py --check_point output/vgg_0.87.pkl

Namespace(batch_size=32, check_point='output/vgg_0.87.pkl', dataset='data/custom/test.txt', img_size=224, n_cpu=8)
767
start to evalue
100%|██████████████████████████████████████████████████████████████████████████| 24/24 [00:03<00:00,  7.77it/s]
evaluation complete in 0m 3s
acc:0.9374185136897001
```
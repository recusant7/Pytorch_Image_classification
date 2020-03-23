from torchvision import transforms
def get_transforms():
    data_transforms={
        "train":
        transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.CenterCrop(size=224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    
        ]),
        "val":
        transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "test":
        transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    
    }
    return data_transforms
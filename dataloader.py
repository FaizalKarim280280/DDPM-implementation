from torchvision import transforms, datasets
from torch.utils.data import DataLoader

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomCrop(size=(64, 64)),
    transforms.Normalize(0.5, 0.5),
    transforms.RandomHorizontalFlip(p = 0.5)
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomCrop(size=(64, 64)),
    transforms.Normalize(0.5, 0.5),
])

params = {
    'train_transform': train_transform,
    'val_transform' : val_transform,
}

class FlowerDataset:
    def __init__(self, type, path, batch_size = 16):
        self.type = type
        self.path = path
        self.batch_size = batch_size
        self.transform = params[type + "_transform"]
        
    def get_loader(self):
        dataset = datasets.ImageFolder(root=self.path, 
                                       transform=self.transform)
        dataloader = DataLoader(dataset, 
                                batch_size=self.batch_size, 
                                shuffle=True, 
                                num_workers=20)
        
        print(f"Loaded {len(dataset)} images from {self.path}")
        return dataloader
    
import os
import torchvision
from PIL import Image

def create_folders(path):
    os.makedirs(os.path.join(path, 'results'), exist_ok=True)   
    
def save_images(images, path):
    grid = torchvision.utils.make_grid(images)
    arr = grid.permute(1, 2, 0).to('cpu').numpy()
    img_grid = Image.fromarray(arr)
    img_grid.save(path)
import torch
import torch.nn as nn
from tqdm import tqdm
from model import Model
from diffusion import Diffusion
from common import save_images
import os

class Trainer(nn.Module):
    def __init__(self, train_loader, lr, device, result_path):
        super().__init__()
        self.train_loader = train_loader
        self.device = device
        self.result_path = os.path.join(result_path, 'results')
        self.model = Model().to(self.device)
        self.diffusion = Diffusion(beta_start=1e-4,
                                   beta_end=2e-2,
                                   noise_steps=700,
                                   img_size = 64, 
                                   device = self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = lr)
        self.loss_fxn = nn.MSELoss()
        
    
    def training_step(self, x):
        time = self.diffusion.sample_timesteps(x.shape[0]).to(self.device)
        x_t, noise = self.diffusion.forward_images(x, time)
        pred_noise = self.model(x_t, time)
        loss = self.loss_fxn(pred_noise, noise)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss
    
    def go_over_batch(self, step_fxn, loader):
        loss = 0
        for x, _ in tqdm(loader):
            x = x.to(self.device)
            loss += step_fxn(x).item()
            
        return loss/len(loader)
    
    def sample_images(self, epoch):
        sample = self.diffusion.sample(self.model, n = 10)
        save_images(images=sample, 
                    path= os.path.join(self.result_path, f"{epoch}.jpg"))
    
    def train(self, epochs):
        
        for epoch in tqdm(range(epochs), colour = 'blue'):
            
            train_loss = self.go_over_batch(self.training_step, self.train_loader)
    
            self.sample_images(epoch=epoch)
            
            print("\n[Epoch: {}] Train:[loss: {:.4f}]\n".format(epoch + 1, train_loss))
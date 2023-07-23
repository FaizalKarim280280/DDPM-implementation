from dataloader import FlowerDataset
from trainer import Trainer
from common import create_folders
import argparse

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--result_path', type=str, default='/scratch/fk/')
    parser.add_argument('--dataset_path', type=str, default='/scratch/fk/flowers/')

    args = parser.parse_args()
    
    create_folders(args.result_path)
    
    train_loader = FlowerDataset(type = 'train',
                                 path = args.dataset_path,  
                                 batch_size = args.batch_size).get_loader()
    
    trainer = Trainer(
        train_loader=train_loader,
        lr = args.lr,
        device=args.device,
        result_path=args.result_path
    )
    
    trainer.train(epochs = args.epochs)
    

if __name__ == "__main__":

    main()
    
    
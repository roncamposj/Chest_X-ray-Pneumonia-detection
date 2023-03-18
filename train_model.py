import argparse
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from train_test import train_model, test, load_model_from_checkpoint
from models import ResNet18_Scratch, ResNet18
from torch.utils.tensorboard import SummaryWriter


def data_loader(batch_size):    
    '''Code taken from pytorch tutorial'''
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.4),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'chest_xray/'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val','test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                    shuffle=True, num_workers=8)
                    for x in ['train', 'val','test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names


def run_main(FLAGS):
    
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)
    
    # Initialize the model and send to device
    if FLAGS.mode == 'scratch':
        model = ResNet18_Scratch(FLAGS.dropout).to(device) 
    else:   
        model = ResNet18(FLAGS.dropout).to(device)
    
    # ======================================================================
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=0.001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    writer = SummaryWriter()

    dataloaders, dataset_sizes, _ = data_loader(FLAGS.batch_size)
        
    best_model = train_model(model, dataloaders, dataset_sizes, device, criterion, optimizer, scheduler, 
                             writer, num_epochs=FLAGS.num_epochs)
    
    time_now = datetime.now().time()
    torch.save(best_model.state_dict(), f"{time_now}-{FLAGS.mode}-model.pth")

    load_model_from_checkpoint(best_model, f"{time_now}-{FLAGS.mode}-model.pth")
    test(best_model, device, dataloaders['test'], 1, writer)
    
    writer.close()
    print("Training and evaluation finished")
    
    
    
    
if __name__ == '__main__':
    # Evaluate model performance in terms of time.
    start = time.time()
    
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('ResNet used to detect pneumonia in chest x-rays.')
    parser.add_argument('--mode',
                        type=str, default='pretrained',
                        help="Select between 'scratch' and 'pretrained'.")
    parser.add_argument('--learning_rate',
                        type=float, default= 0.001,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=40,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=16,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--dropout', type = float, default = 0.3, help= 'Dropout rate.')
    parser.add_argument('--log_dir',
                        type=str,
                        default=str,
                        help='Directory to put logging.')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    
    run_main(FLAGS)
    time_elapsed = time.time() - start
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

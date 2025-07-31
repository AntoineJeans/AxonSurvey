from ..utils.viz import display_grayscale
def random_image_display(dataloader, n_images):
    for idx, img_mask in enumerate(dataloader):
        if idx > n_images: break

        img = img_mask[0].float().to("cpu")[0][0]
        display_grayscale(img)

import matplotlib.pyplot as plt
def disp_loss(train_losses, train_dcs, val_losses, val_dcs):
    n_epochs = len(train_losses) 
    epochs_list = list(range(1,n_epochs  + 1))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1) 
    plt.plot(epochs_list, train_losses, label='Training Loss')
    plt.plot(epochs_list, val_losses, label='Validation Loss')
    plt.xticks(ticks=[1] + list(range(10, n_epochs, 10))) 
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.tight_layout()

    plt.legend()


    plt.subplot(1, 2, 2)
    plt.plot(epochs_list, train_dcs, label='Training DICE')
    plt.plot(epochs_list, val_dcs, label='Validation DICE')
    plt.xticks(ticks=[1] + list(range(10, n_epochs, 10)))  
    plt.title('DICE Coefficient over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('DICE')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

import torch
from torch.utils.data import DataLoader
from .Dataset import SimpleAxonDataset, TiledAxonDataset, AugmentedAxonDataset, GIGAAugmentedAxonDataset
from .inference import load_model
import random

from ..tracers.metrics import dice_coefficient

def random_images_inference(n_images, dataset_path, model_pth, model_type, dataset_type=AugmentedAxonDataset):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_dataset = AugmentedAxonDataset(dataset_path)
    test_dataloader = DataLoader(dataset=test_dataset,
                                num_workers=0, pin_memory=False,
                                batch_size=16,
                                shuffle=True)
    
    model = load_model(model_pth, model_type, device)
    # Iterate for the images, masks and paths
    for idx in range(n_images):

        random_index = random.randint(0, len(test_dataloader.dataset) - 1)
        random_sample = test_dataloader.dataset[random_index]
        
        # Predict the imagen with the model
        pred_mask = model(random_sample[0].to(device).unsqueeze(0)).cpu()
        
        pred_mask = pred_mask.squeeze(0).permute(1,2,0)

        # Show the images
        img = random_sample[0].cpu().detach().permute(1, 2, 0)
        pred_mask = pred_mask.cpu().detach()
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask > 0.5] = 1

        mask = random_sample[1].cpu().permute(1, 2, 0)

        print(f"DICE coefficient: {round(float(dice_coefficient(pred_mask, mask)),5)}")
        
        plt.figure(figsize=(15, 16))
        plt.subplot(131), plt.imshow(img, cmap="gray"), plt.title("original")
        plt.subplot(132), plt.imshow(pred_mask, cmap="gray"), plt.title("predicted")
        plt.subplot(133), plt.imshow(mask, cmap="gray"), plt.title("mask")
        plt.show()
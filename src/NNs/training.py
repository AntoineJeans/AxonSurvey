
from torch.utils.data import DataLoader
def make_dataloaders(train_dataset, test_dataset, batch_size, num_workers = 0):
    train_dataloader = DataLoader(dataset=train_dataset,
                              num_workers=num_workers, pin_memory=False,
                              batch_size=batch_size,
                              shuffle=True)
    val_dataloader = DataLoader(dataset=test_dataset,
                                num_workers=num_workers, pin_memory=False,
                                batch_size=batch_size,
                                shuffle=True)
    return train_dataloader, val_dataloader



import torch
from tqdm import tqdm
from torch import optim, nn
from ..tracers.metrics import dice_coefficient


from .viz import disp_loss, random_image_display
def train_and_save_Unet(train_dataset, val_dataset, n_epochs, model_type, model_path, criterion, 
                        lr = 3e-4, batch_size = 16, n_epochs_display=10, display_loss=True, make_schedular_function = None):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataloader, val_dataloader = make_dataloaders(train_dataset, val_dataset, batch_size)
    model = model_type(in_channels=1, num_classes=1).to(device)
    
    optimizer = optim.RAdam(model.parameters(), lr=lr)

    scheduler = None
    if make_schedular_function is not None:
        num_training_steps = len(train_dataloader) * n_epochs
        scheduler = make_schedular_function(
            optimizer, num_warmup_steps = num_training_steps * 0.2, num_training_steps=num_training_steps
        )

        
    torch.cuda.empty_cache()

        
    train_losses, train_dcs, val_losses, val_dcs = [], [], [], []
    for epoch in tqdm(range(n_epochs)):
        
        train_loss, train_dc = train_model(model, train_dataloader, criterion, optimizer, device, scheduler)
        train_losses.append(train_loss)
        train_dcs.append(train_dc)

        

        val_loss, val_dc = evaluate_model(model, val_dataloader, criterion, device)
        val_losses.append(val_loss)
        val_dcs.append(val_dc)
        
        if epoch % n_epochs_display == 0:
            print("-" * 30)
            print(f"Training Loss EPOCH {epoch + 1}: {train_loss:.4f}")
            print(f"Training DICE EPOCH {epoch + 1}: {train_dc:.4f}")
            print("\n")
            print(f"Validation Loss EPOCH {epoch + 1}: {val_loss:.4f}")
            print(f"Validation DICE EPOCH {epoch + 1}: {val_dc:.4f}")
            print("-" * 30)

    # Saving the model
    torch.save(model.state_dict(), model_path)
    if display_loss: disp_loss(train_losses, train_dcs, val_losses, val_dcs)

def train_model(model, train_dataloader, criterion, optimizer, device, scheduler):

    model.train()
    train_running_loss = 0
    train_running_dc = 0
    
    for idx, pair in enumerate(train_dataloader):
        img = pair[0].float().to(device)
        mask = pair[1].float().to(device)
        
        y_pred = model(img)
        y_pred.to(device)
        optimizer.zero_grad()
        dc = dice_coefficient(y_pred, mask)
        loss = criterion(y_pred, mask)
        
        train_running_loss += loss.item()
        train_running_dc += dc.item()

        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None: scheduler.step()

    return train_running_loss / (idx + 1), train_running_dc / (idx + 1)

def evaluate_model(model, val_dataloader, criterion, device):
    model.eval()
    val_running_loss = 0
    val_running_dc = 0

    with torch.no_grad():
        for idx, img_mask in enumerate(val_dataloader):
                        
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)
            y_pred = model(img)
            loss = criterion(y_pred, mask)
            dc = dice_coefficient(y_pred, mask)
            
            val_running_loss += loss.item()
            val_running_dc += dc.item()
    return val_running_loss / (idx + 1), val_running_dc / (idx + 1)
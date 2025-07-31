import torch
def load_model(path, model_type, device):
    model = model_type(in_channels=1, num_classes=1).to(device)
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    return model

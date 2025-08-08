import numpy as np
import torchvision.transforms as transforms
from torchvision import transforms
from torch.utils.data.dataset import Dataset


from ..utils.ImageLoader import ImageLoader
from ..utils.trace_manips import thicken_trace

class AxonDataset(Dataset):
    def __init__(self, dataset_path, input_size, split_step, axon_thiccness = None, normalize_images = False):   
        self.loader = ImageLoader(dataset_path, final_img_size=input_size, split_step=split_step)
        self.loader.load()
        
        self.images = self.loader.get_images()
        self.masks = self.loader.get_masks()
        
        assert all([len(im.shape) == 3 for im in self.images]), "all image in Dataset have to be 3D"
        assert all([len(im.shape) == 3 for im in self.images]), "all masks in Dataset have to be 2D"

        self.axon_thiccness = axon_thiccness
        self.normalize_images = normalize_images

        self.tensor_transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        img = self.tensor_transform(self.images[index].astype(np.float32))
        if self.normalize_images: pass # Normalize the input image luminosity and contrast and all that

        numpy_mask = self.masks[index].astype(np.float32)
        if self.axon_thiccness is not None: numpy_mask = thicken_trace(numpy_mask, self.axon_thiccness)
        mask = self.tensor_transform(numpy_mask)
        
        
        return img, mask

    def __len__(self):
        return len(self.images)
    

class SimpleAxonDataset(AxonDataset):
    def __init__(self, dataset_path, input_size = 128, *args, **kwargs):  
        super().__init__(dataset_path, input_size, None, *args, **kwargs)       
        

class TiledAxonDataset(AxonDataset):
    def __init__(self, dataset_path, input_size = 128, split_step=32, *args, **kwargs):   
        super().__init__(dataset_path, input_size, split_step, *args, **kwargs)     

    
class AugmentedAxonDataset(AxonDataset):
    def __init__(self, dataset_path, input_size = 128, split_step=32, *args, **kwargs):   
        super().__init__(dataset_path, input_size, split_step, *args, **kwargs)   

        contrast_factors = [0.8, 1.0, 1.2]
        brightness_offsets = [0]
        self.images, self.masks = self.augment_images(self.images, self.masks, brightness_offsets = brightness_offsets, contrast_factors=contrast_factors)
    
    def augment_images(self, images, masks, contrast_factors = [0.8, 1.0, 1.2], brightness_offsets=[-30, 0, 30]):
        augmented_im = []
        augmented_mas = []
        
        for img, mask in zip(images, masks):
            for k in range(4):  # 0, 90, 180, 270 degrees
                rotated = np.rot90(img, k).copy()
                rot_mask = np.rot90(mask, k)
                for contrast in contrast_factors:
                    for brightness in brightness_offsets:
                        mod_img = rotated.astype(np.uint8)
                        mod_img = mod_img * contrast + brightness
                        mod_img = np.clip(mod_img, 0, 255).astype(np.float32)
                        augmented_im.append(mod_img)
                        augmented_mas.append(rot_mask.copy())
        return augmented_im, augmented_mas
    

class GIGAAugmentedAxonDataset(AxonDataset):
    def __init__(self, dataset_path, input_size = 128, split_step=32, *args, **kwargs):   
        super().__init__(dataset_path, input_size, split_step, *args, **kwargs)   

        contrast_factors = [0.5, 0.8, 1.0, 1.2, 2.0]
        brightness_offsets = [-30, -15, 0, 15, 30]
        self.images, self.masks = self.augment_images(self.images, self.masks, brightness_offsets = brightness_offsets, contrast_factors=contrast_factors)
    
    def augment_images(self, images, masks, contrast_factors = [0.8, 1.0, 1.2], brightness_offsets=[-30, 0, 30]):
        augmented_im = []
        augmented_mas = []
        
        for img, mask in zip(images, masks):
            for k in range(4):  # 0, 90, 180, 270 degrees
                rotated = np.rot90(img, k).copy()
                rot_mask = np.rot90(mask, k)
                for contrast in contrast_factors:
                    for brightness in brightness_offsets:
                        mod_img = rotated.astype(np.uint8)
                        mod_img = mod_img * contrast + brightness
                        mod_img = np.clip(mod_img, 0, 255).astype(np.float32)
                        augmented_im.append(mod_img)
                        augmented_mas.append(rot_mask.copy())
        return augmented_im, augmented_mas
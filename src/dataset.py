import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.config import SEED, BATCH_SIZE, CHANNELS, IMAGE_SIZE

class ManufacturingImageDataset(Dataset):
    def __init__(self):
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        
        self.num_samples = 500
        
        # Standart normal dağılımda resimler üretiyoruz
        self.images = torch.randn(self.num_samples, CHANNELS, IMAGE_SIZE, IMAGE_SIZE) * 0.5
        self.labels = torch.randint(0, 2, (self.num_samples, 1)).float()
        
        for i in range(self.num_samples):
            if self.labels[i] == 1:
                # 1. RASTGELE KONUM: Leke artık hep ortada değil. 32x32'lik resmin herhangi bir yerinde çıkabilir.
                h_start = np.random.randint(2, IMAGE_SIZE - 7)
                w_start = np.random.randint(2, IMAGE_SIZE - 7)
                
                # 2. RASTGELE BOYUT: Kusurun boyutu sabit 5x5 değil, 3x3 ile 5x5 arasında değişiyor.
                h_size = np.random.randint(3, 6)
                w_size = np.random.randint(3, 6)
                
                # 3. DÜŞÜK KONTRAST (GERÇEKÇİ LEKE): Parlaklığı +5.0 gibi ekstrem bir değerden +0.8'e çektik.
                # Leke artık arka plan gürültüsünün içine kamufle oldu. CNN'in bunu bulması kolay olmayacak!
                self.images[i, :, h_start:h_start+h_size, w_start:w_start+w_size] += 0.8

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def get_data_loaders():
    dataset = ManufacturingImageDataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader
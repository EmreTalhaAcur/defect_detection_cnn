import torch

# Resim Ayarları
CHANNELS = 3       # RGB (Kırmızı, Yeşil, Mavi)
IMAGE_SIZE = 32    # 32x32 piksellik mini resimler

# Hiperparametreler
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
SEED = 42

# Donanım Seçimi (Eğer ekran kartın destekliyorsa CUDA kullanır)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
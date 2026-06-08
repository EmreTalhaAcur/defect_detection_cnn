import torch
import torch.nn as nn

class DefectCNN(nn.Module):
    def __init__(self):
        super(DefectCNN, self).__init__()
        
        # 1. EVRİŞİM KATMANI (Convolution): 3 kanallı resmi alır, 16 filtreyle tarar.
        # Girdi: (3, 32, 32) -> Çıktı: (16, 32, 32)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU() # [KAVRAM]: Aktivasyon fonksiyonu
        
        # POLING KATMANI: Resmi yarı yarıya küçültür.
        # Girdi: (16, 32, 32) -> Çıktı: (16, 16, 16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2. EVRİŞİM KATMANI: 16 kanalı alır, daha karmaşık detaylar için 32 filtreye çıkarır.
        # Girdi: (16, 16, 16) -> Çıktı: (32, 16, 16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        # İKİNCİ POOLING: Resmi bir kez daha yarıya indirir.
        # Girdi: (32, 16, 16) -> Çıktı: (32, 8, 8)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # TAM BAĞLANTILI KATMANLAR (Fully Connected / Linear)
        # Resim matrisini düz bir çizgiye (Flatten) çevirdiğimizde elimizde 32 * 8 * 8 = 2048 nöron kalır.
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.relu3 = nn.ReLU()
        
        # ÇIKIŞ KATMANI: Tek bir değer üretir (Kusurlu olma olasılığı logit değeri)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # [KAVRAM]: İleri Yayılım (Forward Pass)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x
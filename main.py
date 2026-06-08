import torch
import torch.nn as nn
import torch.optim as optim
from src.config import DEVICE, LEARNING_RATE
from src.dataset import get_data_loaders
from src.model import DefectCNN
from src.trainer import train_cnn

def main():
    # 1. Resim veri yükleyicilerini al
    data_loader = get_data_loaders()
    
    # 2. CNN Modelini ayağa kaldır
    model = DefectCNN().to(DEVICE)
    
    # 3. Loss ve Optimizer'ı tanımla [KAVRAM: Learning Rate burada verilir]
    criterion = nn.BCEWithLogitsLoss() # İkili sınıflandırma hata fonksiyonu
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. CNN Eğitimi Başlat
    train_cnn(model, data_loader, criterion, optimizer, DEVICE)
    
    # 5. Görsel Modeli Kaydet
    torch.save(model.state_dict(), "defect_cnn_model.pt")
    print("\nCNN Görüntü Modeli başarıyla 'defect_cnn_model.pt' olarak kaydedildi!")

if __name__ == "__main__":
    main()
import torch
from src.config import EPOCHS

def train_cnn(model, data_loader, criterion, optimizer, device):
    print(f"\n--- CNN Model Eğitimi Başlıyor ({device} üzerinde) ---")
    
    model.train()
    for epoch in range(1, EPOCHS + 1):
        toplam_hata = 0.0
        
        for batch_images, batch_labels in data_loader:
            # Resimleri ve etiketleri eğitim donanımına taşıyoruz
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            
            # 1. Gradyanları Sıfırla
            optimizer.zero_grad()
            
            # 2. Forward Pass (İleri Yayılım)
            predictions = model(batch_images)
            
            # 3. Loss (Hata) Hesaplama
            loss = criterion(predictions, batch_labels)
            
            # 4. [KAVRAM]: Backpropagation (Geriye Yayılım)
            loss.backward()
            
            # 5. [KAVRAM]: Optimizer Adımı (Learning Rate kadar ağırlıkları güncelle)
            optimizer.step()
            
            toplam_hata += loss.item()
            
        ortalama_hata = toplam_hata / len(data_loader)
        print(f"Epoch: {epoch:02d}/{EPOCHS} | Ortalama CNN Hatası (Loss): {ortalama_hata:.4f}")
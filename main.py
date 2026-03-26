from src.train import start_training

if __name__ == "__main__":
    try:
        print("Süreç Başlıyor: Veri yükleniyor ve model inşa ediliyor...")
        history = start_training()
        print("\nİşlem Başarıyla Tamamlandı!")
    except Exception as e:
        print(f"Bir hata oluştu: {e}")
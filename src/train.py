import os
from .data_loader import get_datasets
from .model import build_malaria_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def start_training():
    # Veriyi yükle
    data_path = "data/cell_images"
    train_ds, val_ds, aug_layer = get_datasets(data_path)

    # Modeli kur
    model = build_malaria_model(augmentation_layer=aug_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Otomatik Kayıt ve Durdurma (Callbacks)
    os.makedirs("models", exist_ok=True)
    checkpoint = ModelCheckpoint("models/best_malaria_model.keras", save_best_only=True)
    early_stop = EarlyStopping(patience=5, restore_best_weights=True)

    # Eğitim
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=[checkpoint, early_stop]
    )
    return history



import matplotlib.pyplot as plt





def save_plots(history):
    os.makedirs("reports", exist_ok=True)

    # Accuracy Grafiği
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Eğitim Başarısı')
    plt.plot(history.history['val_accuracy'], label='Doğrulama Başarısı')
    plt.title('Model Başarısı')
    plt.legend()

    # Loss Grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    plt.title('Model Kaybı')
    plt.legend()

    plt.savefig("reports/training_performance.png")
    print("Grafikler 'reports/' klasörüne kaydedildi.")


def start_training():
    train_ds, val_ds, aug_layer = get_datasets("data/cell_images")
    model = build_malaria_model(augmentation_layer=aug_layer)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # CALLBACKS: Modelin "akıllı" eğitilmesini sağlar
    callbacks = [
        ModelCheckpoint("models/best_malaria_model.keras", save_best_only=True),
        EarlyStopping(patience=7, restore_best_weights=True),
        # Başarı artmıyorsa öğrenme hızını %20'sine düşür (Daha hassas öğrenme)
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=callbacks
    )

    save_plots(history)
    return history
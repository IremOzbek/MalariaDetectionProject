import tensorflow as tf
from tensorflow.keras import layers

def get_datasets(data_path, img_size=(128, 128), batch_size=32):
    # Eğitim Seti
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    # Doğrulama Seti
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    # Veri Artırma Katmanı (Farklı veri setlerinde başarı için kritik)
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
    ])

    return train_ds, val_ds, data_augmentation
from tensorflow.keras import layers, models, regularizers


def build_malaria_model(input_shape=(128, 128, 3), augmentation_layer=None):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    if augmentation_layer:
        model.add(augmentation_layer)

    model.add(layers.Rescaling(1. / 255))

    # Özellik Çıkarımı: Daha derin ve normalize edilmiş katmanlar
    filters_list = [32, 64, 128, 256]
    for filters in filters_list:
        model.add(layers.Conv2D(filters, (3, 3), padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))

    # Global pooling, klasik Flatten'a göre farklı veri setlerinde daha iyi genelleme yapar
    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model
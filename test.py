import tensorflow as tf
import sys

print(f"Python Sürümü: {sys.version}")
print(f"TensorFlow Sürümü: {tf.__version__}")
print("GPU Mevcut mu: ", "Evet" if tf.config.list_physical_devices('GPU') else "Hayır (CPU kullanılıyor)")
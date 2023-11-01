# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = tf.keras.models.load_model('/content/drive/MyDrive/inteligencia_artificial/Banana_Ripening_Process/model2.h5')  # Asegúrate de reemplazar 'ruta/al/modelo/modelo_entrenado.keras' con la ruta real de tu modelo

# Image path
img_path = '/content/drive/MyDrive/inteligencia_artificial/Banana_Ripening_Process/classes/ freshunripe/musa-acuminata-banana-ae901e71-4037-11ec-89ee-94b86d66fd1d_jpg.rf.d49671932bbf1c13ec1552bfb7f163d9.jpg'  # Asegúrate de reemplazar 'ruta/a/la/imagen/de/prueba.jpg' con la ruta real de tu imagen

# Load and prepare the image for inference
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.  # Normaliza la imagen

# Performs the prediction
predictions = model.predict(img_array)

# Define a class dictionary
class_dict = {0: 'freshripe', 1: 'freshunripe', 2: 'overripe', 3: 'ripe', 4: 'rotten'}  # Actualiza con tus propias clases

# Obtain the index of the class with the highest probability
predicted_class_index = np.argmax(predictions)

# Get the predicted class
predicted_class = class_dict[predicted_class_index]

print("La clase con la mayor probabilidad es:", predicted_class)

print(predictions)
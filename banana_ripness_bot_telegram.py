#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 14:02:42 2023

@author: maitrestick
"""

import os
import telebot
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

  

token = "your token"

BOT_TOKEN = os.environ.get(token)

bot = telebot.TeleBot(token)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    #print(message)
    chat_id = message.chat.id
    #print(chat_id)
    name = message.chat.first_name
    
    bienvenida = "Hey " + name + " send a banana picture to predict"
    bot.send_message(chat_id, bienvenida)

@bot.message_handler(content_types=['photo'])
def photo(message):
    #print ('message.photo =', message.photo)
    fileID = message.photo[-1].file_id
    #print ('fileID =', fileID)
    file_info = bot.get_file(fileID)
    #print ('file.file_path =', file_info.file_path)
    downloaded_file = bot.download_file(file_info.file_path)
    
    model = tf.keras.models.load_model('/your/path/BotBananaPrediction/model2.h5')  
    
    with open("image.jpg", 'wb') as new_file:
        new_file.write(downloaded_file)
    
    imgPath = '/your/path/BotBananaPrediction/image.jpg'
    img = image.load_img(imgPath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.
    
    predictions = model.predict(img_array)
    class_dict = {0: 'freshripe', 1: 'freshunripe', 2: 'overripe', 3: 'ripe', 4: 'rotten'}
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_dict[predicted_class_index]
    
    chat_id = message.chat.id
    name = message.chat.first_name
    nombreClase = "Hey " + name + " the banana class is " + predicted_class
    bot.send_message(chat_id, nombreClase)

@bot.message_handler(func=lambda msg: True)
def echo_all(message):
    bot.reply_to(message, message.text)
    

bot.infinity_polling()
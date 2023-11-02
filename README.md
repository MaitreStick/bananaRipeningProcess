
# Banana Ripening Process

<a href="[https://universe.roboflow.com/april-public-yibrz/never-gonna](https://universe.roboflow.com/fruit-ripening/banana-ripening-process)">
    <img src="https://app.roboflow.com/images/download-dataset-badge.svg"></img>
</a>

<a href="[https://universe.roboflow.com/april-public-yibrz/never-gonna/model/](https://universe.roboflow.com/fruit-ripening/banana-ripening-process/model/2)">
    <img src="https://app.roboflow.com/images/try-model-badge.svg"></img>
</a>

This project aims to classify banana images according to their maturity using a deep learning model. It involves data preprocessing, model building using TensorFlow and Keras, class balancing, hyperparameter tuning and saving of the trained model. 

## Dataset
For this banana ripening process classification project, I used a dataset provided by Roboflow. The dataset includes images of bananas at various stages of ripeness, labeled with categories such as freshripe, freshunripe, overripe, ripe, rotten, and unripe. You can find the original dataset https://universe.roboflow.com/fruit-ripening/banana-ripening-process.

P.S.: You can use any dataset of 2 classes and up, you only have to refactor the code up to a certain point and then the process is the same in this case. 

## Installation
To run this project, make sure you have the following dependencies installed:

- Python 3
- TensorFlow
- Keras
- pandas
- numpy
- PIL
- matplotlib

P.S.: It can also be executed in Google Colab with `BananaRipnessProcess.ipynb` which is available in the repository.

## Clone the repository

```bash
  git clone https://github.com/MaitreStick/bananaRipeningProcess
```

## Usage
Follow these steps to use the project:

- Set up the required libraries and dependencies.
- Run the data preprocessing script to clean and organize the dataset.
- Build the deep learning model using TensorFlow and Keras.
- Train the model using the prepared dataset.
- Evaluate the model performance and make necessary adjustments.
- Save the trained model for future use.

## Results
Since I used only 5 classes of the 6 that the dataset offers and I balanced the classes so that they were all with the same number of elements, the percentage of accuracy was 62.2%, however in roboflow there are some examples where more than 90% of accuracy is obtained, this is only another way to do it. The trained model is available in the repository for reference.

P.S.: I should also say that the training took us 8 hours in total, as the best hyperparameter to do the training with was first searched for.

## Performing Model Inference

Now that the model is trained and saved, we can perform model inference, so we can test that it works; to run that inference project, make sure you have the following dependencies installed.

- Python 3
- tensorFlow
- tensorflow.keras.preprocessing
- numpy

## Usage
Follow these steps to use the project:

- Set up the required libraries and dependencies.
- Set an image path.
- Load and prepare the image for inference.
- Performs the prediction.
- Get the predicted class.

<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> d75449c (Update Readme andbot process)
## Implementation

For the implementation of the model with a client that can be used by any user, I used the telegram api, BotFather.

P.S.: You can use a lot of languages with BotFather's API, you just need to search about it and probably you will find a very explained tutorial with the language you need.

## Telegram Bot API - BotFather

First, you will need a token in order to configure your own bot, so; how can you get the token? you can follow these steps:

- Search for @botfather in Telegram.
- Start a conversation with BotFather by clicking on the Start button.
- Type /newbot, and follow the prompts to set up a new bot. The BotFather will give you a token that you will use to authenticate your bot and grant it access to the Telegram API.

Note: Make sure you store the token securely. Anyone with your token access can easily manipulate your bot.

P.S.: You can find all these steps at https://www.freecodecamp.org/news/how-to-create-a-telegram-bot-using-python/

## Performing Model with BotFather

Having the code with which to perform the model inference and the token together with the bot working, what we have to do is to put both parts together and test, the code for reference is `banana_ripness_bot_telegram.py` 

<<<<<<< HEAD
>>>>>>> ecab5e5 (client)
=======
>>>>>>> d75449c (Update Readme andbot process)
## Feedback

If you have any feedback, please reach me out at mastijaci99@gmail.com



import os
from pyexpat import features
import random
import shutil
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
import speech_recognition as sr
import pygame
import pygame._sdl2.audio as sdl2_audio
from time import sleep
from gtts import gTTS
import numpy as np
import pandas as pd
import json
import nltk
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import string
import joblib
from keras.models import load_model
from sklearn.model_selection import cross_val_score, train_test_split
from keras.models import Sequential


keras = tf.keras# не се импортваха по нормалния начин
Tokenizer = tf.keras.preprocessing.text.Tokenizer
Input = tf.keras.layers.Input
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Embedding = tf.keras.layers.Embedding
GlobalMaxPooling1D = tf.keras.layers.GlobalMaxPooling1D
Flatten = tf.keras.layers.Flatten
Model = tf.keras.models.Model
pad_sequences = keras.preprocessing.sequence.pad_sequences
count = 0

def get_devices(capture_devices: bool = False) -> tuple[str, ...]:
    init_by_me = not pygame.mixer.get_init()
    if init_by_me:
        pygame.mixer.init()
    devices = tuple(sdl2_audio.get_audio_device_names(capture_devices))
    if init_by_me:
        pygame.mixer.quit()
    return devices[2]

def play(text):

    global count
    device = "CABLE Input (VB-Audio Virtual Cable)"
    if device is None:
        devices = get_devices()
        if not devices:
            raise RuntimeError("No device!")
        device = devices[0]
    
    try:
        if text.strip():  
            tts = gTTS(text, lang='en')
            tts.save(f'audio/speech{count}.mp3')
        else:
            print("Error: No text provided to speak")
    except AssertionError as e:
        print("Assertion Error:", e)
    
    print("Play: {}\r\nDevice: {}".format(f'audio/speech{count}.mp3', device))
    pygame.mixer.init()
    pygame.mixer.music.load(f'audio/speech{count}.mp3')
    pygame.mixer.music.play()
    count += 1

def history(question, response):
    with open('history.txt', 'a') as file:
        file.write(f"Question: {question}\n")
        file.write(f"Response: {response}\n")

def recognize_speech():
    recognizer = sr.Recognizer()


    with sr.Microphone() as source:
        print("Say something:")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        query = recognizer.recognize_google(audio)
        print("You said:", query)
        return query.lower()
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand that.")
        return "Sorry, I couldn't understand that."
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return ""
    

def train_model(model):
    train = model.fit(x_train, y_train, epochs=300)

    
    plt.plot(train.history['accuracy'], label='training set accuracy')
    plt.plot(train.history['loss'], label='training set loss')
    plt.legend()
    plt.show()


def create_model():
    vocabulary, output_length = define_vocabulary()
    
    i = Input(shape=(input_shape,))
    x = Embedding(vocabulary+1, 1000)(i)
    x = LSTM(30, return_sequences=True)(x)
    x = Flatten()(x)
    x = Dense(output_length, activation="softmax")(x)
    model = Model(i, x)
    return model


def compile_model(model):
    model.compile(loss="sparse_categorical_crossentropy",optimizer='adam',metrics=['accuracy'])

def define_vocabulary():
    with open('data.json') as content:
        data1 = json.load(content)

    tags = []
    inputs = []
    global responses
    responses={}
    for intent in data1['intents']:
        responses[intent['tag']]=intent['responses']
        for lines in intent['input']:
            inputs.append(lines)
            tags.append(intent['tag'])

    data = pd.DataFrame({"inputs":inputs,
                        "tags":tags})

    data = data.sample(frac=1)

    data['inputs'] = data['inputs'].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
    data['inputs'] = data['inputs'].apply(lambda wrd: ''.join(wrd))

    global tokenizer
    tokenizer = Tokenizer(num_words=2000)
    tokenizer.fit_on_texts(data['inputs'])
    train = tokenizer.texts_to_sequences(data['inputs'])

    global x_train
    global y_train

    x_train = pad_sequences(train)

    global le
    le = LabelEncoder()
    y_train = le.fit_transform(data['tags'])

    global input_shape
    input_shape = x_train.shape[1]
    print(input_shape)

    vocabulary = len(tokenizer.word_index)
    print("number of unique words : ",vocabulary)
    output_length = le.classes_.shape[0]
    print("output length: ",output_length)
    return vocabulary, output_length


answer = input("what do you want to do, train or chat:")
if answer == "train":
    define_vocabulary()
    model = create_model()
    compile_model(model)
    train_model(model)
    model.save('model3.keras')
    
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

elif answer == "chat":
    model = load_model('model3.keras')
    define_vocabulary()

    while True:

        texts_p = []
        question = recognize_speech()
        question_text = question

        question = [letters.lower() for letters in question if letters not in string.punctuation]
        question = ''.join(question)
        texts_p.append(question)
        question = tokenizer.texts_to_sequences(texts_p)
        question = np.array(question).reshape(-1)
        question = pad_sequences([question],input_shape)

       
        output = model.predict(question)
        output = output.argmax()

        response_tag = le.inverse_transform([output])[0]
        response = random.choice(responses[response_tag])
        play(response)
        history(question_text, response)
        print("Tanpopo : ",response)
        print(response_tag)
        if response_tag == "goodbye":
            break

elif answer == "test":

    model = load_model('model3.keras')
    define_vocabulary()

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier()

    scores = cross_val_score(clf, x_train, y_train, cv=5)
    print("Average Accuracy:", scores.mean())
    
else:
    print("option not found")

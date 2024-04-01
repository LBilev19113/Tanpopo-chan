from time import sleep
from gtts import gTTS
import pygame
import speech_recognition as sr

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
    # except sr.RequestError as e:
    #     print(f"Could not request results from Google Speech Recognition service; {e}")
    #     return ""


def play(file_path: str):
    device = "CABLE Input (VB-Audio Virtual Cable)"
    print("Play: {}\r\nDevice: {}".format(file_path, device))
    pygame.mixer.init(devicename=device)
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    sleep(5)
   

def convert_audio(text):
    tts = gTTS(text, lang='en', slow=False)
    tts.save("output.mp3")

convert_audio(recognize_speech())
play("output.mp3")
print("stop")
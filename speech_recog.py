import os
import time
import speech_recognition as sr
from gtts import gTTS
import pyttsx3

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        said = ""

        try:
            said = r.recognize_google(audio)
        except Exception as e:
            print(f"Exception: {str(e)}")

    return said.lower()

wake_sentence = 'hello'

def save_text(text):
    with open('textfile.txt', 'w') as file:
        file.write(text)

while True:
    text = get_audio()
    if wake_sentence in text:
        print('At your service!')
        speak("At your service!")

        text = get_audio()

        print("Stopped recording")
        speak(f'You said: {text}')
        print("Saving file...")
        save_text(text)
        break
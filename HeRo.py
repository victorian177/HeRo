import json
import pickle
import random

import pyttsx3
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer


class HeRo:
    def __init__(self) -> None:
        # Speech Processing
        self.recog = sr.Recognizer()
        for device_index, _ in enumerate(sr.Microphone.list_microphone_names()):
            self.mic = sr.Microphone(device_index=device_index)
            break
        else:
            raise Exception("Microphone not found!")
        self.engine = pyttsx3.init()

        # Model initialisation
        try:
            with open('model.pickle', 'rb') as mdl:
                self.model = pickle.load(mdl)
        except:
            raise Exception("Model file not present.")

        try:
            with open('vocab.json') as vcb:
                vocab = json.load(vcb)
        except:
            raise Exception("Vocabulary file not present.")

        self.tfidf = TfidfVectorizer(stop_words='english', vocabulary=vocab)

        # Outcomes
        try:
            with open('responses.json') as rsp:
                self.responses = json.load(rsp)

        except:
            raise Exception("Responses file not present.")

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def listen(self):
        print("Listening...")
        with self.mic as source:
            self.recog.adjust_for_ambient_noise(source)
            self.recog.pause_threshold = 2
            audio = self.recog.listen(source)

            try:
                speech = self.recog.recognize_google(audio)
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
                return None
            except sr.RequestError as e:
                print("Could not request from Google Speech Recognition service")
                return None

        print("Input taken")

        return speech.lower()

    def predict(self, speech):
        vector = self.tfidf.fit_transform([speech])
        label = self.model.predict(vector)[0]
        prob = max(self.model.predict_proba(vector)[0])

        return label, prob

    def response(self, label, prob):
        if prob < 0.2:
            return random.choice(self.responses['no answer'])
        else:
            return random.choice(self.responses[label])
        

instance = HeRo()
print(instance.predict("what is your name"))

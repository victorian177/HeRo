import pyttsx3
import speech_recognition as sr

engine = pyttsx3.init()
r = sr.Recognizer()


def speak(text):
    engine.say(text)
    engine.runAndWait()


def get_audio():
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        r.pause_threshold = 2
        audio = r.listen(source)
        said = ""

        try:
            said = r.recognize_google(audio)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            return None
        except sr.RequestError as e:
            print(
                "Could not request from Google Speech Recognition service; {0}".format(e))
            return None

    return said.lower()


def save_text(text):
    with open('textfile.txt', 'w') as file:
        file.write(text)


def save_audio(audio):
    with open('audiofile.wav', 'wb') as file:
        file.write(audio)


def speech_to_text():
    print('Listening...')
    text = get_audio()
    print("Input taken")

    return text

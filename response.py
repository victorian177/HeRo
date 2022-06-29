import json
import random

with open("intents.json") as f:
    intents = json.load(f)


def response(information):
    if information == "no answer":
        return "Sorry, I don't understand"

    elif information == "wake":
        response_choices = intents['intents'][0]["responses"]
        return random.choice(response_choices)

    elif information == "sleep":
        response_choices = intents['intents'][1]["responses"]
        return random.choice(response_choices)

    elif information == "contact":
        response_choice = intents['intents'][2]["responses"][0]
        return response_choice

    elif information == "name":
        response_choice = intents['intents'][3]["responses"][0]
        return response_choice

    elif information == "prescription":
        response_choice = intents['intents'][4]["responses"][0]
        return response_choice

import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent

model = tf.keras.models.load_model(
    BASE_DIR / "zh_chatbot.keras"
)

with open(
    BASE_DIR / "chatbot_metadata.json",
    "r",
    encoding="utf-8",
) as file:
    metadata = json.load(file)


labels = metadata["labels"]
responses = metadata["responses"]


def get_response(message: str):
    prediction = model.predict(
        np.array([message]),
        verbose=0,
    )[0]

    predicted_index = int(np.argmax(prediction))
    confidence = float(prediction[predicted_index])
    predicted_tag = labels[predicted_index]

    print(f"Intent: {predicted_tag}")
    print(f"Confidence: {confidence:.2%}")

    if confidence < 0.55:
        return (
            "I'm not confident enough to answer that. "
            "Please ask about ZH's experience, skills, "
            "education or projects."
        )

    return random.choice(responses[predicted_tag])


while True:
    user_message = input("\nYou: ").strip()

    if user_message.lower() in {"exit", "quit"}:
        break

    if not user_message:
        continue

    print("Bot:", get_response(user_message))
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


print("TensorFlow:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices("GPU"))


BASE_DIR = Path(__file__).resolve().parent

INTENTS_FILE = BASE_DIR / "intents.json"
MODEL_FILE = BASE_DIR / "zh_chatbot.keras"
METADATA_FILE = BASE_DIR / "chatbot_metadata.json"


def load_training_data():

    with open(INTENTS_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    sentences = []
    labels = []
    responses = {}

    for intent in data["intents"]:

        tag = intent["tag"]

        responses[tag] = intent["responses"]

        for pattern in intent["patterns"]:
            sentences.append(pattern)
            labels.append(tag)

    return sentences, labels, responses



def build_model(classes):

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=3000,
        output_mode="int",
        output_sequence_length=25
    )


    text_input = tf.keras.Input(
        shape=(),
        dtype=tf.string,
        name="message"
    )


    x = vectorizer(text_input)


    x = tf.keras.layers.Embedding(
        input_dim=3000,
        output_dim=64
    )(x)


    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64)
    )(x)


    x = tf.keras.layers.Dropout(0.35)(x)


    x = tf.keras.layers.Dense(
        64,
        activation="relu"
    )(x)


    x = tf.keras.layers.Dropout(0.25)(x)


    output = tf.keras.layers.Dense(
        classes,
        activation="softmax"
    )(x)


    model = tf.keras.Model(
        text_input,
        output
    )


    return model, vectorizer



def main():

    sentences, labels, responses = load_training_data()


    label_names = sorted(set(labels))


    label_to_number = {
        label:i
        for i,label in enumerate(label_names)
    }


    y = np.array(
        [
            label_to_number[x]
            for x in labels
        ],
        dtype=np.int32
    )


    # IMPORTANT FIX
    X = tf.constant(
        sentences,
        dtype=tf.string
    )


    model, vectorizer = build_model(
        len(label_names)
    )


    vectorizer.adapt(X)


    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )


    model.summary()



    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_FILE,
            monitor="loss",
            save_best_only=True,
            save_weights_only=False,
            mode="min",
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=20,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=5,
            min_lr = 0.00001
        )
    ]



    model.fit(
        X,
        y,
        epochs=300,
        batch_size=8,
        shuffle=True,
        callbacks=callbacks
    )



    model.save(
        MODEL_FILE
    )


    metadata = {
        "labels":label_names,
        "responses":responses
    }


    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False
        )


    print("Training completed")
    print(MODEL_FILE)



if __name__ == "__main__":
    main()
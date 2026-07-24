import json
import random
from pathlib import Path
import numpy as np
import tensorflow as tf


# Make training more reproducible.
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


BASE_DIR = Path(__file__).resolve().parent
INTENTS_FILE = BASE_DIR / "intents.json"
MODEL_FILE = BASE_DIR / "zh_chatbot.keras"
METADATA_FILE = BASE_DIR / "chatbot_metadata.json"


def load_training_data():
    """Load questions, intent labels and responses."""

    with open(INTENTS_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    training_sentences = []
    training_labels = []
    responses = {}

    for intent in data["intents"]:
        tag = intent["tag"]
        responses[tag] = intent["responses"]

        for pattern in intent["patterns"]:
            training_sentences.append(pattern)
            training_labels.append(tag)

    return training_sentences, training_labels, responses


def build_model(number_of_classes: int):
    """Create the TensorFlow neural network."""

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=3000,
        output_mode="int",
        output_sequence_length=25,
        standardize="lower_and_strip_punctuation",
        name="text_vectorizer",
    )

    text_input = tf.keras.Input(
        shape=(),
        dtype=tf.string,
        name="message",
    )

    token_ids = vectorizer(text_input)

    x = tf.keras.layers.Embedding(
        input_dim=3000,
        output_dim=64,
        name="word_embedding",
    )(token_ids)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64),
        name="bidirectional_lstm",
    )(x)

    x = tf.keras.layers.Dropout(0.35)(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu",
    )(x)

    x = tf.keras.layers.Dropout(0.25)(x)

    output = tf.keras.layers.Dense(
        number_of_classes,
        activation="softmax",
        name="intent_prediction",
    )(x)

    model = tf.keras.Model(
        inputs=text_input,
        outputs=output,
        name="zh_tensorflow_assistant",
    )

    return model, vectorizer


def main():
    sentences, string_labels, responses = load_training_data()

    # Give every intent a numerical class.
    label_names = sorted(set(string_labels))

    label_to_number = {
        label: index
        for index, label in enumerate(label_names)
    }

    numerical_labels = np.array(
        [label_to_number[label] for label in string_labels],
        dtype=np.int32,
    )

    sentences_array = np.array(sentences, dtype=str)

    model, vectorizer = build_model(
        number_of_classes=len(label_names)
    )

    # Learn the vocabulary from our own dataset.
    vectorizer.adapt(sentences_array)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=25,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=10,
            min_lr=0.00001,
        ),
    ]

    model.fit(
        sentences_array,
        numerical_labels,
        epochs=300,
        batch_size=8,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(MODEL_FILE)

    metadata = {
        "labels": label_names,
        "responses": responses,
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nTraining completed.")
    print(f"Model saved to: {MODEL_FILE}")
    print(f"Metadata saved to: {METADATA_FILE}")


if __name__ == "__main__":
    main()
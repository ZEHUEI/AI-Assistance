import tensorflow as tf
import json
import numpy as np

model = tf.keras.models.load_model("zh_chatbot.keras")

with open("chatbot_metadata.json") as f:
    metadata = json.load(f)

labels = metadata["labels"]

tests = [
    # Greeting
    "hello",
    "hi",
    "hey there",
    "good morning",
    "can you help me",
    "who are you",
    "how are you",
    "thank you",
    "thanks",

    # Profile
    "who is ZH?",
    "tell me about ZH",
    "introduce Lim Ze Huei",
    "what is ZH background?",
    "give me a summary of ZH",
    "what does ZH do?",

    # Skills
    "what programming languages does ZH know?",
    "does ZH know Python?",
    "does ZH know Java?",
    "what technologies can ZH use?",
    "tell me about ZH's technical skills",
    "what frameworks does ZH know?",

    # Cloud / DevOps
    "does ZH know AWS?",
    "what cloud platforms has ZH used?",
    "has ZH deployed applications?",
    "does ZH know Docker?",
    "what DevOps experience does ZH have?",

    # Ministry XR
    "what did ZH do at Ministry XR?",
    "tell me about Ministry XR",
    "where did ZH work?",
    "what was ZH's full stack role?",
    "what applications did ZH build?",

    # Tox
    "where did ZH intern?",
    "tell me about ZH internship",
    "did ZH work in Malaysia?",
    "what IT support experience does ZH have?",

    # RotorAI
    "tell me about RotorAI",
    "what is RotorAI?",
    "does ZH know YOLO?",
    "what machine learning projects has ZH done?",
    "how does RotorAI detect defects?",
    "does ZH have computer vision experience?",

    # Booking
    "tell me about the booking website",
    "what booking system did ZH build?",
    "does ZH have Firebase experience?",
    "how did ZH prevent scheduling conflicts?",
    "what is the Beyonk project?",

    # Web3
    "tell me about HRDC project",
    "what is Web3 Workforce Hub?",
    "does ZH know JWT?",
    "does ZH have security experience?",
    "how did ZH solve CORS problem?",

    # Education
    "where did ZH study?",
    "what degree does ZH have?",
    "which university did ZH attend?",
    "did ZH study computer science?",

    # Database
    "what databases does ZH know?",
    "does ZH know MySQL?",
    "does ZH know PostgreSQL?",
    "has ZH used Firestore?",
    "does ZH know Snowflake?",

    # Contact
    "how can I contact ZH?",
    "what is ZH email?",
    "I want to hire ZH",
    "can I arrange an interview?",

    # Phone
    "what is ZH phone number?",
    "can I call ZH?",

    # Salary
    "what salary does ZH want?",
    "what is ZH expected salary?",
    "is ZH available immediately?",

    # =====================
    # UNKNOWN TEST CASES
    # =====================

    "I forgot my password",
    "reset my account password",
    "what is the weather today?",
    "tell me a joke",
    "who is Elon Musk?",
    "how do I cook pasta?",
    "what is the capital of France?",
    "play music",
    "open my email",
    "what time is it?",
    "how much money do I have?",
    "transfer money to my account",
    "book a flight",
    "buy a laptop",
    "write me a poem",
    "solve this math question",
]

CONFIDENCE_THRESHOLD = 0.75

for text in tests:
    prediction = model.predict(
        tf.constant([text]),
        verbose=0
    )

    confidence = np.max(prediction)

    index = np.argmax(prediction)

    if confidence < CONFIDENCE_THRESHOLD:
        intent = "unknown"
    else:
        intent = labels[index]

    print(
        text,
        "=>",
        intent,
        confidence
    )
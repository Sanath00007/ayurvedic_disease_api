import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import os
import warnings
import json

warnings.filterwarnings("ignore", category=UserWarning)

# ======= Load Ayurvedic Remedies JSON =======
REMEDIES_URL = "https://raw.githubusercontent.com/Sanath00007/ayurveda-api/main/disease.json"
try:
    remedies_data = requests.get(REMEDIES_URL).json()
    print("✅ Ayurvedic remedies data loaded.")
except Exception as e:
    remedies_data = {}
    print(f"⚠️ Could not load remedies data: {e}")

# ======= Load TensorFlow SavedModel (for Keras 3) =======
MODEL_PATH = "ayur_disease_model_tf"
if not os.path.exists(MODEL_PATH):
    print("❌ Model folder not found!")
else:
    print(f"📂 Found model folder: {MODEL_PATH}")

try:
    model = tf.keras.layers.TFSMLayer(MODEL_PATH, call_endpoint="serving_default")
    print("✅ Model loaded using TFSMLayer.")
except Exception as e:
    model = None
    print(f"❌ Error loading model: {e}")

# ======= Load Class Names =======
with open("class_names.json", "r") as f:
    CLASS_NAMES = json.load(f)

# ======= Herb → Google Link Converter =======
def make_google_link(text):
    HERBS = [
        "neem", "Nimba", "turmeric", "Haridra",
        "coconut oil", "Narikela",
        "aloe vera", "Kumari",
        "vetiver", "Ushira",
        "sandalwood", "Chandana",
        "ash gourd", "Kushmanda",
        "manjistha", "Rubia cordifolia"
    ]

    for herb in HERBS:
        url = f"https://www.google.com/search?q={herb.replace(' ', '+')}+Ayurveda"
        text = text.replace(
            herb,
            f"[{herb}]({url})"
        )
    return text


# ======= Prediction Function =======
def predict(img: Image.Image):
    logs = []

    def log(msg):
        print(msg)
        logs.append(msg)

    try:
        if model is None:
            raise RuntimeError("Model not loaded properly!")

        log("🖼️ Received image for prediction.")
        img = img.convert("RGB").resize((224, 224))
        log("🔄 Image resized to 224x224.")

        x = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)
        log(f"📏 Preprocessed image shape: {x.shape}")

        preds = model(x)
        output_key = list(preds.keys())[0]
        preds = preds[output_key].numpy()
        log(f"📊 Raw predictions: {preds.tolist()}")

        predicted_class = CLASS_NAMES[np.argmax(preds)]
        log(f"✅ Predicted: {predicted_class}")

        data = remedies_data.get(predicted_class, None)

        if not data:
            remedy_text = "No Ayurvedic remedies found."
        else:
            remedy_text = f"🌿 **Remedy:**\n- {make_google_link(data['remedy'])}\n\n"

            if "pathya" in data:
                remedy_text += "✅ **Recommended (Pathya):**\n"
                for item in data["pathya"]:
                    remedy_text += f"- {make_google_link(item)}\n"
                remedy_text += "\n"

            if "apathya" in data:
                remedy_text += "🚫 **Avoid (Apathya):**\n"
                for item in data["apathya"]:
                    remedy_text += f"- {item}\n"
                remedy_text += "\n"

            remedy_text += f"📖 **Source:** {data['source']}"

        logs_output = "\n".join(logs)

        return (
            f"🩺 **Predicted Disease:** {predicted_class}\n\n"
            f"{remedy_text}\n\n"
            f"🧾 **Logs:**\n{logs_output}"
        )

    except Exception as e:
        logs_output = "\n".join(logs)
        return f"❌ Error: {e}\n\nLogs:\n{logs_output}"


# ======= Gradio Interface =======
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload skin image"),
    outputs="markdown",
    title="🌿 Ayurvedic Disease Detector",
    description="Upload a skin image to detect conditions and get Ayurvedic remedies instantly."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10000)



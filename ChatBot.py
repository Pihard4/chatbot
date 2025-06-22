from flask import Flask, request, jsonify, render_template
import pandas as pd
import random
import speech_recognition as sr
from gtts import gTTS
import os
import tempfile
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from openai import OpenAI
from dotenv import load_dotenv
import re

load_dotenv()

client = OpenAI(api_key=os.getenv("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"))  # Use your .env file
DEFAULT_MODEL = "gpt-4o"

app = Flask(_name_)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

data = None
chat_history = []

GREETINGS = ['hi', 'hello', 'hey', 'hola', 'howdy', 'greetings']
WELL_BEING_QUESTIONS = ['how are you', 'how are you doing', 'how do you feel', "what's up"]

econometrics_kb = {
    "ols": "Régression linéaire multiple...",
    "heteroscedasticity": "L’hétéroscédasticité signifie...",
    "autocorrelation": "L’autocorrélation fait référence..."
}

def translate(key, lang):
    translation_dict = {
        "greeting_response": {
            "en": ["Hello!", "Hi there!"],
            "fr": ["Bonjour !", "Salut !"],
            "zh": ["你好！", "嗨！"]
        },
        "wellbeing_response": {
            "en": ["I'm doing well, thank you!"],
            "fr": ["Je vais bien, merci !"],
            "zh": ["我很好，谢谢你！"]
        },
        "unknown_question": {
            "en": "🤔 I didn't understand.",
            "fr": "🤔 Je n'ai pas compris.",
            "zh": "🤔 我不太明白。"
        }
    }
    return translation_dict.get(key, {}).get(lang, key)

def speak(text, lang="en"):
    try:
        tts = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            os.system(f"start {fp.name}")
    except Exception as e:
        print("Text-to-speech failed:", e)

def listen():
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            audio = r.listen(source)
            text = r.recognize_google(audio)
            return text.lower()
    except Exception as e:
        return str(e)

def run_regression(df, y_var, x_vars):
    try:
        formula = f"{y_var} ~ {' + '.join(x_vars)}"
        model = smf.ols(formula=formula, data=df).fit()
        return model
    except Exception as e:
        return str(e)

def get_llm_response(prompt):
    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert in econometrics."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return "LLM Error: " + str(e)

def get_fallback_response(user_input, lang_code):
    cleaned_input = user_input.lower()
    if any(greet in cleaned_input for greet in GREETINGS):
        return random.choice(translate("greeting_response", lang_code))
    elif any(cleaned_input == q for q in WELL_BEING_QUESTIONS):
        return random.choice(translate("wellbeing_response", lang_code))
    elif cleaned_input in econometrics_kb:
        return econometrics_kb[cleaned_input]
    else:
        return translate("unknown_question", lang_code)

def parse_regression_input(text):
    match = re.search(r"regression with (.+?) as y and (.+?) as x", text.lower())
    if match:
        y_var = match.group(1).strip()
        x_vars = [x.strip() for x in match.group(2).split(',')]
        return y_var, x_vars
    return None, None

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    global data
    file = request.files['file']
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        if file.filename.endswith(".csv"):
            data = pd.read_csv(filepath)
        elif file.filename.endswith(".xlsx"):
            data = pd.read_excel(filepath)
        elif file.filename.endswith(".txt"):
            data = pd.read_csv(filepath, delimiter="\t")
        return jsonify({"columns": list(data.columns)})
    return "No file uploaded", 400

@app.route('/regression', methods=['POST'])
def regression():
    global data
    if data is None:
        return "No data available", 400

    req = request.json
    y = req.get("y")
    x = req.get("x", [])
    model = run_regression(data, y, x)
    if isinstance(model, str):
        return jsonify({"error": model})

    summary_text = model.summary().as_text()
    return jsonify({
        "summary": summary_text,
        "r2": model.rsquared,
        "r2_adj": model.rsquared_adj,
        "f_stat": model.fvalue,
        "f_pvalue": model.f_pvalue
    })

@app.route('/chat', methods=['POST'])
def chat():
    global chat_history
    req = request.json
    message = req.get("message")
    lang = req.get("lang", "en")
    y_var, x_vars = parse_regression_input(message)

    if y_var and x_vars and data is not None:
        if y_var in data.columns and all(x in data.columns for x in x_vars):
            model = run_regression(data, y_var, x_vars)
            if not isinstance(model, str):
                chat_history.append({"role": "user", "content": message})
                return jsonify({"response": f"Regression done with Y={y_var} and X={x_vars}."})

    chat_history.append({"role": "user", "content": message})
    chat_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history])
    reply = get_llm_response(chat_prompt)
    if "LLM Error" in reply:
        reply = get_fallback_response(message, lang)
    chat_history.append({"role": "assistant", "content": reply})
    return jsonify({"response": reply})

if _name_ == '_main_':
    app.run(debug=True)

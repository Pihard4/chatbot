import streamlit as st
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
from openai import OpenAI
import re
from dotenv import load_dotenv # type: ignore
from streamlit_chat import message # type: ignore

load_dotenv()

client = OpenAI(api_key=os.getenv("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"))
DEFAULT_MODEL = "gpt-4o"

GREETINGS = ['hi', 'hello', 'hey', 'hola', 'howdy', 'greetings']
WELL_BEING_QUESTIONS = ['how are you', 'how are you doing', 'how do you feel', "what's up"]

econometrics_kb = {
    "ols": "*Régression Linéaire Multiple (OLS)* :\n\nLa régression linéaire multiple est une méthode statistique utilisée pour modéliser la relation entre une variable dépendante continue et plusieurs variables indépendantes. Elle permet d’évaluer l’effet individuel de chaque variable explicative.\n\n*Utilisations :* Prédiction, évaluation d’impact.\n\n*Hypothèses :* Linéarité, homoscédasticité, indépendance des erreurs, normalité des résidus, absence de multicolinéarité.\n\n*Résultats clés :* Coefficients, R², R² ajusté, F-statistique, p-values, Durbin-Watson.",
    "heteroscedasticity": "L’hétéroscédasticité signifie que la variance des erreurs n’est pas constante. Cela viole une des hypothèses de base du modèle OLS et peut conduire à des inférences statistiques incorrectes.",
    "autocorrelation": "L’autocorrélation fait référence à la corrélation entre les erreurs d’un modèle à différents points dans le temps. Cela est courant dans les données de séries temporelles."
}

def translate(key, lang):
    translation_dict = {
        "greeting_response": {
            "en": ["Hello!", "Hi there!", "Hey!", "Howdy!"],
            "fr": ["Bonjour !", "Salut !", "Coucou !", "Salut à toi !"],
            "zh": ["你好！", "嗨！", "嘿！", "您好！"]
        },
        "wellbeing_response": {
            "en": ["I'm doing well, thank you!", "I'm feeling great today!", "I'm a chatbot, I don't have feelings, but thank you for asking!"],
            "fr": ["Je vais bien, merci !", "Je me sens en pleine forme aujourd'hui !", "Je suis un chatbot, je n'ai pas de sentiments, mais merci de demander !"],
            "zh": ["我很好，谢谢你！", "我今天感觉很好！", "我是一个聊天机器人，没有感情，但谢谢你的关心！"]
        },
        "unknown_question": {
            "en": "🤔 I didn't understand. Can you specify a known model or ask clearly?",
            "fr": "🤔 Je n'ai pas compris. Pouvez-vous préciser un modèle connu ou poser une question plus claire ?",
            "zh": "🤔 我不太明白。你能指定一个已知的模型或更清楚地提问吗？"
        },
        "upload_success": {
            "en": "Data uploaded successfully!",
            "fr": "Données téléchargées avec succès !",
            "zh": "数据上传成功！"
        },
        "upload_fail": {
            "en": "Could not load data:",
            "fr": "Impossible de charger les données :",
            "zh": "无法加载数据："
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
        st.warning(f"Text-to-speech failed: {e}")

def listen():
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening... Speak now.")
            audio = r.listen(source)
            text = r.recognize_google(audio)
            return text.lower()
    except sr.UnknownValueError:
        return "Sorry, I could not understand your voice."
    except sr.RequestError:
        return "Error with speech service."
    except Exception as e:
        return f"Microphone error: {e}"

def run_regression(df, y_var, x_vars):
    try:
        formula = f"{y_var} ~ {' + '.join(x_vars)}"
        model = smf.ols(formula=formula, data=df).fit()
        return model
    except Exception as e:
        return f"Erreur dans la régression : {e}"

def show_model_diagnostics(model):
    st.subheader("📈 Résumé du Modèle")
    summary_str = str(model.summary())
    st.text(summary_str)
    st.download_button("⬇ Télécharger le résumé", data=summary_str, file_name="summary_model.txt", mime='text/plain')
    st.markdown("---")
    st.subheader("📊 Analyse de la robustesse")
    st.markdown(f"*R² :* {model.rsquared:.4f}")
    st.markdown(f"*R² ajusté :* {model.rsquared_adj:.4f}")
    st.markdown(f"*Statistique F :* {model.fvalue:.4f} (p = {model.f_pvalue:.4g})")
    st.markdown(f"*Durbin-Watson :* {sm.stats.durbin_watson(model.resid):.4f}")
    fig1, ax1 = plt.subplots()
    sns.histplot(model.resid, kde=True, ax=ax1)
    st.pyplot(fig1)
    fig2, ax2 = plt.subplots()
    sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, ax=ax2)
    st.pyplot(fig2)

def get_llm_response(prompt):
    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "system", "content": "You are an expert assistant who answers questions about econometrics, public figures, and general knowledge using your expertise."}, {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        if "insufficient_quota" in str(e).lower() or "429" in str(e):
            return "⚠️ OpenAI quota exceeded. Using local fallback response."
        return f"LLM error: {e}"

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

def main():
    st.markdown("""
        <style>
            body {
                background-color: #e6dccc;
                color: #2e1d0f;
            }
            .stApp {
                background-color: #f1e6d4;
            }
            .block-container {
                padding-top: 2rem;
            }
            h1, h2, h3, h4 {
                color: #3b2414;
            }
            .css-1d391kg, .css-1v0mbdj, .st-c9 {
                background-color: #d0bfa5 !important;
                border-radius: 12px;
                padding: 1em;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
            }
            .stButton>button {
                background-color: #5b3b1e;
                color: white;
                border: none;
                padding: 0.6em 1.2em;
                border-radius: 8px;
                font-weight: bold;
            }
            .stButton>button:hover {
                background-color: #472e17;
            }
            .chat-user {
                background-color: #b89676;
                color: #2e1d0f;
                padding: 10px;
                border-radius: 10px;
                margin: 8px 0;
                text-align: right;
            }
            .chat-assistant {
                background-color: #f5e8d7;
                color: #2e1d0f;
                padding: 10px;
                border-radius: 10px;
                margin: 8px 0;
                text-align: left;
            }
        </style>
    """, unsafe_allow_html=True)
    st.image("wmremove-transformed-removebg-preview.png", width=120)
    st.title("📈 Assistant Économétrique Intelligent")
    st.markdown("""
        <h4 style='text-align: center; color: #5b3b1e;'>Welcome to ZO Analytics 🤎</h4>
        <p style='text-align: center; color: #3b2414;'>Created with passion by <strong>Zainab Dribigi</strong> & <strong>Oussama Sabik</strong></p>
    """, unsafe_allow_html=True)
    language = st.selectbox("🌐 Choisissez votre langue", ["French", "English", "Chinese"])
    lang_map = {"English": "en", "French": "fr", "Chinese": "zh"}
    lang_code = lang_map.get(language, "en")
    tabs = st.tabs(["📤 Données", "📊 Régression", "💬 Chat"])

    with tabs[0]:
        uploaded_file = st.file_uploader("Importez un fichier CSV, Excel ou TXT", type=["csv", "xlsx", "txt"])
        global data
        data = None
        if uploaded_file:
            try:
                if uploaded_file.name.endswith(".csv"):
                    data = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(".xlsx"):
                    data = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith(".txt"):
                    data = pd.read_csv(uploaded_file, delimiter="\t")
                st.success(translate("upload_success", lang_code))
                st.dataframe(data)
            except Exception as e:
                st.error(f"{translate('upload_fail', lang_code)} {e}")

    with tabs[1]:
        st.header("⚙ Régression")
        if data is not None:
            numeric_cols = data.select_dtypes(include='number').columns
            y_var = st.selectbox("Choisissez la variable dépendante (Y)", numeric_cols)
            x_vars = st.multiselect("Choisissez les variables explicatives (X)", [col for col in numeric_cols if col != y_var])
            if st.button("Lancer la régression"):
                model = run_regression(data, y_var, x_vars)
                if isinstance(model, str):
                    st.error(model)
                else:
                    show_model_diagnostics(model)
        else:
            st.info("Veuillez d'abord importer un fichier de données dans l'onglet Données.")

    with tabs[2]:
        if st.button("🆕 New Chat"):
            st.session_state.chat_history = []
            st.rerun()
        st.header("💬 Discussion Économétrique")
        input_method = st.radio("Méthode d'entrée", ["Texte", "Voix"])
        user_input = ""
        if input_method == "Texte":
            user_input = st.text_input("💬 Tapez votre message")
        elif input_method == "Voix":
            if st.button("🎙 Parler"):
                user_input = listen()
                st.write("🗣 Vous avez dit :", user_input)

        if user_input:
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []

            y_var, x_vars = parse_regression_input(user_input)
            if y_var and x_vars and data is not None:
                if y_var in data.columns and all(x in data.columns for x in x_vars):
                    model = run_regression(data, y_var, x_vars)
                    if not isinstance(model, str):
                        st.session_state.chat_history.append({"role": "user", "content": user_input})
                        st.markdown(f"### 📊 Régression automatique lancée\n\n**Y :** {y_var} — **X :** {', '.join(x_vars)}")
                        show_model_diagnostics(model)
                        return

            st.session_state.chat_history.append({"role": "user", "content": user_input})
            chat_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history])
            response = get_llm_response(chat_prompt)
            if "⚠️ OpenAI quota exceeded" in response:
                response = get_fallback_response(user_input, lang_code)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

        if 'chat_history' in st.session_state:
            for i, msg in enumerate(st.session_state.chat_history):
                is_user = msg["role"] == "user"
                message(msg["content"], is_user=is_user, key=str(i))
st.markdown("""
        <hr style='border-top: 1px solid #ccbbaa;'>
        <p style='text-align: center; font-size: 13px; color: #7a5b3e;'>© 2025 - All rights reserved by Zainab Dribigi & Oussama Sabik</p>
    """, unsafe_allow_html=True)
if __name__ == "__main__":
    main()

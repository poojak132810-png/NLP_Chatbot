from flask import Flask, render_template, request, jsonify, session
import os
import requests

# ----------------------------
# Configuration (no API key needed)
# ----------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL = os.getenv("MODEL", "llama3.2")

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful, concise assistant. Answer clearly and avoid unnecessary jargon.",
)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")


def _generate_reply(messages):
    resp = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "messages": messages, "stream": False},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _history_to_messages(history, user_msg):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        role = "user" if turn["role"] == "user" else "assistant"
        messages.append({"role": role, "content": turn["text"]})
    messages.append({"role": "user", "content": user_msg})
    return messages


@app.route("/", methods=["GET"])
def home():
    if "history" not in session:
        session["history"] = []
    return render_template("chat.html", history=session["history"])


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    history = session.get("history", [])
    messages = _history_to_messages(history, user_msg)

    try:
        reply = _generate_reply(messages)
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Can't reach Ollama. Open the Ollama app and try again."}), 500
    except Exception as e:
        return jsonify({"error": f"Error: {e}"}), 500

    history.append({"role": "user", "text": user_msg})
    history.append({"role": "model", "text": reply})
    session["history"] = history[-20:]

    return jsonify({"reply": reply})


@app.route("/reset", methods=["POST"])
def reset():
    session["history"] = []
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)

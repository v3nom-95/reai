# 🚑 REAI — Responsible AI Medical Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Algorand Testnet](https://img.shields.io/badge/Algorand-Testnet-blueviolet.svg)

A blockchain-backed, responsible AI medical assistant built with Python, Flask, and the Algorand SDK. The app stores every interaction on the Algorand testnet so that responses are transparent, auditable, and traceable.

---

## ✨ Project Overview

REAI is designed to combine simple medical guidance with strong disclaimers and safe behavior. It is ideal for learning how to integrate AI-style Q&A with secure blockchain storage.

### Why this project exists
- Provide general health information without offering medical diagnosis
- Record all user interactions immutably on Algorand Testnet
- Let the system learn from previous health questions and responses
- Keep the interface easy to use with a clean chat experience

---

## 🚀 Key Features

- **Responsible medical guidance** with clear disclaimers
- **Algorand Testnet integration** using `py-algorand-sdk`
- **Blockchain audit trail** for every chat interaction
- **Semantic learning** from similar past queries
- **Flask-powered UI** with simple chat interface

---

## 🧠 Architecture

| Layer | Technology | Purpose |
|---|---|---|
| Backend | Flask | Web server, routing, API |
| AI | Rule-based logic + embeddings | Health query matching and answer selection |
| Blockchain | Algorand Testnet | Persistent, immutable storage |
| Frontend | HTML/CSS | Chat interface and interaction |

---

## 🛠️ Install & Setup

### Prerequisites
- Python 3.8 or later
- `pip`
- Algorand Testnet account with ALGO for transaction fees

### Install dependencies
```bash
git clone https://github.com/VenkatBabu95/reai.git
cd reai
pip install -r requirements.txt
```

> `requirements.txt` includes `py-algorand-sdk` so Algorand blockchain interaction works correctly.

### Run the app
```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser to use the medical chat assistant.

---

## 💬 Usage Guide

### How to chat
1. Visit `http://127.0.0.1:5000`
2. Type a health-related question
3. Press Enter or click **Send**
4. Read the response and the safety disclaimer

### Example prompts
- `I have a headache and fever`
- `What should I do for a sore throat?`
- `How can I manage stress and anxiety?`

---

## 📌 Supported Health Topics

- Fever & illnesses
- Headaches & migraines
- Respiratory symptoms
- Throat and cold symptoms
- Digestive issues
- Pain and injury management
- Diabetes and heart health
- Mental health support
- Sleep, allergies, skin, eye, and dental issues

---

## 🔗 Algorand Blockchain Features

- All chat results are saved as transactions on the Algorand Testnet
- Each interaction becomes part of an immutable audit trail
- The app can search previous records to improve similarity matching
- Testnet usage means development is safe and isolated from mainnet

---

## ⚠️ Important Disclaimer

This app is for educational and informational purposes only.

- Not medical advice
- Not a diagnosis tool
- Not a replacement for professional healthcare
- Always consult a qualified medical professional for personal health decisions

---

## 📦 Dependencies

The app uses the following Python packages:

- `Flask` — web server
- `py-algorand-sdk` — Algorand blockchain integration
- `sentence-transformers` — semantic embeddings
- `torch` — deep learning backend for embeddings
- `numpy` — numerical operations

> External libraries are governed by their own licenses. This repository is licensed under the MIT License.

---

## 📁 Project Files

- `app.py` — main Flask application
- `blockchain.py` — Algorand testnet logic and transaction management
- `custom_model.py` — AI logic and response generation
- `model.py` — medical topic modeling and embeddings
- `templates/index.html` — chat interface
- `static/style.css` — styling for the chat UI
- `LICENSE` — MIT license for this project

---

## 🤝 Contributing

Feel free to improve the project by:

- Expanding the medical knowledge base
- Improving the learning algorithm
- Adding additional safety checks
- Enhancing the UI experience

---

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for full terms.

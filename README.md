# Responsible AI Medical Assistant

A trustworthy, blockchain-backed AI medical assistant that provides responsible health advice while ensuring accountability through immutable storage on the Algorand blockchain.

- License: MIT

## Features

- **Responsible AI**: Provides general health information with strong disclaimers, never gives personalized medical advice
- **Blockchain Integration**: All AI interactions are stored immutably on Algorand testnet for transparency and learning
- **Learning System**: AI improves over time by learning from similar past queries stored on-chain
- **Medical Knowledge**: Covers 20+ common health conditions with detailed, evidence-based guidance

## How It Works

### Architecture
- **Backend**: Python Flask web application
- **AI Engine**: Custom rule-based system augmented with embedding-based semantic search for improved matching and learning
- **Blockchain**: Algorand testnet for immutable data storage and retrieval
- **Frontend**: Simple HTML/CSS chat interface

### AI Logic
1. **Query Processing**: User submits a health-related question
2. **Blockchain Learning**: Searches past interactions on-chain for similar queries
3. **Response Generation**: Uses rule-based matching for medical conditions
4. **Storage**: Saves the interaction on blockchain for future learning

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip
- Algorand testnet account with some ALGO for transactions

### Installation
```bash
# Clone the repository
git clone https://github.com/VenkatBabu95/reai.git
cd reai

# Install dependencies
pip install -r requirements.txt
```

> This project uses `py-algorand-sdk` for Algorand blockchain integration, installed via `requirements.txt`.

### Configuration
The app uses hardcoded Algorand testnet credentials. Ensure your testnet account has sufficient ALGO for transaction fees.

## Running the Application

### Start the Server
```bash
python app.py
```

The app will run on `http://127.0.0.1:5000`

### Access the Chat Interface
1. Open your browser and navigate to `http://127.0.0.1:5000`
2. You'll see a simple chat interface
3. Type your health-related question in the input box
4. Click "Send" or press Enter

### Example Interactions

**User**: "I have a fever"
**AI**: "For fever, rest in bed, stay hydrated... [detailed advice] ... Disclaimer: This is general information. Please consult a healthcare professional for medical advice."

**User**: "I'm feeling anxious"
**AI**: "For anxiety or stress, practice deep breathing... [detailed advice] ... Disclaimer: This is general information. Please consult a healthcare professional for medical advice."

## Supported Health Topics

The AI provides guidance for:
- Fever & Temperature
- Headaches & Migraines
- Cough & Respiratory Issues
- Sore Throat
- Colds & Flu
- Digestive Issues (Nausea, Vomiting, Diarrhea)
- Pain Management & Injuries
- Diabetes
- Heart Health
- Mental Health (Anxiety, Depression, Stress)
- Sleep Disorders
- Allergies
- Skin Conditions
- Eye & Ear Issues
- Dental Problems

## Blockchain Features

### Transparency
- Every AI interaction is recorded on the Algorand blockchain
- Each response includes a transaction ID for verification
- Data is immutable and publicly auditable

### Learning Mechanism
- AI queries past blockchain data for similar questions
- Improves responses based on historical interactions
- Decentralized learning ensures no single point of control

## Important Disclaimers

⚠️ **This AI is NOT a substitute for professional medical advice**
- Always consult qualified healthcare professionals
- Never use this for emergencies - call emergency services
- Responses are for general information only
- Individual health conditions vary - seek personalized care

## Technical Details

### Dependencies
- Flask: Web framework
- py-algorand-sdk: Blockchain integration
- difflib: Similarity matching for learning
- sentence-transformers + torch: Embedding-based semantic search for improved matching
- numpy: Numerical operations for embeddings

> Third-party dependencies are installed via `pip` and are governed by their own licenses. This project itself is licensed under the MIT License.

### API Endpoints
- `GET /`: Chat interface
- `POST /chat`: Send message and receive AI response

### Blockchain Integration
- Uses Algorand testnet indexer for data retrieval
- Stores JSON data in transaction notes
- Handles large responses by storing hashes when needed

## Contributing

This project demonstrates responsible AI development with blockchain accountability. For improvements:
1. Enhance medical knowledge base
2. Improve similarity matching algorithms
3. Add more health topics
4. Optimize blockchain storage

## License

This project is licensed under the MIT License. See the included `LICENSE` file for details.
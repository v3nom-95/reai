from flask import Flask, request, jsonify, render_template
import algosdk
import hashlib
import json
from model import (
    find_similar_by_embedding, load_model, train_model_on_batch, 
    get_model_info, get_blockchain_stats,
    propose_training_to_consensus, approve_training_proposal, apply_consensus_training,
    propose_train_from_chain, apply_consensus_training_with_dp
)
from blockchain import blockchain
from custom_model import custom_llm
import os
import difflib

app = Flask(__name__)

# Algorand testnet setup
algod_token = ""  # No token needed for public testnet
algod_address = "https://testnet-api.algonode.cloud"
algod_client = algosdk.v2client.algod.AlgodClient(algod_token, algod_address)

# Use your own testnet credentials here
private_key = "R7OkpcgoDySzHThJSA3VNwMb4H61wothGATktDTzmuC91a9ywL+H1bqddSTRO6wZD+iqDYkacxjjlzcNSm3Q8A=="
address = "XXK264WAX6D5LOU5OUSNCO5MDEH6RKQNRENHGGHDS43Q2STN2DYFEFWDGY"

print(f"Using provided account address: {address}")
print("Ensure this address is funded on testnet.")

# Custom Learning Medical Assistant AI
DATA_FILE = 'medical_data.json'

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    return []

def save_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f)

def find_similar_response(user_message, data, threshold=0.6):
    best_match = None
    best_score = 0
    for entry in data:
        score = difflib.SequenceMatcher(None, user_message.lower(), entry['query'].lower()).ratio()
        if score > best_score and score >= threshold:
            best_match = entry['response']
            best_score = score
    return best_match

def load_data_from_blockchain():
    # Initialize indexer client
    indexer_token = ""
    indexer_address = "https://testnet-idx.algonode.cloud"
    indexer_client = algosdk.v2client.indexer.IndexerClient(indexer_token, indexer_address)
    
    data = []
    try:
        # Query transactions for the address
        response = indexer_client.search_transactions_by_address(address=address, limit=100)
        for txn in response['transactions']:
            if 'note' in txn and txn['note']:
                try:
                    # Decode note
                    note_bytes = algosdk.encoding.base64.b64decode(txn['note'])
                    note_str = note_bytes.decode('utf-8')
                    # Try to parse as JSON
                    interaction = json.loads(note_str)
                    if isinstance(interaction, dict) and 'query' in interaction and 'response' in interaction:
                        data.append(interaction)
                except:
                    # If not JSON, skip
                    pass
    except Exception as e:
        print(f"Error loading data from blockchain: {e}")
    return data

def get_medical_advice(user_message, data):
    # Try to learn from past data
    learned_response = find_similar_response(user_message, data)
    if learned_response:
        return learned_response
    
    # Fallback to enhanced rule-based
    msg = user_message.lower()
    if "fever" in msg or "temperature" in msg or "hot" in msg:
        return "For fever, rest in bed, stay hydrated with water or electrolyte drinks, and take over-the-counter fever reducers like acetaminophen (Tylenol) or ibuprofen if needed. Monitor your temperature. If fever exceeds 103°F (39.4°C), lasts more than 3 days, or is accompanied by severe symptoms like difficulty breathing, chest pain, or confusion, seek immediate medical attention."
    elif "headache" in msg or "migraine" in msg or "head pain" in msg:
        return "For headaches, rest in a dark, quiet room, apply a cold compress to your forehead, and stay hydrated. Over-the-counter pain relievers like ibuprofen or aspirin may help, but avoid overuse. If headaches are severe, frequent, sudden, or accompanied by nausea, vision changes, or neck stiffness, consult a doctor to rule out serious causes."
    elif "cough" in msg or "coughing" in msg:
        return "For cough, stay hydrated, use honey in warm water or herbal teas, and consider over-the-counter cough syrups. Elevate your head while sleeping. If cough is persistent (over 2 weeks), bloody, or associated with shortness of breath, wheezing, chest pain, or high fever, see a healthcare provider."
    elif "sore throat" in msg or "throat pain" in msg:
        return "For sore throat, gargle with warm salt water several times a day, stay hydrated, suck on lozenges, and use a humidifier. Over-the-counter pain relievers can help. If sore throat lasts more than a week, is severe, or comes with fever, difficulty swallowing, or rash, consult a doctor—it could be strep throat or another infection."
    elif "cold" in msg or "flu" in msg or "runny nose" in msg or "congestion" in msg:
        return "For colds or flu, rest, drink plenty of fluids, use saline nasal sprays for congestion, and take over-the-counter medications for symptoms like acetaminophen for fever/pain and decongestants if needed. Get adequate sleep and isolate to avoid spreading. See a doctor if symptoms worsen, last over 10 days, or include high fever, severe cough, or difficulty breathing."
    elif "stomach" in msg or "nausea" in msg or "vomit" in msg or "vomiting" in msg or "diarrhea" in msg:
        return "For nausea, vomiting, or diarrhea, sip clear fluids slowly to avoid dehydration, avoid solid foods temporarily, and rest. Over-the-counter anti-nausea medications or rehydration solutions may help. If symptoms persist over 48 hours, include blood in vomit/stool, severe dehydration signs (dry mouth, dizziness, reduced urination), or high fever, seek medical help."
    elif "back pain" in msg or "backache" in msg:
        return "For back pain, use proper posture, avoid heavy lifting, apply heat or cold, and do gentle stretches. Over-the-counter pain relievers can help. If pain is severe, radiates down the leg, or is accompanied by numbness/weakness, see a doctor to rule out serious issues."
    elif "joint pain" in msg or "arthritis" in msg:
        return "For joint pain, maintain a healthy weight, exercise gently, and use heat/cold therapy. Over-the-counter anti-inflammatory medications may help. If pain is severe, swelling persists, or affects mobility, consult a rheumatologist."
    elif "pain" in msg or "injury" in msg or "sprain" in msg or "strain" in msg:
        return "For pain or minor injuries, rest the affected area, apply ice for 15-20 minutes every few hours, compress with a bandage, and elevate if possible (RICE method). Use over-the-counter pain relievers like ibuprofen. If pain is severe, swelling doesn't improve, you can't bear weight, or injury involves head/neck/back, see a doctor immediately."
    elif "diabetes" in msg or "blood sugar" in msg or "glucose" in msg:
        return "For diabetes management, monitor blood sugar regularly as advised, follow your prescribed diet, take medications as directed, and exercise consistently. Recognize signs of hypo/hyperglycemia and have a treatment plan. Consult your endocrinologist regularly and seek immediate care for severe symptoms like confusion, seizures, or unconsciousness."
    elif "heart" in msg or "chest pain" in msg or "palpitations" in msg:
        return "Chest pain, palpitations, or heart-related symptoms are serious. If you experience chest pain, shortness of breath, dizziness, fainting, or irregular heartbeat, call emergency services (911) immediately. Do not drive yourself. Even if symptoms resolve, see a doctor for evaluation."
    elif "anxiety" in msg or "stress" in msg or "panic" in msg:
        return "For anxiety or stress, practice deep breathing, mindfulness, or relaxation techniques. Regular exercise, adequate sleep, and a healthy diet can help. If anxiety is overwhelming, persistent, or interferes with daily life, consider talking to a mental health professional. Avoid self-medicating with alcohol or drugs."
    elif "depression" in msg or "sad" in msg or "mood" in msg:
        return "For depression or low mood, maintain routines, exercise, connect with others, and consider healthy hobbies. Professional help from a therapist or counselor is often beneficial. If you have thoughts of self-harm or suicide, seek immediate help from a crisis hotline or emergency services."
    elif "sleep" in msg or "insomnia" in msg:
        return "For sleep issues, maintain a consistent sleep schedule, create a relaxing bedtime routine, avoid screens before bed, and ensure your sleep environment is cool and dark. Limit caffeine and heavy meals in the evening. If insomnia persists or affects daily functioning, consult a doctor."
    elif "allergy" in msg or "allergies" in msg or "hives" in msg or "rash" in msg:
        return "For allergies, avoid triggers, use over-the-counter antihistamines, and apply topical creams for rashes. If symptoms are severe, involve swelling of the face/throat, difficulty breathing, or anaphylaxis, seek emergency care. For chronic allergies, see an allergist for testing and management."
    elif "skin" in msg or "rash" in msg or "itchy" in msg:
        return "For skin issues, keep the area clean and moisturized, avoid irritants, and use over-the-counter hydrocortisone cream for mild itching. If rash is widespread, painful, infected (red, warm, pus), or accompanied by fever, see a dermatologist."
    elif "eye" in msg or "vision" in msg or "red eye" in msg:
        return "For eye issues, avoid rubbing, use artificial tears for dryness, and rest your eyes. If you have pain, vision changes, discharge, sensitivity to light, or injury, see an eye doctor promptly."
    elif "ear" in msg or "earache" in msg or "hearing" in msg:
        return "For ear issues, avoid inserting objects, use over-the-counter pain relievers, and apply warm compresses. If pain is severe, accompanied by fever, discharge, dizziness, or hearing loss, see a doctor."
    elif "dental" in msg or "tooth" in msg or "mouth" in msg:
        return "For dental issues, maintain good oral hygiene, rinse with warm salt water for pain, and avoid hard foods. Use over-the-counter pain relievers. See a dentist for persistent pain, swelling, or infection."
    else:
        return "I'm a responsible medical assistant. For general health concerns, maintain a balanced diet, exercise regularly, stay hydrated, get adequate sleep, and manage stress. For specific symptoms or conditions, please consult a qualified healthcare professional. I cannot provide personalized medical advice or diagnoses."

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    
    # Load data from blockchain
    data = load_data_from_blockchain()
    
    # Train model on this query and mine block
    try:
        train_result = train_model_on_batch([user_message], labels=[1], learning_rate=0.01)
        blockchain.mine_block(miner_id='medical-assistant')
    except Exception as e:
        print(f"Model training error: {e}")
    
    # Attempt to use embedding-based similarity first (faster/more robust matching)
    try:
        # Ensure model is loaded (lazy load)
        try:
            load_model()
        except Exception as e:
            print(f"Warning: model failed to load at import time: {e}")

        found = None
        try:
            found = find_similar_by_embedding(user_message, data)
        except Exception as e:
            print(f"Embedding lookup failed: {e}")

        if found:
            # found is (response, score)
            ai_response = found[0]
        else:
            ai_response = get_medical_advice(user_message, data)

        # Add disclaimer
        ai_response += "\n\nDisclaimer: This is general information. Please consult a healthcare professional for medical advice."

        # Prepare data for blockchain storage including response hash
        response_hash = hashlib.sha256(ai_response.encode()).hexdigest()
        interaction_data = {
            "query": user_message,
            "response": ai_response,
            "hash": response_hash
        }
        note = json.dumps(interaction_data).encode('utf-8')
        
        # If note is too long, store hash instead
        if len(note) > 1000:
            note = response_hash.encode('utf-8')
    except Exception as e:
        ai_response = f"I apologize, but I'm experiencing technical difficulties: {str(e)}. Please consult a healthcare professional for medical advice."
        response_hash = hashlib.sha256(ai_response.encode()).hexdigest()
        note = b"error"
    
    # Create a transaction to store the interaction on blockchain
    params = algod_client.suggested_params()
    txn = algosdk.transaction.PaymentTxn(
        sender=address,
        sp=params,
        receiver=address,
        amt=0,
        note=note
    )
    signed_txn = txn.sign(private_key)
    
    try:
        txid = algod_client.send_transaction(signed_txn)
        # Wait for confirmation
        algosdk.transaction.wait_for_confirmation(algod_client, txid, 4)
        blockchain.mine_block(miner_id='blockchain-recorder')
        
        return jsonify({
            'response': ai_response,
            'txid': txid,
            'hash': response_hash,
            'model_info': get_model_info(),
            'blockchain_stats': get_blockchain_stats()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    """Get current model information and training status."""
    return jsonify({
        'model': get_model_info(),
        'blockchain': get_blockchain_stats(),
        'blockchain_chain_length': len(blockchain.get_chain())
    })


@app.route('/blockchain-history', methods=['GET'])
def blockchain_history():
    """Get blockchain transaction history."""
    history = blockchain.get_model_history('custom-medical-llm-v1')
    return jsonify({
        'total_records': len(history),
        'records': history[:50]  # Latest 50
    })


@app.route('/train', methods=['POST'])
def train():
    """Manual model training endpoint."""
    data = request.json
    texts = data.get('texts', [])
    learning_rate = data.get('learning_rate', 0.01)
    
    if not texts:
        return jsonify({'error': 'No texts provided'}), 400
    
    try:
        metrics = train_model_on_batch(texts, learning_rate=learning_rate)
        block_result = blockchain.mine_block(miner_id='training-endpoint')
        
        return jsonify({
            'status': 'success',
            'metrics': metrics,
            'block': block_result,
            'model_info': get_model_info()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/save-checkpoint', methods=['POST'])
def save_checkpoint():
    """Save model checkpoint to disk and blockchain."""
    try:
        checkpoint_hash = custom_llm.save_checkpoint('checkpoints/model_checkpoint.json')
        block_result = blockchain.mine_block(miner_id='checkpoint-saver')
        
        return jsonify({
            'status': 'success',
            'checkpoint_hash': checkpoint_hash,
            'block': block_result,
            'model_info': get_model_info()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/decentralized/propose-training', methods=['POST'])
def propose_training():
    """Step 1: Propose training update to blockchain validators."""
    data = request.json
    texts = data.get('texts', [])
    learning_rate = data.get('learning_rate', 0.01)
    
    if not texts:
        return jsonify({'error': 'No texts provided'}), 400
    
    try:
        result = propose_training_to_consensus(texts, learning_rate)
        return jsonify({
            'status': 'success',
            'proposal': result,
            'blockchain_stats': get_blockchain_stats()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/decentralized/propose-from-chain', methods=['POST'])
def propose_from_chain():
    """Build dataset from on-chain records and propose training."""
    data = request.json or {}
    limit = int(data.get('limit', 200))
    learning_rate = float(data.get('learning_rate', 0.01))
    filter_pii = bool(data.get('filter_pii', True))

    try:
        result = propose_train_from_chain(limit=limit, learning_rate=learning_rate, filter_pii=filter_pii)
        return jsonify({'status': 'success', 'proposal': result, 'blockchain_stats': get_blockchain_stats()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/decentralized/approve-proposal', methods=['POST'])
def approve_proposal():
    """Step 2: Validator approves training proposal."""
    data = request.json
    proposal_hash = data.get('proposal_hash')
    validator_id = data.get('validator_id', 'validator-' + os.urandom(4).hex())
    
    if not proposal_hash:
        return jsonify({'error': 'proposal_hash required'}), 400
    
    try:
        result = approve_training_proposal(proposal_hash, validator_id)
        return jsonify({
            'status': 'success',
            'approval': result,
            'blockchain_stats': get_blockchain_stats()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/decentralized/apply-training', methods=['POST'])
def apply_training():
    """Step 3: Apply consensus-approved training to model."""
    data = request.json
    proposal_hash = data.get('proposal_hash')
    texts = data.get('texts', [])
    learning_rate = data.get('learning_rate', 0.01)
    
    if not proposal_hash or not texts:
        return jsonify({'error': 'proposal_hash and texts required'}), 400
    
    try:
        dp_params = data.get('dp_params') if isinstance(data, dict) else None
        if dp_params:
            metrics = apply_consensus_training_with_dp(proposal_hash, texts, learning_rate, dp_params=dp_params)
        else:
            metrics = apply_consensus_training(proposal_hash, texts, learning_rate)

        if metrics.get('status') == 'failed':
            return jsonify({'error': metrics.get('reason')}), 403

        return jsonify({
            'status': 'success',
            'metrics': metrics,
            'model_info': get_model_info(),
            'blockchain_stats': get_blockchain_stats()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/decentralized/status', methods=['GET'])
def decentralized_status():
    """Get decentralized ML training status."""
    chain = blockchain.get_chain()
    pending = len(blockchain.pending_records)
    
    # Count proposals and approvals
    proposals = 0
    approvals = 0
    updates = 0
    
    for block in chain:
        for record in block.get('records', []):
            if record['type'] == 'training_proposal':
                proposals += 1
            elif record['type'] == 'validator_approval':
                approvals += 1
            elif record['type'] == 'model_update_approved':
                updates += 1
    
    return jsonify({
        'blockchain': get_blockchain_stats(),
        'model': get_model_info(),
        'decentralized_stats': {
            'total_proposals': proposals,
            'total_approvals': approvals,
            'applied_updates': updates,
            'pending_records': pending
        },
        'consensus_required': True,
        'validators_required': 1
    })


if __name__ == '__main__':
    app.run(debug=True)
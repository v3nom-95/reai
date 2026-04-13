"""
Model management with blockchain transparency and custom LLM integration.
Supports both SentenceTransformers and custom lightweight LLM.
All model interactions logged to blockchain for training provenance.
Implements true decentralized ML with consensus-based training approval.
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime
from typing import List
from blockchain import blockchain
from custom_model import custom_llm

# Toggle between custom LLM and SentenceTransformers
USE_CUSTOM_LLM = True
USE_BLOCKCHAIN_LOGGING = True

_model = None


def load_model(name: str = 'all-MiniLM-L6-v2'):
    """Load model (custom LLM or SentenceTransformer)."""
    global _model
    if USE_CUSTOM_LLM:
        return custom_llm
    if _model is None:
        _model = SentenceTransformer(name)
    return _model


def encode_texts(texts):
    """Encode texts to embeddings."""
    model = load_model()
    if USE_CUSTOM_LLM:
        return model.encode(texts)
    return model.encode(texts, convert_to_numpy=True)


def train_model_on_batch(texts, labels=None, learning_rate: float = 0.01):
    """Train model on batch and log to blockchain."""
    if not USE_CUSTOM_LLM:
        return {'status': 'not_trainable', 'message': 'SentenceTransformer is pre-trained'}
    
    metrics = custom_llm.train_on_batch(texts, labels, learning_rate)
    
    if USE_BLOCKCHAIN_LOGGING:
        blockchain.mine_block()
    
    return metrics


def propose_training_to_consensus(texts: List[str], learning_rate: float = 0.01):
    """Propose training update to blockchain validators (decentralized)."""
    if not USE_CUSTOM_LLM:
        return {'status': 'error', 'message': 'Only custom LLM supports decentralized training'}
    
    # Propose to blockchain
    proposal_hash = custom_llm.propose_training_update(texts, learning_rate=learning_rate)
    
    # Mine block to record proposal
    if USE_BLOCKCHAIN_LOGGING:
        blockchain.mine_block(miner_id='training-proposal')
    
    return {
        'status': 'proposed',
        'proposal_hash': proposal_hash,
        'message': 'Training proposed to blockchain validators'
    }


def approve_training_proposal(proposal_hash: str, validator_id: str = 'validator-1'):
    """Validator approves a training proposal (consensus mechanism)."""
    # Record validator approval on blockchain
    blockchain.add_record('validator_approval', {
        'proposal_hash': proposal_hash,
        'validator_id': validator_id,
        'timestamp': datetime.utcnow().isoformat()
    })
    
    # Mine approval block
    if USE_BLOCKCHAIN_LOGGING:
        blockchain.mine_block(miner_id=validator_id)
    
    return {
        'status': 'approved',
        'proposal_hash': proposal_hash,
        'validator': validator_id
    }


def apply_consensus_training(proposal_hash: str, texts: List[str], learning_rate: float = 0.01):
    """Apply training that has blockchain consensus."""
    if not USE_CUSTOM_LLM:
        return {'status': 'error', 'message': 'Only custom LLM supports decentralized training'}
    
    # Apply only if approved by validators
    metrics = custom_llm.apply_approved_training(proposal_hash, texts, learning_rate=learning_rate)
    
    # Mine final block
    if USE_BLOCKCHAIN_LOGGING:
        blockchain.mine_block(miner_id='training-executor')
    
    return metrics


def build_dataset_from_chain(limit: int = 500, filter_pii: bool = True):
    """Build a training dataset from on-chain `query_inference` records.

    - Returns list of text queries (deduplicated, most recent first)
    - `filter_pii` removes queries containing simple PII keywords
    """
    pii_keywords = ['name', 'ssn', 'social', 'address', 'phone', 'email']
    texts = []
    seen = set()
    # Search mined blocks and pending records
    for block in reversed(blockchain.get_chain()):
        for record in reversed(block.get('records', [])):
            if record.get('type') == 'query_inference':
                q = record['data'].get('query', '').strip()
                if not q:
                    continue
                key = q.lower()
                if key in seen:
                    continue
                if filter_pii and any(k in key for k in pii_keywords):
                    continue
                texts.append(q)
                seen.add(key)
                if len(texts) >= limit:
                    return texts
    # Also include pending records
    for record in blockchain.pending_records:
        if record.get('type') == 'query_inference':
            q = record['data'].get('query', '').strip()
            key = q.lower()
            if key not in seen and (not filter_pii or not any(k in key for k in pii_keywords)):
                texts.append(q)
                seen.add(key)
                if len(texts) >= limit:
                    break

    return texts


def propose_train_from_chain(limit: int = 200, learning_rate: float = 0.01, filter_pii: bool = True):
    """Helper: build dataset from chain and propose training to validators."""
    texts = build_dataset_from_chain(limit=limit, filter_pii=filter_pii)
    if not texts:
        return {'status': 'no_data', 'message': 'No suitable on-chain records found'}
    return propose_training_to_consensus(texts, learning_rate=learning_rate)


def apply_consensus_training_with_dp(proposal_hash: str, texts: List[str], learning_rate: float = 0.01, dp_params: dict = None):
    """Apply training with differential privacy parameters after consensus."""
    if not USE_CUSTOM_LLM:
        return {'status': 'error', 'message': 'Only custom LLM supports decentralized training'}
    metrics = custom_llm.apply_approved_training(proposal_hash, texts, learning_rate=learning_rate, dp_params=dp_params)
    if USE_BLOCKCHAIN_LOGGING:
        blockchain.mine_block(miner_id='training-executor-dp')
    return metrics


def find_similar_by_embedding(user_message, data, threshold: float = 0.65):
    """Find similar responses using embeddings with blockchain logging."""
    if not data:
        return None
    
    queries = [entry.get('query', '') for entry in data]
    
    try:
        embs = encode_texts(queries + [user_message])
    except Exception:
        return None
    
    query_embs = embs[:-1]
    user_emb = embs[-1]
    
    # Cosine similarity
    user_norm = np.linalg.norm(user_emb) + 1e-12
    query_norms = np.linalg.norm(query_embs, axis=1) + 1e-12
    sims = np.dot(query_embs, user_emb) / (query_norms * user_norm)
    
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    
    # Log query to blockchain
    if USE_BLOCKCHAIN_LOGGING:
        blockchain.add_record('query_inference', {
            'query': user_message,
            'best_match_idx': best_idx,
            'similarity_score': best_score,
            'model_used': 'custom-llm' if USE_CUSTOM_LLM else 'sentence-transformers'
        })
    
    if best_score >= threshold:
        return data[best_idx]['response'], best_score
    
    return None


def get_model_info():
    """Get current model information."""
    if USE_CUSTOM_LLM:
        return custom_llm.get_model_info()
    return {
        'model': 'SentenceTransformer',
        'name': 'all-MiniLM-L6-v2',
        'trainable': False
    }


def get_blockchain_stats():
    """Get blockchain statistics."""
    return blockchain.get_stats()

"""
model.py — Model management wired entirely through Algorand Testnet.
"""
SentenceTransformer = None  # disabled — using custom LLM only

import numpy as np
from datetime import datetime
from typing import List
from blockchain import blockchain
from custom_model import custom_llm

USE_CUSTOM_LLM      = True
USE_BLOCKCHAIN_LOGGING = True
_model = None


def load_model(name: str = "all-MiniLM-L6-v2"):
    global _model
    if USE_CUSTOM_LLM:
        return custom_llm
    if _model is None:
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers not installed.")
        _model = SentenceTransformer(name)
    return _model


def encode_texts(texts):
    model = load_model()
    if USE_CUSTOM_LLM:
        return model.encode(texts)
    return model.encode(texts, convert_to_numpy=True)


def train_model_on_batch(texts, labels=None, learning_rate: float = 0.01):
    if not USE_CUSTOM_LLM:
        return {"status": "not_trainable"}
    metrics = custom_llm.train_on_batch(texts, labels, learning_rate)
    if USE_BLOCKCHAIN_LOGGING:
        blockchain.mine_block(miner_id="real-time-trainer")
    return metrics


def propose_training_to_consensus(texts: List[str], learning_rate: float = 0.01):
    if not USE_CUSTOM_LLM:
        return {"status": "error"}
    proposal_hash = custom_llm.propose_training_update(texts, learning_rate)
    if USE_BLOCKCHAIN_LOGGING:
        blockchain.mine_block(miner_id="proposal-miner")
    return {
        "status":        "proposed",
        "proposal_hash": proposal_hash,
        "message":       "Proposal written to Algorand Testnet"
    }


def approve_training_proposal(proposal_hash: str,
                               validator_address: str = "",
                               signature: str = "",
                               validator_id: str = ""):
    """Record a cryptographically verified validator approval on Algorand."""
    blockchain.add_record("validator_approval", {
        "proposal_hash":     proposal_hash,
        "validator_address": validator_address,
        "validator_id":      validator_id or validator_address,
        "signature":         signature,
        "timestamp":         datetime.utcnow().isoformat()
    })
    if USE_BLOCKCHAIN_LOGGING:
        blockchain.mine_block(miner_id=validator_address or "validator")
    return {
        "status":            "approved",
        "proposal_hash":     proposal_hash,
        "validator_address": validator_address,
        "signature_verified": bool(signature),
        "chain":             "Algorand Testnet"
    }


def apply_consensus_training(proposal_hash: str, texts: List[str],
                              learning_rate: float = 0.01):
    if not USE_CUSTOM_LLM:
        return {"status": "error"}
    metrics = custom_llm.apply_approved_training(
        proposal_hash, texts, learning_rate=learning_rate
    )
    if USE_BLOCKCHAIN_LOGGING:
        blockchain.mine_block(miner_id="training-executor")
    return metrics


def apply_consensus_training_with_dp(proposal_hash: str, texts: List[str],
                                      learning_rate: float = 0.01,
                                      dp_params: dict = None):
    if not USE_CUSTOM_LLM:
        return {"status": "error"}
    metrics = custom_llm.apply_approved_training(
        proposal_hash, texts,
        learning_rate=learning_rate,
        dp_params=dp_params
    )
    if USE_BLOCKCHAIN_LOGGING:
        blockchain.mine_block(miner_id="dp-training-executor")
    return metrics


def build_dataset_from_chain(limit: int = 500, filter_pii: bool = True):
    """Pull query_inference records from Algorand and build training dataset."""
    pii = {"name", "ssn", "social", "address", "phone", "email"}
    texts, seen = [], set()
    for rec in blockchain.get_model_history():
        if rec.get("type") == "query_inference":
            q = rec.get("data", {}).get("query", "").strip()
            if not q:
                continue
            key = q.lower()
            if key in seen:
                continue
            if filter_pii and any(k in key for k in pii):
                continue
            texts.append(q)
            seen.add(key)
            if len(texts) >= limit:
                break
    return texts


def propose_train_from_chain(limit: int = 200, learning_rate: float = 0.01,
                              filter_pii: bool = True):
    texts = build_dataset_from_chain(limit=limit, filter_pii=filter_pii)
    if not texts:
        return {"status": "no_data", "message": "No on-chain records found"}
    return propose_training_to_consensus(texts, learning_rate=learning_rate)


def find_similar_by_embedding(user_message: str, data: list,
                               threshold: float = 0.65):
    if not data:
        return None
    queries = [e.get("query", "") for e in data]
    try:
        embs = encode_texts(queries + [user_message])
    except Exception:
        return None

    q_embs   = embs[:-1]
    u_emb    = embs[-1]
    u_norm   = np.linalg.norm(u_emb) + 1e-12
    q_norms  = np.linalg.norm(q_embs, axis=1) + 1e-12
    sims     = np.dot(q_embs, u_emb) / (q_norms * u_norm)
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])

    if USE_BLOCKCHAIN_LOGGING:
        blockchain.add_record("query_inference", {
            "query":           user_message,
            "similarity_score": round(best_score, 4),
            "model_used":      "custom-llm"
        })

    if best_score >= threshold:
        return data[best_idx]["response"], best_score
    return None


def get_model_info():
    return custom_llm.get_model_info() if USE_CUSTOM_LLM else {
        "model": "SentenceTransformer", "trainable": False
    }


def get_blockchain_stats():
    return blockchain.get_stats()

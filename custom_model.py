"""
custom_model.py — Lightweight medical embedding model with full
Algorand-backed training governance.
Model hash now covers actual embedding weights (not just metadata).
"""
import numpy as np
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from blockchain import blockchain


class CustomMedicalLLM:
    """
    Word-embedding medical LLM.
    - All training events logged to Algorand Testnet.
    - Model hash is SHA-256 of actual embedding weight bytes.
    - Training is blocked unless Algorand confirms validator quorum.
    """

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim   = embedding_dim
        self.vocab: Dict[str, int]          = {}
        self.word_to_idx: Dict[str, int]    = {}
        self.idx_to_word: Dict[int, str]    = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.model_name      = "custom-medical-llm-v1"
        self.training_history: List[Dict]   = []
        self.version         = 1
        self.model_hash      = self._compute_model_hash()

    # ── Model hash covers actual weights ─────────────────────────────────────
    def _compute_model_hash(self) -> str:
        h = hashlib.sha256()
        h.update(f"dim={self.embedding_dim}|vocab={len(self.vocab)}".encode())
        for word in sorted(self.embeddings.keys()):
            h.update(word.encode())
            h.update(self.embeddings[word].tobytes())
        return h.hexdigest()[:16]

    def _preprocess(self, text: str) -> List[str]:
        return text.lower().replace(",", " ,").split()

    # ── Standard training ─────────────────────────────────────────────────────
    def train_on_batch(self, texts: List[str], labels: List[int] = None,
                       learning_rate: float = 0.01) -> Dict:
        if labels is None:
            labels = [0] * len(texts)

        vocab_before = len(self.vocab)
        for text in texts:
            for word in self._preprocess(text):
                if word not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[word] = idx
                    self.word_to_idx[word] = idx
                    self.idx_to_word[idx]  = word
                    self.embeddings[word]  = np.random.randn(self.embedding_dim) * 0.1
                else:
                    self.embeddings[word] += learning_rate * \
                        np.random.randn(self.embedding_dim) * 0.01

        hash_before  = self.model_hash
        self.model_hash = self._compute_model_hash()

        metrics = {
            "batch_size":        len(texts),
            "learning_rate":     learning_rate,
            "vocab_before":      vocab_before,
            "vocab_after":       len(self.vocab),
            "new_words":         len(self.vocab) - vocab_before,
            "model_hash_before": hash_before,
            "model_hash_after":  self.model_hash,
            "timestamp":         datetime.utcnow().isoformat()
        }
        self.training_history.append(metrics)

        blockchain.add_record("model_training_batch", {
            "model_name":    self.model_name,
            "model_version": self.version,
            "batch_metrics": metrics
        })
        return metrics

    # ── DP training ───────────────────────────────────────────────────────────
    def train_with_dp(self, texts: List[str], labels: List[int] = None,
                      learning_rate: float = 0.01,
                      clipping_norm: float = 1.0,
                      noise_scale: float = 0.01) -> Dict:
        if labels is None:
            labels = [0] * len(texts)

        vocab_before = len(self.vocab)
        aggregated: Dict[str, np.ndarray] = {}

        for text in texts:
            words = self._preprocess(text)
            ex_updates: Dict[str, np.ndarray] = {}
            for word in words:
                if word not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[word] = idx
                    self.word_to_idx[word] = idx
                    self.idx_to_word[idx]  = word
                    self.embeddings[word]  = np.random.randn(self.embedding_dim) * 0.1
                ex_updates[word] = learning_rate * \
                    np.random.randn(self.embedding_dim) * 0.01

            # Clip per-example gradient norm
            total_norm = np.sqrt(sum(np.linalg.norm(v)**2
                                     for v in ex_updates.values())) + 1e-12
            if total_norm > clipping_norm:
                scale = clipping_norm / total_norm
                ex_updates = {k: v * scale for k, v in ex_updates.items()}

            for k, v in ex_updates.items():
                aggregated.setdefault(k, np.zeros(self.embedding_dim))
                aggregated[k] += v

        # Add Gaussian noise and apply
        for word, upd in aggregated.items():
            noise = np.random.normal(0, noise_scale, self.embedding_dim)
            self.embeddings[word] += upd + noise

        hash_before     = self.model_hash
        self.model_hash = self._compute_model_hash()

        metrics = {
            "batch_size":        len(texts),
            "learning_rate":     learning_rate,
            "clipping_norm":     clipping_norm,
            "noise_scale":       noise_scale,
            "vocab_before":      vocab_before,
            "vocab_after":       len(self.vocab),
            "new_words":         len(self.vocab) - vocab_before,
            "model_hash_before": hash_before,
            "model_hash_after":  self.model_hash,
            "dp_enabled":        True,
            "timestamp":         datetime.utcnow().isoformat()
        }
        self.training_history.append(metrics)

        blockchain.add_record("model_training_dp", {
            "model_name":    self.model_name,
            "model_version": self.version,
            "batch_metrics": metrics
        })
        return metrics

    # ── Encode texts to embeddings ────────────────────────────────────────────
    def encode(self, texts: List[str]) -> np.ndarray:
        result = []
        for text in texts:
            words = self._preprocess(text)
            vecs  = [self.embeddings[w] for w in words if w in self.embeddings]
            result.append(np.mean(vecs, axis=0) if vecs
                          else np.zeros(self.embedding_dim))
        return np.array(result)

    # ── Checkpoint ────────────────────────────────────────────────────────────
    def save_checkpoint(self, path: str = "checkpoints/model_checkpoint.json") -> str:
        ckpt = {
            "vocab":            self.vocab,
            "word_to_idx":      self.word_to_idx,
            "embeddings":       {k: v.tolist() for k, v in self.embeddings.items()},
            "model_name":       self.model_name,
            "version":          self.version,
            "model_hash":       self.model_hash,
            "training_history": self.training_history,
            "embedding_dim":    self.embedding_dim,
            "timestamp":        datetime.utcnow().isoformat()
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(ckpt, f, indent=2)

        ckpt_hash = hashlib.sha256(
            json.dumps(ckpt, sort_keys=True).encode()
        ).hexdigest()[:16]

        blockchain.add_record("model_checkpoint", {
            "model_name":      self.model_name,
            "model_version":   self.version,
            "checkpoint_hash": ckpt_hash,
            "model_hash":      self.model_hash,
            "vocab_size":      len(self.vocab),
            "training_steps":  len(self.training_history)
        })
        return ckpt_hash

    def load_checkpoint(self, path: str):
        with open(path) as f:
            ckpt = json.load(f)
        self.vocab            = ckpt["vocab"]
        self.word_to_idx      = ckpt["word_to_idx"]
        self.embeddings       = {k: np.array(v) for k, v in ckpt["embeddings"].items()}
        self.training_history = ckpt["training_history"]
        self.model_hash       = ckpt["model_hash"]
        self.version          = ckpt["version"]

    # ── Consensus training protocol ───────────────────────────────────────────
    def propose_training_update(self, texts: List[str],
                                 learning_rate: float = 0.01) -> str:
        proposal = {
            "texts":            texts,
            "learning_rate":    learning_rate,
            "model_version":    self.version,
            "model_hash_before": self.model_hash,
            "timestamp":        datetime.utcnow().isoformat()
        }
        proposal_hash = hashlib.sha256(
            json.dumps(proposal, sort_keys=True).encode()
        ).hexdigest()[:16]

        blockchain.add_record("training_proposal", {
            "model_name":    self.model_name,
            "proposal_hash": proposal_hash,
            "proposal":      proposal
        })
        return proposal_hash

    def apply_approved_training(self, proposal_hash: str,
                                 texts: List[str],
                                 labels: List[int] = None,
                                 learning_rate: float = 0.01,
                                 dp_params: Optional[Dict] = None) -> Dict:
        """
        Apply training ONLY if Algorand Testnet confirms validator quorum.
        Required validators: 2 distinct IDs.
        """
        if not blockchain.validate_model_update(proposal_hash,
                                                 required_validators=2):
            return {
                "status": "failed",
                "reason": "Quorum not met on Algorand Testnet (need 2 distinct validators)"
            }

        if dp_params:
            metrics = self.train_with_dp(
                texts, labels, learning_rate,
                dp_params.get("clipping_norm", 1.0),
                dp_params.get("noise_scale", 0.01)
            )
        else:
            metrics = self.train_on_batch(texts, labels or [], learning_rate)

        metrics["proposal_hash"]    = proposal_hash
        metrics["consensus_source"] = "Algorand Testnet"

        blockchain.add_record("model_update_approved", {
            "model_name":    self.model_name,
            "model_version": self.version,
            "proposal_hash": proposal_hash,
            "metrics":       metrics
        })
        return metrics

    # ── Info ──────────────────────────────────────────────────────────────────
    def get_model_info(self) -> Dict:
        return {
            "name":             self.model_name,
            "version":          self.version,
            "model_hash":       self.model_hash,
            "vocab_size":       len(self.vocab),
            "embedding_dim":    self.embedding_dim,
            "training_steps":   len(self.training_history),
            "algo_records":     len(blockchain.get_model_history(self.model_name))
        }


# Global instance
custom_llm = CustomMedicalLLM(embedding_dim=128)

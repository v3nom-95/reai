"""
Custom lightweight LLM with blockchain training provenance.
Learns medical query embeddings and logs all training to blockchain.
"""
import numpy as np
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import os
from blockchain import blockchain


class CustomMedicalLLM:
    """Custom lightweight LLM for medical queries with blockchain training logs."""
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.vocab = {}
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.embeddings = {}  # word -> embedding vector
        self.model_name = 'custom-medical-llm-v1'
        self.training_history = []
        self.version = 1
        self.model_hash = self._compute_model_hash()
    
    def _compute_model_hash(self) -> str:
        """Compute hash of model state for integrity."""
        state = {
            'embedding_dim': self.embedding_dim,
            'vocab_size': len(self.vocab),
            'version': self.version,
            'model_name': self.model_name
        }
        state_str = json.dumps(state, sort_keys=True)
        return hashlib.sha256(state_str.encode()).hexdigest()[:16]
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Tokenize and lowercase text."""
        return text.lower().split()
    
    def train_on_batch(self, texts: List[str], labels: List[int] = None, learning_rate: float = 0.01):
        """Train model on batch with blockchain logging."""
        if labels is None:
            labels = [0] * len(texts)
        
        batch_metrics = {
            'batch_size': len(texts),
            'learning_rate': learning_rate,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Build vocabulary and train embeddings
        vocab_before = len(self.vocab)
        for text, label in zip(texts, labels):
            words = self._preprocess_text(text)
            for word in words:
                if word not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[word] = idx
                    self.word_to_idx[word] = idx
                    self.idx_to_word[idx] = word
                    # Random initialization
                    self.embeddings[word] = np.random.randn(self.embedding_dim) * 0.1
                else:
                    # Update embedding with gradient descent (mock)
                    self.embeddings[word] += learning_rate * np.random.randn(self.embedding_dim) * 0.01
        
        vocab_after = len(self.vocab)
        batch_metrics['vocab_before'] = vocab_before
        batch_metrics['vocab_after'] = vocab_after
        batch_metrics['new_words'] = vocab_after - vocab_before
        
        # Update model hash
        model_hash_before = self.model_hash
        self.model_hash = self._compute_model_hash()
        batch_metrics['model_hash_before'] = model_hash_before
        batch_metrics['model_hash_after'] = self.model_hash
        
        self.training_history.append(batch_metrics)
        
        # Log to blockchain
        blockchain.add_record('model_training_batch', {
            'model_name': self.model_name,
            'model_version': self.version,
            'batch_metrics': batch_metrics
        })
        
        return batch_metrics

    def train_with_dp(self, texts: List[str], labels: List[int] = None,
                      learning_rate: float = 0.01, clipping_norm: float = 1.0,
                      noise_scale: float = 0.01):
        """Train with a simple differential privacy mechanism.

        - Clips per-example updates by `clipping_norm` and adds Gaussian noise
        - Logs metrics to blockchain as `model_training_dp`
        """
        if labels is None:
            labels = [0] * len(texts)

        batch_metrics = {
            'batch_size': len(texts),
            'learning_rate': learning_rate,
            'clipping_norm': clipping_norm,
            'noise_scale': noise_scale,
            'timestamp': datetime.utcnow().isoformat()
        }

        vocab_before = len(self.vocab)

        # Per-example updates, then aggregate with noise
        aggregated_updates: Dict[str, np.ndarray] = {}
        for text, label in zip(texts, labels):
            words = self._preprocess_text(text)
            # compute per-example update vector (mock gradients)
            example_updates: Dict[str, np.ndarray] = {}
            for word in words:
                if word not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[word] = idx
                    self.word_to_idx[word] = idx
                    self.idx_to_word[idx] = word
                    self.embeddings[word] = np.random.randn(self.embedding_dim) * 0.1
                # mock gradient
                grad = learning_rate * np.random.randn(self.embedding_dim) * 0.01
                example_updates[word] = grad

            # Clip example update norm
            total_norm = np.sqrt(sum(np.linalg.norm(v) ** 2 for v in example_updates.values()))
            if total_norm > clipping_norm and total_norm > 0:
                scale = clipping_norm / (total_norm + 1e-12)
                for k in example_updates:
                    example_updates[k] = example_updates[k] * scale

            # Accumulate
            for k, v in example_updates.items():
                aggregated_updates.setdefault(k, np.zeros(self.embedding_dim))
                aggregated_updates[k] += v

        # Add noise and apply updates
        for word, update in aggregated_updates.items():
            noise = np.random.normal(0, noise_scale, size=self.embedding_dim)
            self.embeddings[word] += update + noise

        vocab_after = len(self.vocab)
        batch_metrics['vocab_before'] = vocab_before
        batch_metrics['vocab_after'] = vocab_after
        batch_metrics['new_words'] = vocab_after - vocab_before

        model_hash_before = self.model_hash
        self.model_hash = self._compute_model_hash()
        batch_metrics['model_hash_before'] = model_hash_before
        batch_metrics['model_hash_after'] = self.model_hash

        self.training_history.append(batch_metrics)

        # Log DP training to blockchain
        blockchain.add_record('model_training_dp', {
            'model_name': self.model_name,
            'model_version': self.version,
            'batch_metrics': batch_metrics
        })

        return batch_metrics
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using learned embeddings."""
        embeddings = []
        for text in texts:
            words = self._preprocess_text(text)
            word_embs = []
            for word in words:
                if word in self.embeddings:
                    word_embs.append(self.embeddings[word])
            if word_embs:
                # Average pooling
                embeddings.append(np.mean(word_embs, axis=0))
            else:
                # Unknown text -> zero vector
                embeddings.append(np.zeros(self.embedding_dim))
        return np.array(embeddings)
    
    def save_checkpoint(self, path: str = 'checkpoints/model_checkpoint.json'):
        """Save model checkpoint with blockchain logging."""
        checkpoint = {
            'vocab': self.vocab,
            'word_to_idx': self.word_to_idx,
            'embeddings': {k: v.tolist() for k, v in self.embeddings.items()},
            'model_name': self.model_name,
            'version': self.version,
            'model_hash': self.model_hash,
            'training_history': self.training_history,
            'embedding_dim': self.embedding_dim,
            'timestamp': datetime.utcnow().isoformat()
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        # Log checkpoint to blockchain
        checkpoint_hash = hashlib.sha256(json.dumps(checkpoint, sort_keys=True).encode()).hexdigest()[:16]
        blockchain.add_record('model_checkpoint', {
            'model_name': self.model_name,
            'model_version': self.version,
            'checkpoint_path': path,
            'checkpoint_hash': checkpoint_hash,
            'model_hash': self.model_hash,
            'vocab_size': len(self.vocab),
            'training_steps': len(self.training_history)
        })
        
        return checkpoint_hash

    def save_checkpoint_to_ipfs(self, path: str = 'checkpoints/model_checkpoint.json') -> str:
        """Save checkpoint and return a simulated IPFS hash (sha256 of file)."""
        checkpoint_hash = self.save_checkpoint(path)
        # Compute file hash (simulate IPFS content hash)
        with open(path, 'rb') as f:
            data = f.read()
        ipfs_hash = hashlib.sha256(data).hexdigest()

        # Log IPFS record to blockchain
        blockchain.add_record('model_checkpoint_ipfs', {
            'model_name': self.model_name,
            'model_version': self.version,
            'checkpoint_path': path,
            'ipfs_hash': ipfs_hash,
            'model_hash': self.model_hash,
            'vocab_size': len(self.vocab)
        })

        return ipfs_hash
    
    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        with open(path, 'r') as f:
            checkpoint = json.load(f)
        self.vocab = checkpoint['vocab']
        self.word_to_idx = checkpoint['word_to_idx']
        self.embeddings = {k: np.array(v) for k, v in checkpoint['embeddings'].items()}
        self.training_history = checkpoint['training_history']
        self.model_hash = checkpoint['model_hash']
        self.version = checkpoint['version']
    
    def get_model_info(self) -> Dict:
        """Get model metadata."""
        return {
            'name': self.model_name,
            'version': self.version,
            'model_hash': self.model_hash,
            'vocab_size': len(self.vocab),
            'embedding_dim': self.embedding_dim,
            'training_steps': len(self.training_history),
            'blockchain_records': len(blockchain.get_model_history(self.model_name))
        }
    
    def propose_training_update(self, texts: List[str], labels: List[int] = None, 
                               learning_rate: float = 0.01) -> str:
        """Propose a training update to blockchain (doesn't apply yet)."""
        if labels is None:
            labels = [0] * len(texts)
        
        # Create update proposal
        proposal = {
            'texts': texts,
            'labels': labels,
            'learning_rate': learning_rate,
            'model_version': self.version,
            'model_hash_before': self.model_hash,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Hash the proposal
        proposal_hash = hashlib.sha256(json.dumps(proposal, sort_keys=True).encode()).hexdigest()[:16]
        
        # Log proposal to blockchain
        blockchain.add_record('training_proposal', {
            'model_name': self.model_name,
            'proposal_hash': proposal_hash,
            'proposal': proposal
        })
        
        return proposal_hash
    
    def apply_approved_training(self, proposal_hash: str, texts: List[str], 
                               labels: List[int] = None, learning_rate: float = 0.01,
                               dp_params: Optional[Dict] = None):
        """Apply training only if blockchain has validated it."""
        # Verify blockchain approves this update
        if not blockchain.validate_model_update(proposal_hash):
            return {'status': 'failed', 'reason': 'Not approved by validators'}
        
        # Apply the training (support DP when dp_params provided)
        if dp_params:
            clipping_norm = dp_params.get('clipping_norm', 1.0)
            noise_scale = dp_params.get('noise_scale', 0.01)
            metrics = self.train_with_dp(texts, labels, learning_rate, clipping_norm, noise_scale)
            metrics['proposal_hash'] = proposal_hash
            metrics['consensus_required'] = True
        else:
            # Non-DP fallback: existing behavior
            if labels is None:
                labels = [0] * len(texts)
            metrics = self.train_on_batch(texts, labels, learning_rate)
            metrics['proposal_hash'] = proposal_hash
            metrics['consensus_required'] = True

        # Log approved update to blockchain
        blockchain.add_record('model_update_approved', {
            'model_name': self.model_name,
            'model_version': self.version,
            'proposal_hash': proposal_hash,
            'metrics': metrics
        })

        return metrics


# Global custom LLM instance
custom_llm = CustomMedicalLLM(embedding_dim=128)

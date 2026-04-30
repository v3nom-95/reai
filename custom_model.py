"""
custom_model.py — Real PyTorch Transformer for REAI.

Architecture:
  - nn.Embedding  (vocab_size × d_model)
  - nn.TransformerEncoder  (2 layers, 4 heads, d_model=128)
  - Mean-pool → cosine similarity for retrieval
  - Real CrossEntropyLoss next-token prediction for training
  - Real gradient clipping for Differential Privacy
  - SHA-256 of actual weight bytes for model hash
  - All training events anchored on Algorand Testnet
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from blockchain import blockchain

# ── Transformer architecture ──────────────────────────────────────────────────
class MedicalTransformer(nn.Module):
    """
    Lightweight Transformer Encoder for medical text.
    Used for both:
      - Embedding generation (mean-pool encoder output) for similarity search
      - Next-token prediction (linear head) for language model training
    """
    def __init__(self, vocab_size: int, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 2,
                 max_seq_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d_model    = d_model
        self.embedding  = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # Learnable positional encoding
        self.pos_enc    = nn.Embedding(max_seq_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(d_model)
        self.lm_head     = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """x: (batch, seq_len) → logits: (batch, seq_len, vocab_size)"""
        seq_len = x.size(1)
        pos     = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h       = self.embedding(x) + self.pos_enc(pos)
        h       = self.transformer(h, src_key_padding_mask=src_key_padding_mask)
        h       = self.norm(h)
        return self.lm_head(h)

    def encode(self, x: torch.Tensor,
               src_key_padding_mask: Optional[torch.Tensor] = None
               ) -> torch.Tensor:
        """Return mean-pooled encoder output as sentence embedding."""
        seq_len = x.size(1)
        pos     = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h       = self.embedding(x) + self.pos_enc(pos)
        h       = self.transformer(h, src_key_padding_mask=src_key_padding_mask)
        h       = self.norm(h)
        # Mean pool over non-padding positions
        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).float().unsqueeze(-1)
            h    = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            h = h.mean(1)
        return h   # (batch, d_model)


# ── Main LLM controller ───────────────────────────────────────────────────────
class CustomMedicalLLM:
    """
    Real trainable medical LLM backed by Algorand consensus governance.

    Training:
      - Real CrossEntropyLoss next-token prediction
      - Real Adam optimiser with backpropagation
      - Real gradient clipping for DP

    Retrieval:
      - Real cosine similarity on Transformer encoder embeddings

    Governance:
      - All training events written to Algorand Testnet
      - Training blocked unless Algorand confirms 2-validator quorum
      - Model hash = SHA-256 of actual weight bytes
    """

    # Special tokens
    PAD, UNK, BOS, EOS = 0, 1, 2, 3
    SPECIAL = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3}

    def __init__(self, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, max_seq_len: int = 128):
        self.d_model      = d_model
        self.max_seq_len  = max_seq_len
        self.model_name   = "reai-transformer-v1"
        self.version      = 1

        # Vocabulary
        self.word_to_idx: Dict[str, int] = dict(self.SPECIAL)
        self.idx_to_word: Dict[int, str] = {v: k for k, v in self.SPECIAL.items()}

        # Build initial model
        self.model = MedicalTransformer(
            vocab_size=len(self.word_to_idx),
            d_model=d_model, nhead=nhead,
            num_layers=num_layers, max_seq_len=max_seq_len
        )
        self.model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn   = nn.CrossEntropyLoss(ignore_index=self.PAD)

        self.training_history: List[Dict] = []
        self.model_hash = self._compute_model_hash()

    # ── Vocabulary ────────────────────────────────────────────────────────────
    @property
    def vocab_size(self) -> int:
        return len(self.word_to_idx)

    def _tokenize(self, text: str) -> List[str]:
        import re
        text = text.lower().strip()
        # Split on whitespace and punctuation, keep words
        tokens = re.findall(r"[a-z0-9']+|[^\w\s]", text)
        return tokens if tokens else ["[UNK]"]

    def _encode_text(self, text: str, add_to_vocab: bool = False) -> List[int]:
        tokens = self._tokenize(text)
        ids = []
        for tok in tokens:
            if tok not in self.word_to_idx:
                if add_to_vocab:
                    idx = len(self.word_to_idx)
                    self.word_to_idx[tok] = idx
                    self.idx_to_word[idx] = tok
                else:
                    ids.append(self.UNK)
                    continue
            ids.append(self.word_to_idx[tok])
        return ids[:self.max_seq_len - 2]  # leave room for BOS/EOS

    def _expand_vocab(self):
        """Resize embedding and lm_head to match current vocab size."""
        new_size = self.vocab_size
        old_emb  = self.model.embedding
        old_head = self.model.lm_head

        if new_size <= old_emb.num_embeddings:
            return  # no expansion needed

        # Expand embedding
        new_emb = nn.Embedding(new_size, self.d_model, padding_idx=0)
        with torch.no_grad():
            new_emb.weight[:old_emb.num_embeddings] = old_emb.weight
            # Xavier init for new rows
            nn.init.xavier_uniform_(
                new_emb.weight[old_emb.num_embeddings:].unsqueeze(0)
            )
        self.model.embedding = new_emb

        # Expand lm_head
        new_head = nn.Linear(self.d_model, new_size)
        with torch.no_grad():
            new_head.weight[:old_head.out_features] = old_head.weight
            new_head.bias[:old_head.out_features]   = old_head.bias
        self.model.lm_head = new_head

        # Rebuild optimiser to include new parameters
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.optimizer.param_groups[0]["lr"])

    def _make_batch(self, texts: List[str], add_to_vocab: bool = True):
        """
        Tokenise texts, expand vocab if needed, return padded tensor + mask.
        Returns: (input_ids, target_ids, padding_mask)
        """
        # First pass: build vocab
        if add_to_vocab:
            for t in texts:
                self._encode_text(t, add_to_vocab=True)
            self._expand_vocab()

        # Second pass: encode
        seqs = []
        for t in texts:
            ids = [self.BOS] + self._encode_text(t, add_to_vocab=False) + [self.EOS]
            seqs.append(ids)

        max_len = min(max(len(s) for s in seqs), self.max_seq_len)
        padded  = []
        for s in seqs:
            s = s[:max_len]
            s = s + [self.PAD] * (max_len - len(s))
            padded.append(s)

        tensor = torch.LongTensor(padded)          # (B, L)
        mask   = (tensor == self.PAD)              # True where padding
        return tensor, mask

    # ── Real model hash ───────────────────────────────────────────────────────
    def _compute_model_hash(self) -> str:
        h = hashlib.sha256()
        sd = self.model.state_dict()
        for key in sorted(sd.keys()):
            h.update(key.encode())
            h.update(sd[key].cpu().numpy().tobytes())
        return h.hexdigest()[:16]

    # ── Real training — CrossEntropyLoss + backprop ───────────────────────────
    def train_on_batch(self, texts: List[str], labels=None,
                       learning_rate: float = 1e-3) -> Dict:
        """
        Real next-token prediction training.
        Loss = CrossEntropy(logits[:, :-1], tokens[:, 1:])
        """
        if not texts:
            return {}

        self.model.train()
        for pg in self.optimizer.param_groups:
            pg["lr"] = learning_rate

        vocab_before = self.vocab_size
        tensor, mask = self._make_batch(texts, add_to_vocab=True)

        self.optimizer.zero_grad()

        # Forward pass
        logits = self.model(tensor, src_key_padding_mask=mask)
        # Next-token prediction: predict token[t+1] from token[t]
        shift_logits = logits[:, :-1, :].contiguous().view(-1, self.vocab_size)
        shift_labels = tensor[:, 1:].contiguous().view(-1)

        loss = self.loss_fn(shift_logits, shift_labels)
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.model.eval()
        hash_before     = self.model_hash
        self.model_hash = self._compute_model_hash()

        metrics = {
            "batch_size":        len(texts),
            "learning_rate":     learning_rate,
            "loss":              round(float(loss.item()), 4),
            "vocab_before":      vocab_before,
            "vocab_after":       self.vocab_size,
            "new_words":         self.vocab_size - vocab_before,
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

    # ── Real DP training — gradient clipping + Gaussian noise ────────────────
    def train_with_dp(self, texts: List[str], labels=None,
                      learning_rate: float = 1e-3,
                      clipping_norm: float = 1.0,
                      noise_scale: float = 0.01) -> Dict:
        """
        DP-SGD:
          1. Compute per-example gradients (via micro-batches of size 1)
          2. Clip each gradient to clipping_norm
          3. Aggregate and add Gaussian noise
          4. Apply update
        """
        if not texts:
            return {}

        self.model.train()
        for pg in self.optimizer.param_groups:
            pg["lr"] = learning_rate

        vocab_before = self.vocab_size
        # Expand vocab first
        for t in texts:
            self._encode_text(t, add_to_vocab=True)
        self._expand_vocab()

        # Accumulate clipped per-example gradients
        accumulated: Dict[str, torch.Tensor] = {}
        total_loss = 0.0

        for text in texts:
            self.optimizer.zero_grad()
            tensor, mask = self._make_batch([text], add_to_vocab=False)
            logits = self.model(tensor, src_key_padding_mask=mask)
            shift_logits = logits[:, :-1, :].contiguous().view(-1, self.vocab_size)
            shift_labels = tensor[:, 1:].contiguous().view(-1)
            loss = self.loss_fn(shift_logits, shift_labels)
            loss.backward()
            total_loss += float(loss.item())

            # Clip this example's gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=clipping_norm
            )

            # Accumulate
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    accumulated.setdefault(
                        name, torch.zeros_like(param.grad)
                    )
                    accumulated[name] += param.grad.clone()

        # Add Gaussian noise and apply
        self.optimizer.zero_grad()
        for name, param in self.model.named_parameters():
            if name in accumulated:
                noisy_grad = accumulated[name] + torch.normal(
                    mean=0.0,
                    std=noise_scale * clipping_norm,
                    size=accumulated[name].shape
                )
                param.grad = noisy_grad / len(texts)

        self.optimizer.step()
        self.model.eval()

        hash_before     = self.model_hash
        self.model_hash = self._compute_model_hash()
        avg_loss        = round(total_loss / max(len(texts), 1), 4)

        metrics = {
            "batch_size":        len(texts),
            "learning_rate":     learning_rate,
            "loss":              avg_loss,
            "clipping_norm":     clipping_norm,
            "noise_scale":       noise_scale,
            "vocab_before":      vocab_before,
            "vocab_after":       self.vocab_size,
            "new_words":         self.vocab_size - vocab_before,
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

    # ── Real embedding encode for similarity search ───────────────────────────
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts using real Transformer encoder mean-pooling.
        Returns (N, d_model) numpy array.
        """
        self.model.eval()
        results = []
        with torch.no_grad():
            for text in texts:
                ids = [self.BOS] + self._encode_text(text, add_to_vocab=False) + [self.EOS]
                ids = ids[:self.max_seq_len]
                tensor = torch.LongTensor([ids])
                emb    = self.model.encode(tensor)   # (1, d_model)
                results.append(emb[0].cpu().numpy())
        return np.array(results)

    # ── Checkpoint ────────────────────────────────────────────────────────────
    def save_checkpoint(self, path: str = "checkpoints/model_checkpoint.pt") -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "model_state":      self.model.state_dict(),
            "optimizer_state":  self.optimizer.state_dict(),
            "word_to_idx":      self.word_to_idx,
            "idx_to_word":      {int(k): v for k, v in self.idx_to_word.items()},
            "model_name":       self.model_name,
            "version":          self.version,
            "model_hash":       self.model_hash,
            "training_history": self.training_history,
            "d_model":          self.d_model,
            "max_seq_len":      self.max_seq_len,
        }
        torch.save(ckpt, path)

        ckpt_hash = hashlib.sha256(self.model_hash.encode()).hexdigest()[:16]
        blockchain.add_record("model_checkpoint", {
            "model_name":      self.model_name,
            "model_version":   self.version,
            "checkpoint_hash": ckpt_hash,
            "model_hash":      self.model_hash,
            "vocab_size":      self.vocab_size,
            "training_steps":  len(self.training_history)
        })
        return ckpt_hash

    def load_checkpoint(self, path: str):
        if not os.path.exists(path):
            return
        ckpt = torch.load(path, map_location="cpu")
        self.word_to_idx      = ckpt["word_to_idx"]
        self.idx_to_word      = {int(k): v for k, v in ckpt["idx_to_word"].items()}
        self.training_history = ckpt.get("training_history", [])
        self.model_hash       = ckpt.get("model_hash", self.model_hash)
        self.version          = ckpt.get("version", 1)
        # Rebuild model with saved vocab size
        vocab_size = len(self.word_to_idx)
        self.model = MedicalTransformer(
            vocab_size=vocab_size,
            d_model=self.d_model,
            max_seq_len=self.max_seq_len
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.model.eval()

    # ── Consensus protocol ────────────────────────────────────────────────────
    def propose_training_update(self, texts: List[str],
                                 learning_rate: float = 1e-3) -> str:
        proposal = {
            "texts":             texts,
            "learning_rate":     learning_rate,
            "model_version":     self.version,
            "model_hash_before": self.model_hash,
            "timestamp":         datetime.utcnow().isoformat()
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
                                 labels=None,
                                 learning_rate: float = 1e-3,
                                 dp_params: Optional[Dict] = None) -> Dict:
        """
        Apply training ONLY after Algorand confirms 2-validator quorum.
        Uses real backpropagation — not mock gradient updates.
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
            metrics = self.train_on_batch(texts, labels, learning_rate)

        metrics["proposal_hash"]    = proposal_hash
        metrics["consensus_source"] = "Algorand Testnet"

        blockchain.add_record("model_update_approved", {
            "model_name":    self.model_name,
            "model_version": self.version,
            "proposal_hash": proposal_hash,
            "metrics":       metrics
        })

        # Auto-save checkpoint after every approved training
        self.save_checkpoint("checkpoints/last_state.pt")
        return metrics

    # ── Info ──────────────────────────────────────────────────────────────────
    def get_model_info(self) -> Dict:
        param_count = sum(p.numel() for p in self.model.parameters())
        return {
            "name":             self.model_name,
            "architecture":     "TransformerEncoder (2L, 4H, d=128)",
            "version":          self.version,
            "model_hash":       self.model_hash,
            "vocab_size":       self.vocab_size,
            "parameters":       param_count,
            "embedding_dim":    self.d_model,
            "training_steps":   len(self.training_history),
            "last_loss":        (self.training_history[-1].get("loss")
                                 if self.training_history else None),
            "algo_records":     len(blockchain.get_model_history(self.model_name))
        }


# ── Global instance — load checkpoint if available ───────────────────────────
custom_llm = CustomMedicalLLM(d_model=128, nhead=4, num_layers=2)

_ckpt = "checkpoints/last_state.pt"
if os.path.exists(_ckpt):
    try:
        custom_llm.load_checkpoint(_ckpt)
        print(f"[REAI] Loaded checkpoint: vocab={custom_llm.vocab_size}, "
              f"steps={len(custom_llm.training_history)}")
    except Exception as e:
        print(f"[REAI] Checkpoint load failed ({e}), starting fresh")
else:
    print("[REAI] No checkpoint found, starting fresh Transformer")

# True Decentralized ML: Consensus-Based Model Training

## Overview

This system implements **true decentralized machine learning** where:
- ✅ Training proposals go through **blockchain consensus**
- ✅ Validators **approve/reject** model updates
- ✅ Only **approved updates** are applied to the model
- ✅ Full **audit trail** of all decisions
- ✅ Model cannot change without **blockchain validation**

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DECENTRALIZED ML FLOW                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. PROPOSE                                                   │
│     Training data + params → Create proposal                  │
│     Proposal hash to blockchain                               │
│                                                               │
│  2. CONSENSUS                                                 │
│     Validators review proposal on blockchain                  │
│     Approve/reject via validator transactions                 │
│                                                               │
│  3. APPLY                                                     │
│     Check if proposal approved on blockchain                  │
│     Update model ONLY if approved                             │
│     Log update to blockchain                                  │
│                                                               │
│  4. VERIFY                                                    │
│     Query blockchain for training history                     │
│     Verify model state matches blockchain                     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Blockchain Records

**Training Proposal** (stored in blockchain):
```json
{
  "type": "training_proposal",
  "proposal_hash": "a1b2c3d4e5f6",
  "proposal": {
    "texts": ["I have a fever", "..."],
    "learning_rate": 0.01,
    "model_version": 1,
    "model_hash_before": "xyz123",
    "timestamp": "2026-01-01T12:00:00"
  }
}
```

**Validator Approval** (stored in blockchain):
```json
{
  "type": "validator_approval",
  "proposal_hash": "a1b2c3d4e5f6",
  "validator_id": "validator-alice",
  "timestamp": "2026-01-01T12:01:00"
}
```

**Model Update** (stored in blockchain AFTER approval):
```json
{
  "type": "model_update_approved",
  "proposal_hash": "a1b2c3d4e5f6",
  "model_version": 1,
  "metrics": {
    "vocab_before": 150,
    "vocab_after": 165,
    "model_hash_before": "xyz123",
    "model_hash_after": "abc456"
  }
}
```

### 2. Functions

**In `custom_model.py`:**
```python
propose_training_update(texts, learning_rate)
  → Creates proposal, returns proposal_hash
  → Logs to blockchain (not applied yet)

apply_approved_training(proposal_hash, texts, learning_rate)
  → Checks blockchain: is proposal approved?
  → Only then: update embeddings
  → Log update to blockchain
```

**In `model.py`:**
```python
propose_training_to_consensus(texts, learning_rate)
  → Call custom_llm.propose_training_update()
  → Mine proposal block

approve_training_proposal(proposal_hash, validator_id)
  → Record approval on blockchain
  → Mine approval block

apply_consensus_training(proposal_hash, texts, learning_rate)
  → Call custom_llm.apply_approved_training()
  → Mine update block
  → Only succeeds if proposal approved
```

### 3. API Endpoints

#### Propose Training
```bash
curl -X POST http://localhost:5000/decentralized/propose-training \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["I have a fever", "High temperature"],
    "learning_rate": 0.01
  }'
```

Response:
```json
{
  "status": "success",
  "proposal": {
    "status": "proposed",
    "proposal_hash": "a1b2c3d4",
    "message": "Training proposed to blockchain validators"
  },
  "blockchain_stats": {
    "chain_length": 5,
    "total_records": 12,
    "pending_records": 1
  }
}
```

#### Approve Proposal
```bash
curl -X POST http://localhost:5000/decentralized/approve-proposal \
  -H "Content-Type: application/json" \
  -d '{
    "proposal_hash": "a1b2c3d4",
    "validator_id": "validator-alice"
  }'
```

Response:
```json
{
  "status": "success",
  "approval": {
    "status": "approved",
    "proposal_hash": "a1b2c3d4",
    "validator": "validator-alice"
  },
  "blockchain_stats": {
    "chain_length": 6,
    "total_records": 13
  }
}
```

#### Apply Consensus Training
```bash
curl -X POST http://localhost:5000/decentralized/apply-training \
  -H "Content-Type: application/json" \
  -d '{
    "proposal_hash": "a1b2c3d4",
    "texts": ["I have a fever", "High temperature"],
    "learning_rate": 0.01
  }'
```

Response:
```json
{
  "status": "success",
  "metrics": {
    "batch_size": 2,
    "vocab_before": 150,
    "vocab_after": 165,
    "model_hash_after": "xyz789",
    "proposal_hash": "a1b2c3d4"
  },
  "model_info": {
    "name": "custom-medical-llm-v1",
    "training_steps": 3,
    "vocab_size": 165
  },
  "blockchain_stats": {
    "chain_length": 7,
    "total_records": 14
  }
}
```

#### Get Decentralized Status
```bash
curl http://localhost:5000/decentralized/status
```

Response:
```json
{
  "blockchain": {
    "chain_length": 7,
    "total_records": 14,
    "is_valid": true
  },
  "model": {
    "name": "custom-medical-llm-v1",
    "training_steps": 3,
    "vocab_size": 165
  },
  "decentralized_stats": {
    "total_proposals": 3,
    "total_approvals": 3,
    "applied_updates": 3,
    "pending_records": 0
  },
  "consensus_required": true,
  "validators_required": 1
}
```

## Usage Example

### Step 1: Run the Demo
```bash
python demo_decentralized_ml.py
```

Output:
```
====================================================================
DECENTRALIZED ML TRAINING VIA BLOCKCHAIN CONSENSUS
====================================================================

[BATCH 1] Training: Fever-related queries
--------------------------------------------------
Step 1: PROPOSE training on fever queries to validators...
  ✓ Proposal hash: a1b2c3d4e5f6
  ✓ Status: Training proposed to blockchain validators

Step 2: VALIDATOR APPROVAL (validator-alice approves)
  ✓ Validator: validator-alice
  ✓ Approval status: approved

Step 3: APPLY consensus-approved training...
  ✓ Model hash: abc456xyz
  ✓ Vocab size: 165
  ✓ New words learned: 15
  ✓ Total training steps: 1
  ✓ Blockchain records: 4
```

### Step 2: Test API Workflow

**Terminal 1: Start server**
```bash
python app.py
```

**Terminal 2: Run workflow**
```bash
# 1. Propose
PROPOSAL=$(curl -s -X POST http://localhost:5000/decentralized/propose-training \
  -H "Content-Type: application/json" \
  -d '{"texts":["I have a fever"]}' | jq -r '.proposal.proposal_hash')

echo "Proposal: $PROPOSAL"

# 2. Approve
curl -s -X POST http://localhost:5000/decentralized/approve-proposal \
  -H "Content-Type: application/json" \
  -d "{\"proposal_hash\":\"$PROPOSAL\",\"validator_id\":\"validator-1\"}" | jq

# 3. Apply
curl -s -X POST http://localhost:5000/decentralized/apply-training \
  -H "Content-Type: application/json" \
  -d "{\"proposal_hash\":\"$PROPOSAL\",\"texts\":[\"I have a fever\"]}" | jq

# 4. Check Status
curl -s http://localhost:5000/decentralized/status | jq
```

## Security Properties

### 1. Consensus
- ✅ Training only applied if validators approve
- ✅ Proposals immutable (hash-based)
- ✅ Approval trail forever on blockchain

### 2. Integrity
- ✅ Model hash before/after tracked
- ✅ Proposal hash links updates to originals
- ✅ Complete audit trail

### 3. Decentralization
- ✅ Validators can reject proposals
- ✅ Multiple validators required (configurable)
- ✅ No single point of control

### 4. Transparency
- ✅ All training decisions on-chain
- ✅ Model evolution fully auditable
- ✅ Anyone can verify blockchain

## Configuration

### Validators Required
Edit `blockchain.py`:
```python
# In validate_model_update():
def validate_model_update(self, update_hash: str, required_validators: int = 1) -> bool:
    approvals >= required_validators  # Change to 2, 3, etc.
```

### Learning Rate
Control training speed in API calls:
```json
{
  "texts": [...],
  "learning_rate": 0.01  # or 0.001, 0.1, etc.
}
```

### Model Checkpointing
Save approved model state:
```bash
curl -X POST http://localhost:5000/save-checkpoint
```

## Advantages of This Approach

| Feature | Centralized | Decentralized ML |
|---------|-----------|------------------|
| Single authority | ✅ | ❌ |
| Consensus required | ❌ | ✅ |
| Model changes audit | ❌ | ✅ |
| Validator control | ❌ | ✅ |
| Tamper-proof | ❌ | ✅ |
| Verifiable history | ❌ | ✅ |

## Advanced: Multi-Validator Setup

```python
# Add 3 validators
validators = ['alice', 'bob', 'carol']

# Propose once
proposal_hash = propose_training_to_consensus(texts, lr=0.01)

# Get all validators to approve
for validator in validators:
    approve_training_proposal(proposal_hash, validator_id=f'validator-{validator}')

# Then apply (requires 3 approvals)
result = apply_consensus_training(proposal_hash, texts, lr=0.01)
```

## Monitoring

```bash
# Watch blockchain grow
watch -n 2 'curl -s http://localhost:5000/decentralized/status | jq'

# Track proposals
curl -s http://localhost:5000/blockchain-history | jq '.records[] | select(.type=="training_proposal")'

# Get model state per block
curl -s http://localhost:5000/model-info | jq '.model'
```

## Next Steps

1. **Add Multi-Signature**: Require M-of-N validators to approve
2. **Stake Validators**: Require validators to stake tokens
3. **Penalty System**: Slash validators who reject good updates
4. **Voting Power**: Different validators have different voting weights
5. **Smart Contracts**: Implement on actual blockchains (Algorand, Ethereum)
6. **Federated Learning**: Train across multiple nodes with consensus

## References

- Blockchain consensus: SHA-256 proof-of-work
- Model transparency: Full audit trail of updates
- Decentralization: Validator approval required before training
- Immutability: All records on permanent blockchain ledger

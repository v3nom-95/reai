"""
Blockchain ledger for ML model training transparency and provenance.
Stores training records, model checkpoints, and versioning immutably.
"""
import hashlib
import json
from datetime import datetime
from typing import Dict, List, Any


class BlockchainLedger:
    """Simple immutable blockchain for ML model training transparency."""
    
    def __init__(self):
        self.chain: List[Dict[str, Any]] = []
        self.pending_records: List[Dict[str, Any]] = []
        self._create_genesis_block()
    
    def _create_genesis_block(self):
        """Initialize blockchain with genesis block."""
        genesis = {
            'index': 0,
            'timestamp': datetime.utcnow().isoformat(),
            'records': [],
            'previous_hash': '0',
            'nonce': 0
        }
        genesis['hash'] = self._compute_hash(genesis)
        self.chain.append(genesis)
    
    def _compute_hash(self, block: Dict) -> str:
        """Compute SHA-256 hash of a block."""
        block_copy = block.copy()
        block_copy.pop('hash', None)
        block_str = json.dumps(block_copy, sort_keys=True)
        return hashlib.sha256(block_str.encode()).hexdigest()
    
    def add_record(self, record_type: str, data: Dict[str, Any]):
        """Add a training/model record to pending queue."""
        record = {
            'type': record_type,
            'timestamp': datetime.utcnow().isoformat(),
            'data': data
        }
        self.pending_records.append(record)
        return record
    
    def mine_block(self, miner_id: str = 'system') -> Dict[str, Any]:
        """Mine a new block with pending records."""
        if not self.pending_records:
            return {'status': 'no_pending_records', 'success': False}
        
        new_block = {
            'index': len(self.chain),
            'timestamp': datetime.utcnow().isoformat(),
            'records': self.pending_records.copy(),
            'miner': miner_id,
            'previous_hash': self.chain[-1]['hash'],
            'nonce': 0
        }
        
        # Simple proof-of-work (2 leading zeros)
        while not self._is_valid_proof(new_block):
            new_block['nonce'] += 1
        
        new_block['hash'] = self._compute_hash(new_block)
        self.chain.append(new_block)
        self.pending_records = []
        
        return {
            'status': 'success',
            'success': True,
            'block_index': new_block['index'],
            'block_hash': new_block['hash'],
            'nonce': new_block['nonce']
        }
    
    def _is_valid_proof(self, block: Dict, difficulty: int = 2) -> bool:
        """Check if block hash meets difficulty requirement."""
        block_hash = self._compute_hash(block)
        return block_hash.startswith('0' * difficulty)
    
    def get_chain(self) -> List[Dict]:
        """Return full immutable chain."""
        return self.chain.copy()
    
    def verify_integrity(self) -> bool:
        """Verify blockchain integrity."""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]
            
            if current['previous_hash'] != previous['hash']:
                return False
            if current['hash'] != self._compute_hash(current):
                return False
        return True
    
    def get_model_history(self, model_name: str) -> List[Dict]:
        """Retrieve all training records for a model."""
        history = []
        for block in self.chain:
            for record in block.get('records', []):
                if record['data'].get('model_name') == model_name:
                    history.append({
                        'block_index': block['index'],
                        'block_hash': block['hash'],
                        'block_timestamp': block['timestamp'],
                        **record
                    })
        return history
    
    def get_stats(self) -> Dict[str, Any]:
        """Get blockchain statistics."""
        total_records = sum(len(block.get('records', [])) for block in self.chain)
        return {
            'chain_length': len(self.chain),
            'total_records': total_records,
            'pending_records': len(self.pending_records),
            'is_valid': self.verify_integrity(),
            'latest_block_hash': self.chain[-1]['hash'] if self.chain else None
        }
    
    def get_latest_model_state(self) -> Dict[str, Any]:
        """Get the latest approved model state from blockchain."""
        for block in reversed(self.chain):
            for record in block.get('records', []):
                if record['type'] == 'model_update_approved':
                    return {
                        'block_index': block['index'],
                        'block_hash': block['hash'],
                        'model_state': record['data'],
                        'timestamp': record['timestamp']
                    }
        return None
    
    def validate_model_update(self, update_hash: str, required_validators: int = 1) -> bool:
        """Validate that model update has consensus approval (checks chain + pending)."""
        approvals = 0
        # Check mined blocks
        for block in self.chain:
            for record in block.get('records', []):
                if (record['type'] == 'validator_approval' and 
                    record['data'].get('proposal_hash') == update_hash):
                    approvals += 1
        # Check pending records too
        for record in self.pending_records:
            if (record['type'] == 'validator_approval' and 
                record['data'].get('proposal_hash') == update_hash):
                approvals += 1
        return approvals >= required_validators


# Global blockchain instance
blockchain = BlockchainLedger()

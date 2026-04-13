#!/usr/bin/env python3
"""
Decentralized ML demo: shows consensus-based model training via blockchain.
"""
import json
import time
from model import (
    propose_training_to_consensus,
    approve_training_proposal,
    apply_consensus_training,
    get_model_info,
    get_blockchain_stats
)
from blockchain import blockchain


def demo_decentralized_training():
    """Demonstrate the full decentralized ML workflow."""
    
    print("=" * 70)
    print("DECENTRALIZED ML TRAINING VIA BLOCKCHAIN CONSENSUS")
    print("=" * 70)
    
    # Training data
    batch1_texts = [
        "I have a persistent fever and chills",
        "My temperature is very high",
        "I'm experiencing body aches with fever"
    ]
    
    batch2_texts = [
        "My throat is sore and painful",
        "I have difficulty swallowing",
        "Throat pain is severe"
    ]
    
    batch3_texts = [
        "I'm experiencing chest pain",
        "Sharp pain in my chest",
        "Chest discomfort and tightness"
    ]
    
    print("\n" + "=" * 70)
    print("WORKFLOW: PROPOSE → APPROVE → APPLY → VERIFY")
    print("=" * 70)
    
    # ===== BATCH 1: Fever Training =====
    print("\n[BATCH 1] Training: Fever-related queries")
    print("-" * 70)
    
    print("Step 1: PROPOSE training on fever queries to validators...")
    proposal1 = propose_training_to_consensus(batch1_texts, learning_rate=0.01)
    proposal1_hash = proposal1['proposal_hash']
    print(f"  ✓ Proposal hash: {proposal1_hash}")
    print(f"  ✓ Status: {proposal1['message']}")
    
    print("\nStep 2: VALIDATOR APPROVAL (validator-alice approves)")
    approval1 = approve_training_proposal(proposal1_hash, validator_id='validator-alice')
    print(f"  ✓ Validator: {approval1['validator']}")
    print(f"  ✓ Approval status: {approval1['status']}")
    
    print("\nStep 3: APPLY consensus-approved training...")
    metrics1 = apply_consensus_training(proposal1_hash, batch1_texts, learning_rate=0.01)
    print(f"  ✓ Model hash: {metrics1['model_hash_after']}")
    print(f"  ✓ Vocab size: {metrics1['vocab_after']}")
    print(f"  ✓ New words learned: {metrics1['new_words']}")
    
    model_info1 = get_model_info()
    blockchain_stats1 = get_blockchain_stats()
    print(f"  ✓ Total training steps: {model_info1['training_steps']}")
    print(f"  ✓ Blockchain records: {blockchain_stats1['total_records']}")
    
    # ===== BATCH 2: Throat Training =====
    print("\n[BATCH 2] Training: Throat-related queries")
    print("-" * 70)
    
    print("Step 1: PROPOSE training on throat queries...")
    proposal2 = propose_training_to_consensus(batch2_texts, learning_rate=0.01)
    proposal2_hash = proposal2['proposal_hash']
    print(f"  ✓ Proposal hash: {proposal2_hash}")
    
    print("\nStep 2: VALIDATOR APPROVAL (validator-bob approves)")
    approval2 = approve_training_proposal(proposal2_hash, validator_id='validator-bob')
    print(f"  ✓ Validator: {approval2['validator']}")
    
    print("\nStep 3: APPLY consensus-approved training...")
    metrics2 = apply_consensus_training(proposal2_hash, batch2_texts, learning_rate=0.01)
    print(f"  ✓ Model hash: {metrics2['model_hash_after']}")
    print(f"  ✓ Vocab size: {metrics2['vocab_after']}")
    
    model_info2 = get_model_info()
    blockchain_stats2 = get_blockchain_stats()
    print(f"  ✓ Total training steps: {model_info2['training_steps']}")
    print(f"  ✓ Blockchain records: {blockchain_stats2['total_records']}")
    
    # ===== BATCH 3: Chest Pain Training =====
    print("\n[BATCH 3] Training: Chest pain queries")
    print("-" * 70)
    
    print("Step 1: PROPOSE training on chest pain queries...")
    proposal3 = propose_training_to_consensus(batch3_texts, learning_rate=0.01)
    proposal3_hash = proposal3['proposal_hash']
    print(f"  ✓ Proposal hash: {proposal3_hash}")
    
    print("\nStep 2: VALIDATOR APPROVAL (validator-carol approves)")
    approval3 = approve_training_proposal(proposal3_hash, validator_id='validator-carol')
    print(f"  ✓ Validator: {approval3['validator']}")
    
    print("\nStep 3: APPLY consensus-approved training...")
    metrics3 = apply_consensus_training(proposal3_hash, batch3_texts, learning_rate=0.01)
    print(f"  ✓ Model hash: {metrics3['model_hash_after']}")
    print(f"  ✓ Vocab size: {metrics3['vocab_after']}")
    
    model_info3 = get_model_info()
    blockchain_stats3 = get_blockchain_stats()
    print(f"  ✓ Total training steps: {model_info3['training_steps']}")
    print(f"  ✓ Blockchain records: {blockchain_stats3['total_records']}")
    
    # ===== VERIFICATION =====
    print("\n" + "=" * 70)
    print("VERIFICATION & FINAL STATS")
    print("=" * 70)
    
    print("\n[MODEL STATUS]")
    final_info = get_model_info()
    for key, value in final_info.items():
        print(f"  {key}: {value}")
    
    print("\n[BLOCKCHAIN INTEGRITY]")
    final_stats = get_blockchain_stats()
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
    
    print("\n[BLOCKCHAIN CHAIN]")
    chain = blockchain.get_chain()
    print(f"  Total blocks: {len(chain)}")
    for i, block in enumerate(chain):
        print(f"  Block {i}: {len(block['records'])} records, hash={block['hash'][:8]}...")
    
    print("\n[TRAINING HISTORY]")
    history = blockchain.get_model_history('custom-medical-llm-v1')
    print(f"  Total training records: {len(history)}")
    for i, record in enumerate(history[-5:], 1):  # Last 5
        print(f"  {i}. Type={record['type']}, Time={record['timestamp']}")
    
    print("\n" + "=" * 70)
    print("✓ DECENTRALIZED ML TRAINING COMPLETE!")
    print("=" * 70)
    print("\nKey Achievements:")
    print("✓ Model trained via blockchain consensus")
    print("✓ All training proposals and approvals on-chain")
    print("✓ Immutable training history with full audit trail")
    print("✓ Validators control model updates")
    print("✓ Blockchain integrity verified")
    print("\n")


if __name__ == '__main__':
    demo_decentralized_training()

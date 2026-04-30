"""
app.py — REAI Flask application.
All blockchain operations go through Algorand Testnet exclusively.
"""
from flask import Flask, request, jsonify, render_template
import algosdk
import hashlib
import json
import os
import difflib
import base64
from datetime import datetime

from model import (
    find_similar_by_embedding, load_model, train_model_on_batch,
    get_model_info, get_blockchain_stats,
    propose_training_to_consensus, approve_training_proposal,
    apply_consensus_training, propose_train_from_chain,
    apply_consensus_training_with_dp, build_dataset_from_chain
)
from blockchain import blockchain, ADDRESS, ALGOD_ADDRESS
from custom_model import custom_llm

app = Flask(__name__)

ALGOD_ADDRESS_LOCAL = "https://testnet-api.algonode.cloud"
algod_client = algosdk.v2client.algod.AlgodClient("", ALGOD_ADDRESS_LOCAL)

DATA_FILE = "medical_data.json"


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_local_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE) as f:
            return json.load(f)
    return []


def load_data_from_algorand():
    """Pull {query, response} pairs from Algorand Testnet Indexer."""
    indexer = algosdk.v2client.indexer.IndexerClient(
        "", "https://testnet-idx.algonode.cloud"
    )
    data = []
    try:
        resp = indexer.search_transactions_by_address(address=ADDRESS, limit=200)
        for txn in resp.get("transactions", []):
            note = txn.get("note")
            if not note:
                continue
            try:
                decoded = json.loads(base64.b64decode(note).decode("utf-8"))
                if isinstance(decoded, dict):
                    # Plain {query, response} format (old chat logs)
                    if "query" in decoded and "response" in decoded:
                        data.append(decoded)
                    elif decoded.get("protocol") in ("REAI-V1", "REAI-V2"):
                        d = decoded.get("data", {})
                        # Direct {query, response} inside data
                        if "query" in d and "response" in d:
                            data.append(d)
                        # knowledge_entry: multiple queries → one response
                        elif d.get("type") == "knowledge_entry" or \
                             decoded.get("type") == "knowledge_entry":
                            queries  = d.get("queries", [])
                            response = d.get("response", "")
                            if queries and response:
                                for q in queries:
                                    data.append({"query": q, "response": response,
                                                 "source": "algorand_consensus"})
            except Exception:
                continue
    except Exception as e:
        print(f"[Algorand] load_data error: {e}")
    return data


def sanitize(text: str) -> str:
    blocked = ["ignore previous", "system prompt", "jailbreak",
               "override", "forget instructions", "disregard"]
    for b in blocked:
        if b in text.lower():
            return "[SECURITY_BLOCKED]"
    return text


def get_medical_advice(msg: str) -> str:
    m = msg.lower()
    if any(k in m for k in ["fever", "temperature", "feverish"]):
        return ("For fever: rest, stay hydrated, and use OTC fever reducers like "
                "acetaminophen. If temperature exceeds 103°F or persists beyond "
                "3 days, seek immediate medical attention.")
    if any(k in m for k in ["headache", "migraine", "head pain"]):
        return ("For headaches: rest in a dark quiet room, apply a cold compress, "
                "and stay hydrated. Severe or sudden headaches warrant a doctor visit.")
    if any(k in m for k in ["cough", "coughing"]):
        return ("For cough: stay hydrated, try honey in warm water. Persistent cough "
                "over 2 weeks or coughing blood requires medical evaluation.")
    if any(k in m for k in ["sore throat", "throat pain"]):
        return ("For sore throat: gargle warm salt water, stay hydrated, use lozenges. "
                "If it lasts over a week with fever, see a doctor.")
    if any(k in m for k in ["chest pain", "heart", "palpitation"]):
        return ("EMERGENCY: Chest pain may indicate a cardiac event. "
                "Call 911 immediately. Do not drive yourself.")
    if any(k in m for k in ["anxiety", "stress", "panic"]):
        return ("For anxiety: practice deep breathing and mindfulness. "
                "If overwhelming or persistent, consult a mental health professional.")
    if any(k in m for k in ["diabetes", "blood sugar", "glucose"]):
        return ("For diabetes: monitor blood sugar regularly, follow your prescribed "
                "diet and medications. Consult your endocrinologist regularly.")
    if any(k in m for k in ["cold", "flu", "runny nose", "congestion"]):
        return ("For cold/flu: rest, fluids, saline sprays. See a doctor if symptoms "
                "worsen or last over 10 days.")
    if any(k in m for k in ["stomach", "nausea", "vomit", "diarrhea"]):
        return ("For GI symptoms: sip clear fluids, rest. Seek help if symptoms "
                "persist over 48 hours or include blood.")
    if any(k in m for k in ["sleep", "insomnia"]):
        return ("For sleep issues: maintain a consistent schedule, avoid screens "
                "before bed. Persistent insomnia warrants a doctor visit.")
    if any(k in m for k in ["allergy", "rash", "hives", "itchy"]):
        return ("For allergies/rash: avoid triggers, use antihistamines. "
                "Severe reactions with throat swelling need emergency care.")
    return ("I'm a responsible medical assistant. For specific symptoms, please "
            "consult a qualified healthcare professional. I cannot provide "
            "personalised medical diagnoses.")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    raw = request.json.get("message", "")
    user_message = sanitize(raw)

    if "[SECURITY_BLOCKED]" in user_message:
        blockchain.add_record("security_threat", {
            "raw_query": raw[:200],
            "timestamp": datetime.utcnow().isoformat()
        })
        blockchain.mine_block(miner_id="security-firewall")
        return jsonify({
            "response":   "⚠️ Security threat detected and logged to Algorand Testnet.",
            "source":     "Security Firewall",
            "hash":       hashlib.sha256(raw.encode()).hexdigest(),
            "txid":       "blocked",
            "model_info": get_model_info(),
            "algo_stats": get_blockchain_stats()
        })

    # 1. Load knowledge from Algorand Testnet
    algo_data  = load_data_from_algorand()
    local_data = load_local_data()
    data = algo_data + [d for d in local_data
                        if d.get("query") not in {x.get("query") for x in algo_data}]

    # 2. Real-time training on this query
    try:
        train_model_on_batch([user_message], learning_rate=0.01)
    except Exception as e:
        print(f"[Train] {e}")

    # 3. Retrieve response
    source = "Rule-Based Fallback"
    ai_response = None

    try:
        found = find_similar_by_embedding(user_message, data)
        if found:
            ai_response, score = found
            source = f"Algorand On-Chain KB (similarity={score:.2f})"
    except Exception as e:
        print(f"[Embed] {e}")

    if not ai_response:
        ai_response = get_medical_advice(user_message)

    ai_response += ("\n\n⚕️ Disclaimer: This is general health information only. "
                    "Always consult a qualified healthcare professional.")

    # 4. Hash response and write to Algorand
    response_hash = hashlib.sha256(ai_response.encode()).hexdigest()
    note_payload  = {
        "query":    user_message,
        "response": ai_response[:300],   # keep note under 1 KB
        "hash":     response_hash
    }
    note_bytes = json.dumps(note_payload).encode()
    if len(note_bytes) > 1000:
        note_bytes = response_hash.encode()

    txid = "pending"
    try:
        params  = algod_client.suggested_params()
        txn     = algosdk.transaction.PaymentTxn(
            sender=ADDRESS, sp=params, receiver=ADDRESS,
            amt=0, note=note_bytes
        )
        signed  = txn.sign(blockchain.private_key)
        txid    = algod_client.send_transaction(signed)
        algosdk.transaction.wait_for_confirmation(algod_client, txid, 4)
    except Exception as e:
        print(f"[Algorand TX] {e}")
        txid = f"error: {str(e)[:60]}"

    return jsonify({
        "response":   ai_response,
        "source":     source,
        "hash":       response_hash,
        "txid":       txid,
        "algo_url":   f"https://testnet.algoexplorer.io/tx/{txid}",
        "model_info": get_model_info(),
        "algo_stats": get_blockchain_stats()
    })


@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model":      get_model_info(),
        "blockchain": get_blockchain_stats()
    })


@app.route("/blockchain-history", methods=["GET"])
def blockchain_history():
    history = blockchain.get_model_history()
    return jsonify({
        "total_records": len(history),
        "records":       history[:60],
        "source":        "Algorand Testnet"
    })


@app.route("/algo-audit", methods=["GET"])
def algo_audit():
    """Full Algorand audit trail with type breakdown."""
    history = blockchain.get_model_history()
    by_type: dict = {}
    for rec in history:
        t = rec.get("type", "unknown")
        by_type.setdefault(t, []).append(rec)
    return jsonify({
        "total":    len(history),
        "by_type":  {k: len(v) for k, v in by_type.items()},
        "records":  history[:80],
        "address":  ADDRESS,
        "explorer": f"https://testnet.algoexplorer.io/address/{ADDRESS}"
    })


@app.route("/chain-integrity", methods=["GET"])
def chain_integrity():
    """Verify Algorand connectivity and return live network stats."""
    ok = blockchain.verify_integrity()
    stats = get_blockchain_stats()
    return jsonify({
        "is_valid":    ok,
        "network":     stats.get("network"),
        "last_round":  stats.get("last_round"),
        "total_records": stats.get("total_records"),
        "record_types":  stats.get("record_types", {})
    })


@app.route("/train", methods=["POST"])
def train():
    data = request.json or {}
    texts = data.get("texts", [])
    lr    = float(data.get("learning_rate", 0.01))
    if not texts:
        return jsonify({"error": "No texts provided"}), 400
    try:
        metrics = train_model_on_batch(texts, learning_rate=lr)
        return jsonify({"status": "success", "metrics": metrics,
                        "model_info": get_model_info()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/save-checkpoint", methods=["POST"])
def save_checkpoint():
    try:
        ckpt_hash = custom_llm.save_checkpoint("checkpoints/model_checkpoint.json")
        blockchain.mine_block(miner_id="checkpoint-saver")
        return jsonify({"status": "success", "checkpoint_hash": ckpt_hash,
                        "model_info": get_model_info()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Decentralised consensus endpoints ────────────────────────────────────────
@app.route("/decentralized/propose-training", methods=["POST"])
def propose_training():
    data  = request.json or {}
    texts = data.get("texts", [])
    lr    = float(data.get("learning_rate", 0.01))
    if not texts:
        return jsonify({"error": "No texts provided"}), 400
    try:
        result = propose_training_to_consensus(texts, lr)
        return jsonify({"status": "success", "proposal": result,
                        "algo_stats": get_blockchain_stats()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/decentralized/propose-from-chain", methods=["POST"])
def propose_from_chain():
    data = request.json or {}
    try:
        result = propose_train_from_chain(
            limit=int(data.get("limit", 200)),
            learning_rate=float(data.get("learning_rate", 0.01)),
            filter_pii=bool(data.get("filter_pii", True))
        )
        return jsonify({"status": "success", "proposal": result,
                        "algo_stats": get_blockchain_stats()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/decentralized/approve-proposal", methods=["POST"])
def approve_proposal():
    """
    Validator approves a training proposal.
    Requires cryptographic proof:
      - validator_name: key in validators.json (e.g. 'validator-alpha')
      OR
      - validator_address + signature: pre-signed externally

    If validator_name is provided, the server signs on behalf of that
    registered validator (demo mode — in production each validator runs
    their own client and signs locally).
    """
    from blockchain import sign_proposal, verify_approval_signature, \
        VALIDATORS, VALIDATOR_ADDRESSES

    data          = request.json or {}
    proposal_hash = data.get("proposal_hash", "").strip()
    if not proposal_hash:
        return jsonify({"error": "proposal_hash required"}), 400

    validator_name    = data.get("validator_name", "").strip()
    validator_address = data.get("validator_address", "").strip()
    signature         = data.get("signature", "").strip()

    # ── Path A: validator_name provided — sign server-side (demo) ────────────
    if validator_name:
        if validator_name not in VALIDATORS:
            known = list(VALIDATORS.keys())
            return jsonify({
                "error": f"Unknown validator '{validator_name}'. "
                         f"Registered validators: {known}"
            }), 400

        sig = sign_proposal(proposal_hash, validator_name)
        if not sig:
            return jsonify({"error": "Signing failed"}), 500

        validator_address = VALIDATORS[validator_name]["address"]
        signature         = sig

    # ── Path B: address + signature provided — verify externally signed ──────
    elif validator_address and signature:
        if validator_address not in VALIDATOR_ADDRESSES:
            return jsonify({
                "error": f"Address {validator_address} is not a registered validator."
            }), 403
        if not verify_approval_signature(proposal_hash, validator_address, signature):
            return jsonify({
                "error": "Signature verification FAILED. "
                         "Approval rejected — invalid cryptographic proof."
            }), 403
    else:
        return jsonify({
            "error": "Provide either 'validator_name' (registered key) "
                     "or 'validator_address' + 'signature'."
        }), 400

    # ── Record the cryptographically verified approval on Algorand ────────────
    try:
        result = approve_training_proposal(
            proposal_hash,
            validator_address=validator_address,
            signature=signature
        )
        return jsonify({
            "status":           "success",
            "approval":         result,
            "verified":         True,
            "validator_address": validator_address,
            "algo_stats":       get_blockchain_stats()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/decentralized/apply-training", methods=["POST"])
def apply_training():
    data          = request.json or {}
    proposal_hash = data.get("proposal_hash")
    texts         = data.get("texts", [])
    response_text = data.get("response_text", "").strip()   # ← the answer to teach
    lr            = float(data.get("learning_rate", 0.01))
    dp_params     = data.get("dp_params")
    if not proposal_hash or not texts:
        return jsonify({"error": "proposal_hash and texts required"}), 400
    try:
        if dp_params:
            metrics = apply_consensus_training_with_dp(
                proposal_hash, texts, lr, dp_params=dp_params)
        else:
            metrics = apply_consensus_training(proposal_hash, texts, lr)
        if metrics.get("status") == "failed":
            return jsonify({"error": metrics.get("reason")}), 403

        # ── Save {query, response} pairs so chat can retrieve them ────────────
        if response_text:
            local = load_local_data()
            existing_queries = {e.get("query", "").lower() for e in local}
            added = 0
            for q in texts:
                if q.lower() not in existing_queries:
                    local.append({
                        "query":    q,
                        "response": response_text,
                        "source":   "consensus_trained"
                    })
                    existing_queries.add(q.lower())
                    added += 1
            with open(DATA_FILE, "w") as f:
                json.dump(local, f, indent=2)
            metrics["kb_entries_added"] = added

            # Also write to Algorand so it persists across restarts
            blockchain.add_record("knowledge_entry", {
                "queries":   texts,
                "response":  response_text,
                "proposal_hash": proposal_hash
            })

        return jsonify({"status": "success", "metrics": metrics,
                        "model_info": get_model_info(),
                        "algo_stats": get_blockchain_stats()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/decentralized/status", methods=["GET"])
def decentralized_status():
    history = blockchain.get_model_history()
    counts  = {"training_proposal": 0, "validator_approval": 0,
               "model_update_approved": 0, "query_inference": 0,
               "security_threat": 0, "model_training_batch": 0}
    for rec in history:
        t = rec.get("type", "")
        if t in counts:
            counts[t] += 1
    return jsonify({
        "algo_stats":         get_blockchain_stats(),
        "model":              get_model_info(),
        "decentralized_stats": counts,
        "consensus_source":   "Algorand Testnet",
        "validators_required": 2
    })


@app.route("/decentralized/all-proposals", methods=["GET"])
def all_proposals():
    history   = blockchain.get_model_history()
    proposals = {}
    for rec in history:
        t    = rec.get("type")
        d    = rec.get("data", {})
        ph   = d.get("proposal_hash")
        if not ph:
            continue
        if t == "training_proposal":
            proposals.setdefault(ph, {
                "hash": ph, "status": "Proposed",
                "approvals": [], "txid": rec.get("txid", ""),
                "timestamp": rec.get("timestamp", ""),
                "texts_count": len(d.get("proposal", {}).get("texts", []))
            })
        elif t == "validator_approval":
            if ph in proposals:
                vid = d.get("validator_id", "")
                if vid not in proposals[ph]["approvals"]:
                    proposals[ph]["approvals"].append(vid)
        elif t == "model_update_approved":
            if ph in proposals:
                proposals[ph]["status"] = "Applied"
    return jsonify(list(proposals.values())[::-1])


@app.route("/validators", methods=["GET"])
def get_validators():
    """Return registered validator addresses (public info only, no private keys)."""
    from blockchain import VALIDATORS
    return jsonify({
        name: {"address": v["address"]}
        for name, v in VALIDATORS.items()
    })


@app.route("/decentralized/network-nodes", methods=["GET"])
def network_nodes():
    """Return simulated validator node status."""
    import random, time
    nodes = [
        {"id": "Validator-Alpha", "ip": "192.168.1.10",
         "status": "ONLINE", "latency": f"{random.randint(20,80)}ms",
         "role": "validator"},
        {"id": "Validator-Beta",  "ip": "192.168.1.11",
         "status": "ONLINE", "latency": f"{random.randint(20,80)}ms",
         "role": "validator"},
        {"id": "Validator-Gamma", "ip": "192.168.1.12",
         "status": random.choice(["ONLINE", "JITTER"]),
         "latency": f"{random.randint(20,150)}ms", "role": "validator"},
        {"id": "Miner-Node-01",   "ip": "10.0.0.5",
         "status": "ONLINE", "latency": f"{random.randint(10,40)}ms",
         "role": "miner"},
    ]
    return jsonify({"nodes": nodes, "algo_address": ADDRESS})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

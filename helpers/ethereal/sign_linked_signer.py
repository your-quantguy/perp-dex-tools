"""
Utility to produce signerSignature for Ethereal linked signer flow.

Minimal use:
    python sign_linked_signer.py \
        --sender 0xYourEOA \
        --subaccount 0xYourBytes32Label

Other fields auto-generated:
- signer = address derived from ETHEREAL_PRIVATE_KEY (read from .env)
- nonce   = time.time_ns()
- signedAt= current seconds
- chainId / verifyingContract / domain name+version use Ethereal defaults (override flags if needed)

Install requirement if missing: pip install eth_account python-dotenv
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict

from dotenv import load_dotenv

from eth_account import Account
from eth_account.messages import encode_typed_data
from eth_utils import to_checksum_address


TYPES = {
    "EIP712Domain": [
        {"name": "name", "type": "string"},
        {"name": "version", "type": "string"},
        {"name": "chainId", "type": "uint256"},
        {"name": "verifyingContract", "type": "address"},
    ],
    "LinkSigner": [
        {"name": "sender", "type": "address"},
        {"name": "signer", "type": "address"},
        {"name": "subaccount", "type": "bytes32"},
        {"name": "nonce", "type": "uint64"},
        {"name": "signedAt", "type": "uint64"},
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create signerSignature for Ethereal linked signer")
    parser.add_argument("--sender", required=True, help="EOA that owns the subaccount")
    parser.add_argument("--subaccount", required=True, help="bytes32 subaccount label (0x...)")
    parser.add_argument("--nonce", help="uint64 nonce as string; default: time.time_ns()")
    parser.add_argument("--signed-at", dest="signed_at", type=int, help="Signed at timestamp (seconds); default: now")
    parser.add_argument("--chain-id", dest="chain_id", type=int, default=5064014, help="Chain ID (default: 5064014)")
    parser.add_argument(
        "--verifying-contract",
        dest="verifying_contract",
        default="0xb3cdc82035c495c484c9ff11ed5f3ff6d342e3cc",
        help="Domain verifyingContract (default: Ethereal mainnet)",
    )
    parser.add_argument("--name", default="Ethereal", help="Domain name (default: Ethereal)")
    parser.add_argument("--version", default="1", help="Domain version (default: 1)")
    return parser.parse_args()


def build_typed_data(args: argparse.Namespace) -> Dict[str, Any]:
    domain = {
        "name": args.name,
        "version": args.version,
        "chainId": args.chain_id,
        "verifyingContract": args.verifying_contract,
    }
    message = {
        "sender": args.sender,
        "signer": args.signer,
        "subaccount": args.subaccount,
        "nonce": str(args.nonce),
        "signedAt": int(args.signed_at),
    }
    return {
        "types": TYPES,
        "primaryType": "LinkSigner",
        "domain": domain,
        "message": message,
    }


def main():
    load_dotenv()
    args = parse_args()

    priv_key = os.getenv("ETHEREAL_PRIVATE_KEY")
    if not priv_key:
        sys.stderr.write("ETHEREAL_PRIVATE_KEY not set in environment or .env\n")
        sys.exit(1)

    account = Account.from_key(priv_key)

    # Autofill signer, nonce, signedAt if missing
    args.signer = to_checksum_address(account.address)
    args.sender = to_checksum_address(args.sender)
    args.subaccount = args.subaccount
    if not args.nonce:
        args.nonce = str(time.time_ns())
    if not args.signed_at:
        args.signed_at = int(time.time())

    typed_data = build_typed_data(args)
    signable = encode_typed_data(full_message=typed_data)
    signed = account.sign_message(signable)

    # Output only the signature hex (signerSignature)
    sys.stdout.write("\n0x" + signed.signature.hex() + "\n")


if __name__ == "__main__":
    main()

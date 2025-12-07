#!/usr/bin/env python3
"""
GRVT Transfer Tool
Transfer tokens between GRVT accounts or sub-accounts
"""
import argparse
import logging
import os
import random
import sys
import time
from decimal import Decimal

from dotenv import load_dotenv
from pysdk import grvt_fixed_types, grvt_raw_types
from pysdk.grvt_raw_base import GrvtApiConfig, GrvtError
from pysdk.grvt_raw_env import GrvtEnv
from pysdk.grvt_raw_signing import sign_transfer
from pysdk.grvt_raw_sync import GrvtRawSync

# Load environment variables
load_dotenv()

# Setup logger
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_config() -> GrvtApiConfig:
    """Get GRVT API configuration from environment variables."""
    return GrvtApiConfig(
        env=GrvtEnv.PROD if os.getenv("GRVT_ENVIRONMENT", "prod") == "prod" else GrvtEnv.TESTNET,
        private_key=os.getenv("GRVT_PRIVATE_KEY"),
        trading_account_id=os.getenv("GRVT_TRADING_ACCOUNT_ID"),
        api_key=os.getenv("GRVT_API_KEY"),
        logger=logger,
    )


def get_main_account_id(api: GrvtRawSync) -> str:
    """Get the main account ID from the API."""
    resp = api.funding_account_summary_v1(grvt_raw_types.EmptyRequest())
    if isinstance(resp, GrvtError):
        raise ValueError(f"Failed to get main account: {resp}")
    if resp.result is None:
        raise ValueError("Funding account summary is null")
    return resp.result.main_account_id


def transfer_tokens(
    from_account_id: str,
    from_sub_account_id: str,
    to_account_id: str,
    to_sub_account_id: str,
    currency: str,
    num_tokens: str,
    expiration_days: int = 20,
) -> None:
    """
    Transfer tokens between GRVT accounts or sub-accounts.
    
    Args:
        from_account_id: Source account ID (hex address)
        from_sub_account_id: Source sub-account ID (0 for main account)
        to_account_id: Destination account ID (hex address)
        to_sub_account_id: Destination sub-account ID (0 for main account)
        currency: Token currency (e.g., USDT)
        num_tokens: Number of tokens to transfer
        expiration_days: Number of days until signature expires
    """
    # Initialize API
    config = get_config()
    api = GrvtRawSync(config=config)
    
    # Validate configuration
    if not all([config.trading_account_id, config.private_key, config.api_key]):
        raise ValueError("GRVT_TRADING_ACCOUNT_ID, GRVT_PRIVATE_KEY, and GRVT_API_KEY must be set")
    
    logger.info(f"🚀 Initiating transfer of {num_tokens} {currency}")
    logger.info(f"   From: {from_account_id} (sub: {from_sub_account_id})")
    logger.info(f"   To: {to_account_id} (sub: {to_sub_account_id})")
    
    # Create transfer object
    transfer = grvt_fixed_types.Transfer(
        from_account_id=from_account_id,
        from_sub_account_id=from_sub_account_id,
        to_account_id=to_account_id,
        to_sub_account_id=to_sub_account_id,
        currency=currency,
        num_tokens=num_tokens,
        signature=grvt_raw_types.Signature(
            signer="",  # Will be populated by sign_transfer
            r="",       # Will be populated by sign_transfer
            s="",       # Will be populated by sign_transfer
            v=0,        # Will be populated by sign_transfer
            expiration=str(
                time.time_ns() + expiration_days * 24 * 60 * 60 * 1_000_000_000
            ),
            nonce=random.randint(0, 2**32 - 1),
        ),
        transfer_type=grvt_fixed_types.TransferType.STANDARD,
        transfer_metadata="",
    )
    
    # Sign the transfer
    logger.info("📝 Signing transfer...")
    signed_transfer = sign_transfer(transfer, config, api.account)
    
    # Execute the transfer
    logger.info("📤 Submitting transfer to GRVT...")
    resp = api.transfer_v1(
        grvt_raw_types.ApiTransferRequest(
            signed_transfer.from_account_id,
            signed_transfer.from_sub_account_id,
            signed_transfer.to_account_id,
            signed_transfer.to_sub_account_id,
            signed_transfer.currency,
            signed_transfer.num_tokens,
            signed_transfer.signature,
            grvt_raw_types.TransferType.STANDARD,
            "",
        )
    )
    
    # Check response
    if isinstance(resp, GrvtError):
        logger.error(f"❌ Transfer failed: {resp}")
        raise ValueError(f"Transfer failed: {resp}")
    
    if resp.result is None:
        logger.error("❌ Transfer response is null")
        raise ValueError("Transfer response is null")
    
    if resp.result.ack:
        logger.info(f"✅ Transfer successful!")
        logger.info(f"   Transaction ID: {resp.result.tx_id}")
    else:
        logger.error(f"❌ Transfer not acknowledged")
        raise ValueError("Transfer not acknowledged")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Transfer tokens between GRVT accounts or sub-accounts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transfer from main account to sub-account
  python grvt_trasfer.py --from-account 0xabc... --from-sub 0 \\
                          --to-account 0xabc... --to-sub 123456 \\
                          --currency USDT --amount 100

  # Transfer between different accounts
  python grvt_trasfer.py --from-account 0xabc... --from-sub 0 \\
                          --to-account 0xdef... --to-sub 0 \\
                          --currency USDT --amount 50

  # Use --use-main-as-from to auto-fill from-account
  python grvt_trasfer.py --use-main-as-from --from-sub 0 \\
                          --to-account 0xdef... --to-sub 0 \\
                          --currency USDT --amount 25
"""
    )
    
    parser.add_argument('--from-account', type=str,
                        help='Source account ID (hex address, e.g., 0xabc...)')
    parser.add_argument('--from-sub', type=str, required=True,
                        help='Source sub-account ID (use 0 for main account)')
    parser.add_argument('--to-account', type=str, required=True,
                        help='Destination account ID (hex address)')
    parser.add_argument('--to-sub', type=str, required=True,
                        help='Destination sub-account ID (use 0 for main account)')
    parser.add_argument('--currency', type=str, default='USDT',
                        help='Token currency (default: USDT)')
    parser.add_argument('--amount', type=str, required=True,
                        help='Number of tokens to transfer')
    parser.add_argument('--expiration-days', type=int, default=20,
                        help='Number of days until signature expires (default: 20)')
    parser.add_argument('--use-main-as-from', action='store_true',
                        help='Use main account address as from-account (auto-fetch)')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Get from_account_id
    from_account_id = args.from_account
    if args.use_main_as_from:
        logger.info("🔍 Fetching main account address...")
        config = get_config()
        api = GrvtRawSync(config=config)
        from_account_id = get_main_account_id(api)
        logger.info(f"   Main account: {from_account_id}")
    
    if not from_account_id:
        logger.error("❌ Error: --from-account is required (or use --use-main-as-from)")
        sys.exit(1)
    
    # Validate amount
    try:
        amount_decimal = Decimal(args.amount)
        if amount_decimal <= 0:
            raise ValueError("Amount must be positive")
    except Exception as e:
        logger.error(f"❌ Invalid amount: {e}")
        sys.exit(1)
    
    try:
        transfer_tokens(
            from_account_id=from_account_id,
            from_sub_account_id=args.from_sub,
            to_account_id=args.to_account,
            to_sub_account_id=args.to_sub,
            currency=args.currency,
            num_tokens=args.amount,
            expiration_days=args.expiration_days,
        )
    except Exception as e:
        logger.error(f"❌ Transfer failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

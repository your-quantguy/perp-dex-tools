import asyncio
import os
import time
import unittest
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import base58
from bulk_keychain import Signer, compute_order_id_from_order, prepare_order

from exchanges.bulk import BulkClient


class BulkAgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Deterministic test keys only; never load local credentials.
        self.owner_key = base58.b58encode(bytes(range(32))).decode()
        self.agent_key = base58.b58encode(bytes(range(32, 64))).decode()
        self.owner = Signer.from_base58(self.owner_key, "mainnet").pubkey
        self.agent = Signer.from_base58(self.agent_key, "mainnet").pubkey
        self.config: Any = SimpleNamespace(
            ticker="ETH", contract_id="ETH-USD", tick_size=Decimal("0.001"),
            quantity=Decimal("0.02"), direction="buy",
        )

    def _client(self, overrides=None) -> Any:
        environment = {"BULK_AGENT_PRIVATE_KEY": self.agent_key, "BULK_ACCOUNT": self.owner}
        environment.update(overrides or {})
        with patch.dict(os.environ, environment, clear=True):
            return cast(Any, BulkClient(self.config))

    def _ready(self, client):
        client._ws_task = asyncio.get_running_loop().create_future()
        client._ws_ready.set()
        client._best_bid, client._best_ask = Decimal("2699"), Decimal("2700")
        client._book_update_time = time.monotonic()
        client._lot_size, client._min_notional = Decimal("0.0001"), Decimal("50")

    def test_owner_mode_remains_compatible(self):
        client = self._client({"BULK_AGENT_PRIVATE_KEY": "", "BULK_PRIVATE_KEY": self.owner_key})
        self.assertFalse(client.uses_agent_wallet)
        self.assertEqual(client.public_key, self.owner)
        self.assertEqual(client.signer_public_key, self.owner)
        signed = client._sign_action({"type": "cancel", "symbol": "ETH-USD", "order_id": self.agent})
        self.assertEqual(signed["account"], self.owner)
        self.assertEqual(signed["signer"], self.owner)

    def test_owner_account_mismatch_still_fails(self):
        with self.assertRaisesRegex(ValueError, "does not match BULK_ACCOUNT"):
            self._client({"BULK_AGENT_PRIVATE_KEY": "", "BULK_PRIVATE_KEY": self.owner_key,
                          "BULK_ACCOUNT": self.agent})

    def test_missing_credentials_fail_without_exposing_keys(self):
        with self.assertRaisesRegex(ValueError, "is required for Bulk trading"):
            self._client({"BULK_AGENT_PRIVATE_KEY": "  ", "BULK_PRIVATE_KEY": " "})

    def test_conflicting_credentials_fail_instead_of_selecting_a_key(self):
        with self.assertRaises(ValueError) as caught:
            self._client({"BULK_PRIVATE_KEY": self.owner_key})
        self.assertIn("Set only one", str(caught.exception))
        self.assertNotIn(self.owner_key, str(caught.exception))
        self.assertNotIn(self.agent_key, str(caught.exception))

    def test_agent_requires_explicit_trading_account(self):
        for account in ("", "  "):
            with self.subTest(account=account), self.assertRaisesRegex(ValueError, "BULK_ACCOUNT is required"):
                self._client({"BULK_ACCOUNT": account})

    def test_agent_rejects_invalid_account_public_keys(self):
        for account in ("not-a-base58-key!", base58.b58encode(b"short").decode()):
            with self.subTest(account=account), self.assertRaisesRegex(ValueError, "BULK_ACCOUNT must be"):
                self._client({"BULK_ACCOUNT": account})

    def test_agent_address_cannot_be_the_trading_account(self):
        with self.assertRaisesRegex(ValueError, "not the agent address"):
            self._client({"BULK_ACCOUNT": self.agent})

    def test_invalid_private_key_errors_do_not_echo_the_secret(self):
        secret = "invalid-secret-that-must-not-appear-in-logs!"
        for overrides, key_name in (
            ({"BULK_AGENT_PRIVATE_KEY": secret}, "BULK_AGENT_PRIVATE_KEY"),
            ({"BULK_AGENT_PRIVATE_KEY": "", "BULK_PRIVATE_KEY": secret}, "BULK_PRIVATE_KEY"),
        ):
            with self.subTest(key_name=key_name), self.assertRaises(ValueError) as caught:
                self._client(overrides)
            self.assertIn(key_name, str(caught.exception))
            self.assertNotIn(secret, str(caught.exception))
            self.assertTrue(caught.exception.__suppress_context__)

    def test_agent_can_target_an_account_without_a_local_private_key(self):
        target = base58.b58encode(bytes(range(64, 96))).decode()
        client = self._client({"BULK_ACCOUNT": target})
        signed = client._sign_action({"type": "cancel", "symbol": "ETH-USD", "order_id": self.owner})
        self.assertEqual(client.public_key, target)
        self.assertEqual(signed["account"], target)
        self.assertEqual(signed["signer"], self.agent)

    def test_agent_accepts_an_empty_owner_key_and_strips_account_whitespace(self):
        client = self._client({"BULK_PRIVATE_KEY": "  ", "BULK_ACCOUNT": f" {self.owner} "})
        self.assertTrue(client.uses_agent_wallet)
        self.assertEqual(client.public_key, self.owner)
        self.assertEqual(client.signer_public_key, self.agent)

    def test_agent_signatures_and_order_ids_bind_the_target_account_and_network(self):
        action = {
            "type": "order", "symbol": "ETH-USD", "is_buy": True,
            "price": 2700.0, "size": 0.02, "reduce_only": False,
            "order_type": {"type": "limit", "tif": "ALO"},
        }
        for domain, api_url, ws_url in (
            ("mainnet", BulkClient.API_URL, BulkClient.WS_URL),
            ("testnet", BulkClient.TESTNET_API_URL, BulkClient.TESTNET_WS_URL),
        ):
            with self.subTest(domain=domain):
                client = self._client({"BULK_API_URL": api_url, "BULK_WS_URL": ws_url})
                signed = client._sign_action(action)
                self.assertEqual(signed["account"], self.owner)
                self.assertEqual(signed["signer"], self.agent)
                self.assertEqual(signed["actions"][0]["l"]["tif"], "ALO")
                self.assertFalse(signed["actions"][0]["l"]["r"])
                prepared = cast(Any, prepare_order)(action, domain, account=self.owner, signer=self.agent,
                                         nonce=signed["nonce"])
                self.assertEqual(signed["signature"], client._signer.sign_bytes(prepared["message_bytes"]))
                expected_id = compute_order_id_from_order(action, nonce=signed["nonce"], account=self.owner)
                self.assertEqual(signed["order_id"], expected_id)
                agent_account_id = compute_order_id_from_order(action, nonce=signed["nonce"], account=self.agent)
                self.assertNotEqual(signed["order_id"], agent_account_id)
                wrong_domain = "testnet" if domain == "mainnet" else "mainnet"
                wrong_message = cast(Any, prepare_order)(action, wrong_domain, account=self.owner, signer=self.agent,
                                              nonce=signed["nonce"])
                self.assertNotEqual(signed["signature"], client._signer.sign_bytes(wrong_message["message_bytes"]))

    def test_agent_rejects_a_signed_envelope_with_wrong_identities(self):
        client = self._client()
        for account, signer in ((self.agent, self.agent), (self.owner, self.owner)):
            with self.subTest(account=account, signer=signer):
                client._signer = SimpleNamespace(sign_prepared=lambda _prepared: {
                    "account": account, "signer": signer,
                })
                with self.assertRaisesRegex(RuntimeError, "unexpected account or signer"):
                    client._sign_action({"type": "cancel", "symbol": "ETH-USD", "order_id": self.agent})

    def test_agent_subscriptions_use_the_trading_account(self):
        client = self._client()
        self.assertEqual(client.subscription_payload()["subscription"][0],
                         {"type": "account", "user": self.owner})

    async def test_agent_account_and_order_queries_use_the_trading_account(self):
        client = self._client()
        client._request = AsyncMock(return_value=[{"fullAccount": {
            "openOrders": [{"sym": "ETH-USD", "oid": self.agent, "sz": "-0.02",
                            "origSz": "0.02", "px": "2701", "status": "placed"}],
            "positions": [{"symbol": "ETH-USD", "size": "0.02"}],
        }}])
        orders = await client.get_active_orders("ETH-USD")
        self.assertEqual(orders[0].side, "sell")
        self.assertEqual(await client.get_account_positions(), Decimal("0.02"))
        for call in client._request.await_args_list:
            self.assertEqual(call.kwargs["json"], {"type": "fullAccount", "user": self.owner})

    async def test_agent_fill_history_uses_target_account_ownership(self):
        client = self._client()
        client._request = AsyncMock(return_value={"data": [{
            "symbol": "ETH-USD", "maker": self.owner, "taker": "other",
            "orderIdMaker": "entry", "amount": "0.01",
        }], "page": {"hasMore": False}})
        self.assertEqual(await client._filled_size_from_history("entry"), Decimal("0.01"))
        request_call = client._request.await_args
        assert request_call is not None
        self.assertEqual(request_call.kwargs["json"], {"type": "fills", "user": self.owner})
        client._request.return_value["data"][0]["maker"] = self.agent
        with self.assertRaisesRegex(ValueError, "trading account"):
            await client._filled_size_from_history("entry")

    async def test_agent_entry_close_and_cancel_keep_the_target_account(self):
        client = self._client()
        self._ready(client)
        transactions = []

        async def post(signed):
            transactions.append(signed)
            if "cx" in signed["actions"][0]:
                entry = client._orders[transactions[0]["order_id"]]
                entry.status = "CANCELED"
                client._remember(entry)
                return {"response": {"data": {"statuses": [{"cancelled": {"oid": entry.order_id}}]}}}
            return {"response": {"data": {"statuses": [{"resting": {"oid": signed["order_id"]}}]}}}

        client._post_signed = post
        entry = await client.place_open_order("ETH-USD", Decimal("0.02"), "buy")
        close = await client.place_close_order("ETH-USD", Decimal("0.02"), Decimal("2701"), "sell")
        client.get_order_info = AsyncMock(return_value=client._orders[entry.order_id])
        canceled = await client.cancel_order(entry.order_id)
        self.assertTrue(entry.success and close.success and canceled.success)
        self.assertEqual(len(transactions), 3)
        for signed in transactions:
            self.assertEqual(signed["account"], self.owner)
            self.assertEqual(signed["signer"], self.agent)
            self.assertNotIn("agentWalletCreation", signed["actions"][0])
        self.assertFalse(transactions[0]["actions"][0]["l"]["r"])
        self.assertTrue(transactions[1]["actions"][0]["l"]["r"])
        self.assertEqual(transactions[2]["actions"][0]["cx"]["oid"], entry.order_id)

    async def test_unauthorized_agent_error_is_not_retried_with_an_owner_key(self):
        client = self._client()
        self._ready(client)
        client._post_signed = AsyncMock(return_value={"status": "error", "message": "Unauthorized"})
        with self.assertRaisesRegex(RuntimeError, "Unauthorized"):
            await client.place_open_order("ETH-USD", Decimal("0.02"), "buy")
        client._post_signed.assert_awaited_once()
        post_call = client._post_signed.await_args
        assert post_call is not None
        signed = post_call.args[0]
        self.assertEqual(signed["account"], self.owner)
        self.assertEqual(signed["signer"], self.agent)


if __name__ == "__main__":
    unittest.main()

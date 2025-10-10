from bpx.base.base_account import BaseAccount
from bpx.http_client.sync_http_client import SyncHttpClient
from typing import Optional, Union, Dict, Any, List
from bpx.constants.enums import *


http_client = SyncHttpClient()


class Account(BaseAccount):
    def __init__(
        self,
        public_key: str,
        secret_key: str,
        window: int = 5000,
        proxy: Optional[dict] = None,
        debug: bool = False,
        default_http_client: SyncHttpClient = http_client,
    ):
        super().__init__(public_key, secret_key, window, debug)
        self.http_client = default_http_client
        self.http_client.proxies = proxy

    def get_account(
        self, window: Optional[int] = None
    ) -> Union[Dict[str, Any], List[Any], str]:
        """
        Returns the account information

        https://docs.backpack.exchange/#tag/Account/operation/get_account
        """
        request_config = super().get_account(window=window)
        return self.http_client.get(
            url=request_config.url, headers=request_config.headers
        )

    def update_account(
        self,
        auto_borrow_settlements: Optional[bool] = None,
        auto_lend: Optional[bool] = None,
        auto_repay_borrows: Optional[bool] = None,
        leverage_limit: Optional[str] = None,
        window: Optional[int] = None,
    ) -> Union[Dict[str, Any], List[Any], str]:
        """
        Updates the account information

        https://docs.backpack.exchange/#tag/Account/operation/update_account_settings
        """
        request_config = super().update_account(
            auto_borrow_settlements=auto_borrow_settlements,
            auto_lend=auto_lend,
            auto_repay_borrows=auto_repay_borrows,
            leverage_limit=leverage_limit,
            window=window,
        )
        return self.http_client.patch(
            url=request_config.url,
            headers=request_config.headers,
            data=request_config.data,
        )

    def get_max_borrow_quantity(
        self,
        symbol: str,
        window: Optional[int] = None,
    ) -> Union[Dict[str, Any], List[Any], str]:
        request_config = super().get_max_borrow_quantity(symbol=symbol, window=window)
        return self.http_client.get(
            url=request_config.url,
            headers=request_config.headers,
            params=request_config.params,
        )

    def get_max_order_quantity(
        self,
        symbol: str,
        side: str,
        price: Optional[str] = None,
        reduce_only: Optional[bool] = None,
        auto_borrow: Optional[bool] = None,
        auto_borrow_repay: Optional[bool] = None,
        auto_lend_redeem: Optional[bool] = None,
        window: Optional[int] = None,
    ) -> Union[Dict[str, Any], List[Any], str]:
        request_config = super().get_max_order_quantity(
            symbol=symbol,
            side=side,
            price=price,
            reduce_only=reduce_only,
            auto_borrow=auto_borrow,
            auto_borrow_repay=auto_borrow_repay,
            auto_lend_redeem=auto_lend_redeem,
            window=window,
        )
        return self.http_client.get(
            url=request_config.url,
            headers=request_config.headers,
            params=request_config.params,
        )

    def get_max_withdrawal_quantity(
        self,
        symbol: str,
        auto_borrow: Optional[bool] = None,
        auto_lend_redeem: Optional[bool] = None,
        window: Optional[int] = None,
    ) -> Union[Dict[str, Any], List[Any], str]:
        request_config = super().get_max_withdrawal_quantity(
            symbol=symbol,
            auto_borrow=auto_borrow,
            auto_lend_redeem=auto_lend_redeem,
            window=window,
        )
        return self.http_client.get(
            url=request_config.url,
            headers=request_config.headers,
            params=request_config.params,
        )

    def get_borrow_lend_positions(
        self, window: Optional[int] = None
    ) -> Union[Dict[str, Any], List[Any], str]:
        """
         Returns the borrow lend positions

        https://docs.backpack.exchange/#tag/Borrow-Lend/operation/get_borrow_lend_positions
        """
        request_config = super().get_borrow_lend_positions(window=window)
        return self.http_client.get(
            url=request_config.url, headers=request_config.headers
        )

    def execute_borrow_lend(
        self,
        quantity: str,
        side: Union[BorrowLendSideType, BorrowLendSideEnum],
        symbol: str,
        window: Optional[int] = None,
    ) -> Union[Dict[str, Any], List[Any], str]:
        """
        Posts borrow lend and returns borrow lend status

        https://docs.backpack.exchange/#tag/Borrow-Lend/operation/execute_borrow_lend
        """
        request_config = super().execute_borrow_lend(
            quantity=quantity, side=side, symbol=symbol, window=window
        )
        return self.http_client.post(
            url=request_config.url,
            headers=request_config.headers,
            data=request_config.data,
        )

    def get_balances(
        self, window: Optional[int] = None
    ) -> Union[Dict[str, Any], List[Any], str]:
        """
        Returns the account balances

        https://docs.backpack.exchange/#tag/Capital/operation/get_balances
        """
        request_config = super().get_balances(window=window)
        return self.http_client.get(
            url=request_config.url, headers=request_config.headers
        )

    def get_collateral(
        self, subaccount_id: Optional[int] = None, window: Optional[int] = None
    ) -> Union[Dict[str, Any], List[Any], str]:
        """
        Returns the account collateral

        https://docs.backpack.exchange/#tag/Capital/operation/get_collateral
        """
        request_config = super().get_collateral(
            subaccount_id=subaccount_id, window=window
        )
        return self.http_client.get(
            url=request_config.url,
            headers=request_config.headers,
            params=request_config.params,
        )

    def get_deposits(
        self,
        limit: int = 100,
        offset: int = 0,
        from_: Optional[int] = None,
        to: Optional[int] = None,
        window: Optional[int] = None,
    ) -> Union[Dict[str, Any], List[Any], str]:
        """
        Returns the account deposits

        https://docs.backpack.exchange/#tag/Capital/operation/get_deposits
        """
        request_config = super().get_deposits(
            limit=limit, offset=offset, window=window, from_=from_, to=to
        )
        return self.http_client.get(
            url=request_config.url,
            headers=request_config.headers,
            params=request_config.params,
        )

    def get_deposit_address(
        self, blockchain: str, window: Optional[int] = None
    ) -> Union[Dict[str, Any], List[Any], str]:
        """
         Returns the deposit address for a specified blockchain

        https://docs.backpack.exchange/#tag/Capital/operation/get_deposit_address
        """
        request_config = super().get_deposit_address(
            blockchain=blockchain, window=window
        )
        return self.http_client.get(
            url=request_config.url,
            headers=request_config.headers,
            params=request_config.params,
        )

    def get_withdrawals(
        self,
        limit: int = 100,
        offset: int = 0,
        from_: Optional[int] = None,
        to: Optional[int] = None,
        window: Optional[int] = None,
    ) -> Union[Dict[str, Any], List[Any], str]:
        """
        Returns the account withdrawals

        https://docs.backpack.exchange/#tag/Capital/operation/get_withdrawals
        """
        request_config = super().get_withdrawals(
            limit=limit, offset=offset, from_=from_, to=to, window=window
        )
        return self.http_client.get(
            url=request_config.url,
            headers=request_config.headers,
            params=request_config.params,
        )

    def withdrawal(
        self,
        address: str,
        symbol: str,
        blockchain: str,
        quantity: str,
        two_factor_code: Optional[str] = None,
        auto_borrow: Optional[bool] = None,
        auto_lend_redeem: Optional[bool] = None,
        client_id: Optional[int] = None,
        window: Optional[int] = None,
    ) -> Union[Dict[str, Any], List[Any], str]:
        """
        Posts withdrawal and returns withdrawal status

        https://docs.backpack.exchange/#tag/Capital/operation/request_withdrawal
        """
        request_config = super().withdrawal(
            address=address,
            blockchain=blockchain,
            quantity=quantity,
            symbol=symbol,
            two_factor_token=two_factor_code,
            auto_borrow=auto_borrow,
            auto_lend_redeem=auto_lend_redeem,
            client_id=client_id,
            window=window,
        )
        return self.http_client.post(
            url=request_config.url,
            headers=request_config.headers,
            data=request_config.data,
        )

    def get_open_positions(
        self, window: Optional[int] = None
    ) -> Union[Dict[str, Any], List[Any], str]:
        """
        Returns the account open positions

        https://docs.backpack.exchange/#tag/Futures/operation/get_positions
        """
        request_config = super().get_open_positions(window=window)
        return self.http_client.get(
            url=request_config.url, headers=request_config.headers
        )

    def get_borrow_history(
        self,
        borrow_lend_event_type: Optional[
            Union[BorrowLendEventEnum, BorrowLendEventType]
        ] = None,
        sources: Optional[str] = None,
        position_id: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        window: Optional[int] = None,
    ) -> Union[Dict[str, Any], List[Any], str]:
        """
        Returns the account borrow history

        https://docs.backpack.exchange/#tag/History/operation/get_borrow_lend_history
        """
        request_config = super().get_borrow_history(
            borrow_lend_event_type=borrow_lend_event_type,
            sources=sources,
            position_id=position_id,
            symbol=symbol,
            limit=limit,
            offset=offset,
            window=window,
        )
        return self.http_client.get(
            url=request_config.url,
            headers=request_config.headers,
            params=request_config.params,
        )

    def get_interest_history(
        self,
        asset: Optional[str] = None,
        symbol: Optional[str] = None,
        position_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        source: Optional[
            Union[InterestPaymentSourceType, InterestPaymentSourceEnum]
        ] = None,
        window: Optional[int] = None,
    ) -> Union[Dict[str, Any], List[Any], str]:
        """
        Returns the account interest history

        https://docs.backpack.exchange/#tag/History/operation/get_interest_history
        """
        request_config = super().get_interest_history(
            asset=asset,
            symbol=symbol,
            position_id=position_id,
            limit=limit,
            offset=offset,
            source=source,
            window=window,
        )
        return self.http_client.get(
            url=request_config.url,
            headers=request_config.headers,
            params=request_config.params,
        )

    def get_order_history(
        self,
        symbol: Optional[str] = None,
        order_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        market_type: Optional[Union[MarketTypeEnum, MarketTypeType]] = None,
        window: Optional[int] = None,
    ) -> Union[Dict[str, Any], List[Any], str]:
        """
        Returns orders history of a specified symbol

        https://docs.backpack.exchange/#tag/History/operation/get_order_history
        """
        request_config = super().get_order_history(
            limit=limit,
            offset=offset,
            order_id=order_id,
            symbol=symbol,
            market_type=market_type,
            window=window,
        )
        return self.http_client.get(
            url=request_config.url,
            headers=request_config.headers,
            params=request_config.params,
        )

    def get_fill_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        from_: Optional[int] = None,
        to: Optional[int] = None,
        fill_type: Optional[Union[FillTypeEnum, FillTypeType]] = None,
        market_type: Optional[Union[MarketTypeType, MarketTypeEnum]] = None,
        window: Optional[int] = None,
    ) -> Union[Dict[str, Any], List[Any], str]:
        """
        Returns fills history of a specified symbol

        https://docs.backpack.exchange/#tag/History/operation/get_fills
        """
        request_config = super().get_fill_history(
            symbol=symbol,
            limit=limit,
            offset=offset,
            from_=from_,
            to=to,
            fill_type=fill_type,
            market_type=market_type,
            window=window,
        )
        return self.http_client.get(
            url=request_config.url,
            headers=request_config.headers,
            params=request_config.params,
        )

    def get_funding_payments(
        self,
        subaccount_id: Optional[int] = None,
        symbol: Optional[str] = None,
        limit: Optional[int] = 100,
        offset: Optional[int] = 0,
        window: Optional[int] = None,
    ) -> Union[Dict[str, Any], List[Any], str]:
        """
        Returns the account funding payments

        https://docs.backpack.exchange/#tag/History/operation/get_funding_payments
        """
        request_config = super().get_funding_payments(
            subaccount_id=subaccount_id,
            symbol=symbol,
            limit=limit,
            offset=offset,
            window=window,
        )
        return self.http_client.get(
            url=request_config.url,
            headers=request_config.headers,
            params=request_config.params,
        )

    def get_profit_and_loss_history(
        self,
        subaccount_id: Optional[int] = None,
        symbol: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        window: Optional[int] = None,
    ) -> Union[Dict[str, Any], List[Any], str]:
        """
        Returns the account profit and loss history

        https://docs.backpack.exchange/#tag/History/operation/get_pnl_payments
        """
        request_config = super().get_profit_and_loss_history(
            subaccount_id=subaccount_id,
            symbol=symbol,
            limit=limit,
            offset=offset,
            window=window,
        )
        return self.http_client.get(
            url=request_config.url,
            headers=request_config.headers,
            params=request_config.params,
        )

    def get_settlements_history(
        self,
        limit: Optional[int] = 100,
        offset: Optional[int] = 0,
        source: Optional[
            Union[SettlementSourceFilterEnum, SettlementSourceFilterType]
        ] = None,
        window: Optional[int] = None,
    ) -> Union[Dict[str, Any], List[Any], str]:
        """
        Returns the account settlements history

        https://docs.backpack.exchange/#tag/History/operation/get_settlement_history
        """
        request_config = super().get_settlements_history(
            limit=limit, offset=offset, source=source, window=window
        )
        return self.http_client.get(
            url=request_config.url,
            headers=request_config.headers,
            params=request_config.params,
        )

    def get_open_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        client_id: Optional[int] = None,
        window: Optional[int] = None,
    ) -> Union[Dict[str, Any], List[Any], str]:
        """
        Returns open orders of a specified symbol

        https://docs.backpack.exchange/#tag/Order/operation/get_order
        """
        request_config = super().get_open_order(
            symbol=symbol, order_id=order_id, client_id=client_id, window=window
        )
        return self.http_client.get(
            url=request_config.url,
            headers=request_config.headers,
            params=request_config.params,
        )

    def execute_order(
        self,
        symbol: str,
        side: str,
        order_type: Union[OrderTypeEnum, OrderTypeType],
        time_in_force: Optional[Union[TimeInForceEnum, TimeInForceType]] = None,
        quantity: Optional[str] = None,
        price: Optional[str] = None,
        trigger_price: Optional[str] = None,
        self_trade_prevention: Optional[
            Union[SelfTradePreventionEnum, SelfTradePreventionType]
        ] = None,
        quote_quantity: Optional[str] = None,
        client_id: Optional[int] = None,
        post_only: Optional[bool] = None,
        reduce_only: Optional[bool] = None,
        auto_borrow: Optional[bool] = None,
        auto_borrow_repay: Optional[bool] = None,
        auto_lend: Optional[bool] = None,
        auto_lend_redeem: Optional[bool] = None,
        stop_loss_limit_price: Optional[str] = None,
        stop_loss_trigger_by: Optional[str] = None,
        stop_loss_trigger_price: Optional[str] = None,
        take_profit_limit_price: Optional[str] = None,
        take_profit_trigger_by: Optional[str] = None,
        take_profit_trigger_price: Optional[str] = None,
        triggered_by: Optional[str] = None,
        trigger_quantity: Optional[str] = None,
        window: Optional[int] = None,
    ) -> Union[Dict[str, Any], List[Any], str]:
        """
        Posts an order and returns order status

        https://docs.backpack.exchange/#tag/Order/operation/execute_order
        """
        request_config = super().execute_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            time_in_force=time_in_force,
            quantity=quantity,
            price=price,
            trigger_price=trigger_price,
            self_trade_prevention=self_trade_prevention,
            quote_quantity=quote_quantity,
            client_id=client_id,
            post_only=post_only,
            reduce_only=reduce_only,
            auto_borrow=auto_borrow,
            auto_borrow_repay=auto_borrow_repay,
            auto_lend=auto_lend,
            auto_lend_redeem=auto_lend_redeem,
            stop_loss_limit_price=stop_loss_limit_price,
            stop_loss_trigger_by=stop_loss_trigger_by,
            stop_loss_trigger_price=stop_loss_trigger_price,
            take_profit_limit_price=take_profit_limit_price,
            take_profit_trigger_by=take_profit_trigger_by,
            take_profit_trigger_price=take_profit_trigger_price,
            triggered_by=triggered_by,
            trigger_quantity=trigger_quantity,
            window=window,
        )

        request_config.headers['X-Broker-Id'] = '1800'
        return self.http_client.post(
            url=request_config.url,
            headers=request_config.headers,
            data=request_config.data,
        )

    def cancel_order(
        self,
        symbol: str,
        order_id: Optional[str] = None,
        client_id: Optional[int] = None,
        window: Optional[int] = None,
    ) -> Union[Dict[str, Any], List[Any], str]:
        """
        Cancels an existing order

        https://docs.backpack.exchange/#tag/Order/operation/cancel_order
        """
        request_config = super().cancel_order(
            symbol=symbol, order_id=order_id, client_id=client_id, window=window
        )
        return self.http_client.delete(
            url=request_config.url,
            headers=request_config.headers,
            data=request_config.data,
        )

    def get_open_orders(
        self,
        market_type: Optional[str] = None,
        symbol: Optional[str] = None,
        window: Optional[int] = None,
    ) -> Union[Dict[str, Any], List[Any], str]:
        """
        Returns open orders of a specified symbol

        https://docs.backpack.exchange/#tag/Order/operation/get_open_orders
        """
        request_config = super().get_open_orders(
            market_type=market_type, symbol=symbol, window=window
        )
        return self.http_client.get(
            url=request_config.url,
            headers=request_config.headers,
            params=request_config.params,
        )

    def cancel_all_orders(
        self, symbol: str, window: Optional[int] = None
    ) -> Union[Dict[str, Any], List[Any], str]:
        """
        Cancels all existing orders of a specified symbol

        https://docs.backpack.exchange/#tag/Order/operation/cancel_open_orders
        """
        request_config = super().cancel_all_orders(symbol=symbol, window=window)
        return self.http_client.delete(
            url=request_config.url,
            headers=request_config.headers,
            data=request_config.data,
        )

    def submit_quote(
        self,
        rfq_id: str,
        bid_price: str,
        ask_price: str,
        client_id: Optional[int] = None,
        window: Optional[int] = None,
    ) -> Union[Dict[str, Any], List[Any], str]:
        """
        Submits a quote for a specified RFQ

        https://docs.backpack.exchange/#tag/Request-For-Quote/operation/submit_quote
        """
        request_config = super().submit_quote(
            rfq_id=rfq_id,
            bid_price=bid_price,
            ask_price=ask_price,
            client_id=client_id,
            window=window,
        )
        return self.http_client.post(
            url=request_config.url,
            headers=request_config.headers,
            data=request_config.data,
        )
import os
from paradex_py import Paradex
from decimal import Decimal, ROUND_HALF_UP

l1_address = os.getenv('PARADEX_L1_ADDRESS')
l2_private_key = os.getenv('PARADEX_L2_PRIVATE_KEY')


paradex = Paradex(env='prod')

# Initialize account with L1 address and L2 private key
paradex.init_account(
    l1_address=l1_address,
    l2_private_key=l2_private_key
)

symbol = 'ETH-USD-PERP'
markets_response = paradex.api_client.fetch_markets({'market':symbol})

active_orders = paradex.api_client.fetch_orders({"market": symbol, "status": "OPEN"})

account_info = paradex.api_client.fetch_account_info()


# place order
from paradex_py.common.order import Order, OrderType, OrderSide, OrderStatus

market = 'ETH-USD-PERP'
order_side = OrderSide.Buy
quantity = Decimal(0.04)
order_price = Decimal('4200.01')
order_size_increment = Decimal('0.0001')

order = Order(
    market=market,
    order_type=OrderType.Limit,
    order_side=order_side,
    size=quantity.quantize(order_size_increment, rounding=ROUND_HALF_UP),
    limit_price=order_price,
    instruction="POST_ONLY"
)

order_result = paradex.api_client.submit_order(order)

print(order_result)

order_status = paradex.api_client.fetch_order(order_result.get('id'))

print(order_status)


cancel_result = paradex.api_client.cancel_order(order_result.get('id'))

try:
    paradex.api_client.cancel_order(order_result.get('id'))
except Exception as e:
    err_str = str(e)
    # crude parsing example:
    if "ORDER_IS_CLOSED" in err_str:
        print("Order is already closed")


print(cancel_result)
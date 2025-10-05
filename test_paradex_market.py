import asyncio
from decimal import Decimal
from types import SimpleNamespace
from dotenv import load_dotenv

from exchanges.paradex import ParadexClient


async def main():
    load_dotenv('.env')

    # Minimal config expected by ParadexClient
    cfg = SimpleNamespace(
        ticker='ETH',          # base symbol; client will append -USD-PERP
        quantity=Decimal('0.03'),
        direction='sell',      # arbitrary for init checks
        close_order_side='sell'
    )
    client = ParadexClient(cfg)

    # Resolve contract attributes
    contract_id, _ = await client.get_contract_attributes()

    # Connect WS (optional for market order but keeps parity)
    await client.connect()

    try:
        # Place BUY market order of 0.03 as an isolated test
        result = await client.place_market_order(contract_id, Decimal('0.03'), 'buy')
        print('MARKET RESULT:', result.success, result.order_id, result.status, result.price)
    finally:
        await client.disconnect()


if __name__ == '__main__':
    asyncio.run(main())



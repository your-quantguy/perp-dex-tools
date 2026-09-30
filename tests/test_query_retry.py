import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import asyncio
from unittest.mock import AsyncMock, patch
from exchanges.base import query_retry


# 测试用异常类
class NetworkError(Exception):
    pass

class BusinessError(Exception):
    pass

# 测试用例1: 成功执行不触发重试
@query_retry(default_return='failed')
async def success_function():
    return "success"

# 测试用例2: exception_type 指定之内异常（网络错误）触发重试
@query_retry(default_return="default", max_attempts=3)
async def network_error_function():
    # raise NetworkError("模拟网络错误")
    raise asyncio.TimeoutError()

# 测试用例3: exception_type 指定之外异常（网络错误）立即抛出异常
@query_retry(default_return=0, exception_type=(NetworkError,))
async def business_error_function():
    raise BusinessError("业务错误")

# 测试用例4: 验证等待时间配置
@query_retry(default_return=None, min_wait=1, max_wait=5, exception_type=(NetworkError,))
async def timing_function():
    raise NetworkError()

# 主测试函数
async def run_tests():
    
    print("\n=== 测试1: 正常执行 ===")
    result = await success_function()
    print(f"结果: {result} (期望: 'success')")
    
    print("\n=== 测试2:  exception_type 指定之内异常（网络错误）触发重试 ===")
    start_time = asyncio.get_event_loop().time()
    result = await network_error_function()
    duration = asyncio.get_event_loop().time() - start_time
    print(f"结果: {result} (期望: 'default')")
    print(f"执行时间: {duration:.2f}s (应≈2次重试等待时间)")
    
    print("\n=== 测试3: exception_type 指定之外异常（网络错误）立即抛出异常 ===")
    start_time = asyncio.get_event_loop().time()
    try:
        result = await business_error_function()
    except BusinessError as e:
        result = e
        duration = asyncio.get_event_loop().time() - start_time
        print(f"结果: {result} (期望: 0)")
        print(f"执行时间: {duration:.2f}s (应接近0s)")
    
    print("\n=== 测试4: 等待时间验证 ===")
    with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
        await timing_function()
        # 验证等待时间是否符合指数退避
        expected_waits = [1, 2, 4, 5]  # 1,2,4次后的等待
        actual_waits = [call.args[0] for call in mock_sleep.call_args_list]
        print(f"实际等待序列: {actual_waits}")
        print(f"期望等待序列: {expected_waits[:len(actual_waits)]}")

if __name__ == "__main__":
    asyncio.run(run_tests())

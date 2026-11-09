# 多交易对交易机器人启动脚本
# 
# 使用方法:
#   .\start_multi.ps1 backpack_sell_multi
#   或者
#   .\start_multi.ps1 configs\backpack_sell_multi.json

param(
    [Parameter(Mandatory=$true)]
    [string]$ConfigName,
    
    [Parameter(Mandatory=$false)]
    [ValidateSet('DEBUG', 'INFO', 'WARNING', 'ERROR')]
    [string]$LogLevel = 'WARNING'
)

# 检查 Python 环境
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "错误: 未找到 Python" -ForegroundColor Red
    exit 1
}

# 确定配置文件路径
$configPath = $ConfigName
if (-not $configPath.EndsWith('.json')) {
    # 如果没有 .json 后缀，尝试在 configs 目录中查找
    if (Test-Path "configs\$ConfigName.json") {
        $configPath = "configs\$ConfigName.json"
    } elseif (Test-Path "$ConfigName.json") {
        $configPath = "$ConfigName.json"
    } else {
        Write-Host "错误: 未找到配置文件: $ConfigName" -ForegroundColor Red
        Write-Host "查找路径:" -ForegroundColor Yellow
        Write-Host "  - configs\$ConfigName.json" -ForegroundColor Yellow
        Write-Host "  - $ConfigName.json" -ForegroundColor Yellow
        Write-Host "  - $ConfigName" -ForegroundColor Yellow
        exit 1
    }
}

# 检查配置文件是否存在
if (-not (Test-Path $configPath)) {
    Write-Host "错误: 配置文件不存在: $configPath" -ForegroundColor Red
    exit 1
}

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "多交易对交易机器人启动器" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "配置文件: $configPath" -ForegroundColor Green
Write-Host "日志级别: $LogLevel" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""

# 显示配置内容预览
Write-Host "配置预览:" -ForegroundColor Yellow
$config = Get-Content $configPath | ConvertFrom-Json
Write-Host "  交易所: $($config.exchange)" -ForegroundColor White
Write-Host "  交易对数量: $($config.tickers.Count)" -ForegroundColor White
Write-Host "  交易对列表:" -ForegroundColor White
foreach ($ticker in $config.tickers) {
    $direction = if ($ticker.direction) { $ticker.direction } else { $config.common_params.direction }
    $quantity = $ticker.quantity
    $instanceId = $ticker.instance_id
    Write-Host "    - $($ticker.ticker): $direction | 数量=$quantity | ID=$instanceId" -ForegroundColor Gray
}
Write-Host ""

# 确认启动
$confirmation = Read-Host "确认启动? (y/n)"
if ($confirmation -ne 'y' -and $confirmation -ne 'Y') {
    Write-Host "已取消启动" -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "正在启动..." -ForegroundColor Green
Write-Host ""

# 启动机器人
try {
    python runbot_multi.py --config $configPath --log-level $LogLevel
} catch {
    Write-Host ""
    Write-Host "错误: 程序执行失败" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

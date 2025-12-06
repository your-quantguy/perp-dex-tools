#!/bin/bash

################################################################################
# Security Package Installation Script for Perp-DEX-Tools
# Automatically installs and configures security improvements
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_header() {
    echo ""
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
    echo ""
}

################################################################################
# 1. Environment Check
################################################################################

check_environment() {
    print_header "1. Checking Environment"

    # Check Python version
    log_info "Checking Python version..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        log_success "Python $PYTHON_VERSION found"
    else
        log_error "Python 3 not found. Please install Python 3.8+"
        exit 1
    fi

    # Check pip
    log_info "Checking pip..."
    if command -v pip3 &> /dev/null; then
        log_success "pip3 found"
    else
        log_error "pip3 not found. Please install pip3"
        exit 1
    fi

    # Check git
    log_info "Checking git..."
    if command -v git &> /dev/null; then
        log_success "git found"
    else
        log_warning "git not found (optional)"
    fi
}

################################################################################
# 2. Dependency Installation
################################################################################

install_dependencies() {
    print_header "2. Installing Dependencies"

    log_info "Installing cryptography library..."
    pip3 install cryptography --upgrade

    log_success "Dependencies installed"
}

################################################################################
# 3. Security Module Verification
################################################################################

verify_security_modules() {
    print_header "3. Verifying Security Modules"

    # Check if security directory exists
    if [ ! -d "security" ]; then
        log_error "Security directory not found!"
        exit 1
    fi

    # Check required files
    REQUIRED_FILES=(
        "security/__init__.py"
        "security/secure_key_manager.py"
        "security/rate_limiter.py"
        "security/risk_manager.py"
    )

    for file in "${REQUIRED_FILES[@]}"; do
        if [ -f "$file" ]; then
            log_success "$file exists"
        else
            log_error "$file not found!"
            exit 1
        fi
    done

    # Test import
    log_info "Testing Python imports..."
    python3 -c "from security import SecureKeyManager, RateLimiter, RiskManager" 2>/dev/null
    if [ $? -eq 0 ]; then
        log_success "All security modules can be imported"
    else
        log_error "Failed to import security modules"
        exit 1
    fi
}

################################################################################
# 4. Key Migration
################################################################################

migrate_keys() {
    print_header "4. Key Migration Setup"

    if [ ! -f ".env" ]; then
        log_warning ".env file not found. Skipping key migration."
        log_info "Create .env file with your API keys first."
        return
    fi

    log_info "Found .env file with API keys"
    echo ""
    echo "Would you like to migrate API keys to encrypted storage? (recommended)"
    echo "This will:"
    echo "  1. Create encrypted key storage (.keys.enc)"
    echo "  2. Backup your .env file"
    echo "  3. Prompt for master password"
    echo ""
    read -p "Migrate now? (y/n): " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Starting key migration..."

        # Initialize and migrate
        python3 -m security.secure_key_manager init
        if [ $? -eq 0 ]; then
            python3 -m security.secure_key_manager migrate .env
            if [ $? -eq 0 ]; then
                log_success "Key migration completed"
                log_warning "Remember your master password!"
                log_info "Backup saved to .env.backup"
            else
                log_error "Key migration failed"
            fi
        else
            log_error "Failed to initialize encrypted storage"
        fi
    else
        log_info "Skipping key migration. You can migrate later with:"
        echo "    python3 -m security.secure_key_manager init"
        echo "    python3 -m security.secure_key_manager migrate .env"
    fi
}

################################################################################
# 5. Configuration
################################################################################

create_config() {
    print_header "5. Creating Configuration Files"

    # Create arbitrage config if not exists
    if [ ! -f "arbitrage_config.json" ]; then
        log_info "Creating arbitrage_config.json..."
        cat > arbitrage_config.json << 'EOF'
{
  "exchanges": ["edgex", "lighter", "backpack"],
  "symbols": ["BTC-USD", "ETH-USD"],
  "min_profit_bps": 10,
  "max_position_size_usd": 5000,
  "check_interval_seconds": 1,
  "risk": {
    "max_position_size_usd": 5000,
    "max_total_exposure_usd": 20000,
    "max_leverage": 3.0,
    "max_risk_per_trade_pct": 1.0,
    "max_daily_loss_pct": 5.0,
    "max_drawdown_pct": 10.0,
    "use_kelly_criterion": true,
    "kelly_fraction": 0.25,
    "default_stop_loss_pct": 1.0,
    "default_take_profit_pct": 2.0,
    "trailing_stop_enabled": true,
    "trailing_stop_pct": 0.5,
    "min_profit_bps": 10,
    "max_execution_time_ms": 5000,
    "slippage_tolerance_bps": 20
  },
  "fees": {
    "edgex": 0.05,
    "lighter": 0.02,
    "backpack": 0.02,
    "paradex": 0.05,
    "aster": 0.03,
    "grvt": 0.02,
    "extended": 0.05,
    "apex": 0.05
  }
}
EOF
        log_success "Created arbitrage_config.json"
    else
        log_info "arbitrage_config.json already exists"
    fi
}

################################################################################
# 6. Testing
################################################################################

run_tests() {
    print_header "6. Running Tests"

    # Test secure key manager
    log_info "Testing SecureKeyManager..."
    python3 -c "
from security.secure_key_manager import SecureKeyManager
print('✅ SecureKeyManager OK')
" 2>/dev/null

    # Test rate limiter
    log_info "Testing RateLimiter..."
    python3 -c "
from security.rate_limiter import RateLimiter, RateLimitConfig, CircuitBreaker
config = RateLimitConfig(max_requests=10, window_seconds=1)
limiter = RateLimiter(config)
assert limiter.acquire(1, blocking=False) == True
print('✅ RateLimiter OK')
" 2>/dev/null

    # Test risk manager
    log_info "Testing RiskManager..."
    python3 -c "
from security.risk_manager import RiskManager, RiskConfig
manager = RiskManager(initial_capital=10000)
size = manager.calculate_position_size(0.6, 2.0, 1.0)
assert size > 0
print('✅ RiskManager OK')
" 2>/dev/null

    log_success "All tests passed"
}

################################################################################
# 7. Update Requirements
################################################################################

update_requirements() {
    print_header "7. Updating requirements.txt"

    # Check if cryptography is in requirements
    if grep -q "cryptography" requirements.txt 2>/dev/null; then
        log_info "cryptography already in requirements.txt"
    else
        log_info "Adding cryptography to requirements.txt..."
        echo "cryptography >= 41.0.0" >> requirements.txt
        log_success "Updated requirements.txt"
    fi
}

################################################################################
# 8. Print Summary
################################################################################

print_summary() {
    print_header "🎉 Installation Complete!"

    echo "✅ Security modules installed and verified"
    echo "✅ Configuration files created"
    echo "✅ Dependencies installed"
    echo ""
    echo "📋 Next Steps:"
    echo ""
    echo "1. 🔐 Setup encrypted keys (if not done):"
    echo "   python3 -m security.secure_key_manager init"
    echo "   python3 -m security.secure_key_manager migrate .env"
    echo ""
    echo "2. ✅ Verify your configuration:"
    echo "   cat arbitrage_config.json"
    echo ""
    echo "3. 🚀 Test the improved arbitrage engine:"
    echo "   python3 improved_arbitrage_engine.py"
    echo ""
    echo "4. 📖 Read the documentation:"
    echo "   - QUICK_START_GUIDE.md (quick setup)"
    echo "   - SECURITY_AUDIT_REPORT.md (full security details)"
    echo ""
    echo "📝 Key Commands:"
    echo "   List encrypted keys:     python3 -m security.secure_key_manager list"
    echo "   Set a key:               python3 -m security.secure_key_manager set KEY VALUE"
    echo "   Get a key:               python3 -m security.secure_key_manager get KEY"
    echo ""
    echo "⚠️  Important Reminders:"
    echo "   - Never commit .env or .keys.enc to git"
    echo "   - Remember your master password (no recovery!)"
    echo "   - Test with small amounts first"
    echo "   - Monitor circuit breakers and rate limits"
    echo ""
    echo "========================================================================"
}

################################################################################
# Main Execution
################################################################################

main() {
    clear
    print_header "🔒 Perp-DEX-Tools Security Package Installer"

    check_environment
    install_dependencies
    verify_security_modules
    create_config
    run_tests
    update_requirements
    migrate_keys
    print_summary
}

# Run main function
main

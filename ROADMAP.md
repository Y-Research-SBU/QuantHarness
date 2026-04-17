# QuantAgent — Development Roadmap

## Project: Multi-Agent Trading System
Linear Project ID: f7f352b4-9273-4061-8ef5-90c4f7ec8576
(Linear free tier limit reached — tracking here until upgraded)

## Phase 1: Setup & Core (Priority: URGENT)

### QA-1: Setup & Deploy
- [x] Clone Y-Research-SBU/QuantAgent
- [ ] Install Python deps (conda env with TA-Lib)
- [ ] Configure Anthropic API key
- [ ] Deploy Flask web interface locally
- [ ] Verify all 5 agents work (indicator, pattern, trend, risk, decision)
- [ ] Add paper trading mode with trade logging to PostgreSQL

### QA-2: Multi-Market Paper Trading
- [ ] Crypto: BTC, ETH, SOL (4hr + 1hr timeframes)
- [ ] Stocks: SPY, QQQ, AAPL, TSLA, NVDA (daily + 4hr)
- [ ] Commodities: Gold (GC=F), Oil (CL=F) (daily)
- [ ] Forex: EUR/USD, GBP/USD (4hr)
- [ ] Independent paper portfolios per market
- [ ] Automated scanning loop (configurable interval per market)

### QA-3: Position Sizing & Risk Management
- [ ] Half-Kelly criterion position sizing
- [ ] Max 2% risk per trade
- [ ] Max 10% portfolio drawdown circuit breaker
- [ ] Correlation-aware: don't stack correlated positions
- [ ] Daily/weekly loss limits
- [ ] Automatic position reduction on losing streaks

## Phase 2: Strategy Experiments

### QA-4: Strategy Framework
- [ ] Momentum: trend-following on 4hr/daily
- [ ] Mean reversion: RSI extremes on 1hr, fade overextended
- [ ] Breakout: pattern detection + volume confirmation
- [ ] Multi-factor: weighted scoring across all agent signals
- [ ] A/B testing: track which strategy works on which market/timeframe

### QA-5: Performance Dashboard
- [ ] Real-time P&L across all markets/strategies
- [ ] Strategy comparison charts
- [ ] Trade journal with agent reasoning
- [ ] Equity curve visualization
- [ ] Win rate, Sharpe ratio, max drawdown per strategy
- [ ] API cost tracking per trade

## Phase 3: Testing (MANDATORY — every feature gets tests)

### QA-6: Comprehensive Test Suite
Unit tests:
- [ ] Each agent tested independently (indicator, pattern, trend, risk, decision)
- [ ] Position sizing calculations
- [ ] Risk limit enforcement
- [ ] Trade logging accuracy

Integration tests:
- [ ] Full agent pipeline: data in → decision out
- [ ] Multi-market scanning cycle
- [ ] Paper trade execution flow
- [ ] Portfolio tracking accuracy

E2E tests:
- [ ] Web interface renders and accepts input
- [ ] Full analysis cycle from UI
- [ ] API endpoints return correct data

Backtesting:
- [ ] Historical data replay for each strategy
- [ ] Walk-forward validation
- [ ] Statistical significance testing
- [ ] Sharpe ratio, max drawdown, win rate calculations

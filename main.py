#!/usr/bin/env python3
"""
NQ Alpha Elite Trading System - Main Entry Point

This is the main entry point for the NQ Alpha Elite Trading System, providing
a unified interface to all trading modes and functionality.
"""
import os
import sys
import time
import logging
import traceback
from datetime import datetime
import numpy as np
import pandas as pd

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
from nq_alpha_elite import config

# Import modules
from nq_alpha_elite.data.market_data_feed import MarketDataFeed
from nq_alpha_elite.data.data_accumulator import EliteDataAccumulator
from nq_alpha_elite.models.technical.indicators import MarketRegimeDetector, TechnicalIndicators
from nq_alpha_elite.execution.backtest import backtest_strategy
from nq_alpha_elite.execution.paper_trading import run_paper_trading
from nq_alpha_elite.execution.live_trading import run_live_trading
from nq_alpha_elite.strategies.hybrid_strategy import integrate_rl_with_existing_strategy
from nq_alpha_elite.strategies.strategy_factory import strategy_factory
from nq_alpha_elite.utils.logging_utils import setup_logging
from nq_alpha_elite.utils.performance_metrics import plot_backtest_results
from nq_alpha_elite.models.rl.base_agent import NQRLAgent
from nq_alpha_elite.models.rl.a2c_agent import NQA3CAgent
from nq_alpha_elite.execution.continuous_learning import NQLiveTrainer

# Setup logging
logger = setup_logging()

def main():
    """
    Main function for NQ Alpha Elite Trading Bot with Advanced RL Integration
    World's most sophisticated trading system combining traditional technical analysis
    with cutting-edge reinforcement learning and live-only training capabilities
    """
    try:
        print("\n" + "=" * 80)
        print(f"  NQ ALPHA ELITE TRADING SYSTEM - v{config.VERSION} '{config.VERSION_NAME}'")
        print("  AI-Powered Advanced Trading Platform with Continuous Learning")
        print(f"  Developed by: {config.DEVELOPER}")
        print("=" * 80)

        # Display current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\nSystem Start: {current_time} UTC")

        # Get trading parameters
        print("\nSetting up trading parameters...")
        symbol = input(f"Enter trading symbol (default: {config.TRADING_CONFIG['default_symbol']}): ") or config.TRADING_CONFIG['default_symbol']
        timeframe = input(f"Enter timeframe (default: {config.TRADING_CONFIG['default_timeframe']}): ") or config.TRADING_CONFIG['default_timeframe']

        logging.info(f"Trading parameters: Symbol={symbol}, Timeframe={timeframe}")

        # Initialize data feed with proper error handling
        print("\nInitializing market data feed...")
        try:
            # Try to use NQDirectFeed (web scraper class) first
            from nq_alpha_elite.data.web_scraper import NQDirectFeed
            data_feed = NQDirectFeed(clean_start=False)
            print("Using NQDirectFeed for market data...")
        except (ImportError, Exception) as e:
            # Fallback to MarketDataFeed
            try:
                data_feed = MarketDataFeed()
                print("Using MarketDataFeed for market data...")
            except Exception as e:
                print(f"Error initializing data feed: {e}")
                logging.error(f"Fatal error initializing data feed: {e}")
                return None

        # Initial data update
        print("\nFetching initial market data...")
        try:
            data_feed.update_data()
            print("Initial market data retrieved successfully")
        except Exception as e:
            print(f"Warning: Error during initial data update: {e}")
            logging.warning(f"Error during initial data update: {e}")

        # Trading mode selection
        print("\n" + "=" * 80)
        print("  EXECUTION MODE")
        print("=" * 80)

        print("\nExecution modes:")
        print("  1: Backtesting (Historical Analysis)")
        print("  2: Paper Trading (Simulated Live Trading)")
        print("  3: Live Trading (Real Market Orders)")
        print("  4: Continuous Learning (Live-Only Training with Paper Trading)")
        print("  5: Elite Autonomous Trading (Continuous Learning with Live Execution)")

        mode = input("\nSelect execution mode (default: 2): ") or "2"

        # RL Integration options
        if mode in ["1", "2", "3"]:  # Traditional modes
            # Get market data for these modes
            print("\nRetrieving market data for analysis...")
            market_data = data_feed.get_market_data(lookback=500)
            if len(market_data) < 30:
                print(f"Warning: Limited data available ({len(market_data)} points). Results may be unreliable.")
                logging.warning(f"Limited data available: {len(market_data)} points")

            # Option to add reinforcement learning
            use_rl = input("\nIntegrate reinforcement learning? (y/n): ").lower() == 'y'

            if use_rl:
                print("\n" + "=" * 80)
                print("  NEURAL REINFORCEMENT LEARNING INTEGRATION")
                print("=" * 80)

                # Ask for RL training preference
                train_new = input("Train new RL model or use existing? (train/use): ").lower() == 'train'

                if train_new:
                    print("\nTraining new reinforcement learning model...")
                    # Set training parameters
                    episodes = int(input("Enter number of training episodes (default: 50): ") or "50")
                    logging.info(f"Training new RL model with {episodes} episodes")

                    # Integrate RL with traditional strategy
                    market_data, rl_agent = integrate_rl_with_existing_strategy(market_data, episodes=episodes)
                else:
                    print("\nLoading existing reinforcement learning model...")
                    logging.info("Using existing RL model")

                    # Load the model
                    try:
                        # Initialize the RL agent
                        rl_agent = NQRLAgent()

                        # Check if production model exists
                        if os.path.exists(os.path.join(config.MODELS_DIR, "rl", "production_rl_agent.h5")):
                            print("Found production RL model, loading...")
                            rl_agent.load(os.path.join(config.MODELS_DIR, "rl", "production_rl_agent"))
                        else:
                            print("No existing model found, will create new one")

                        # Integrate with market data
                        market_data, rl_agent = integrate_rl_with_existing_strategy(market_data, mode='use')

                    except Exception as e:
                        print(f"Error loading RL model: {e}")
                        logging.error(f"Error loading RL model: {e}")
                        # Create a new one as fallback
                        market_data, rl_agent = integrate_rl_with_existing_strategy(market_data)

                # Advanced strategy selection
                print("\n" + "=" * 80)
                print("  STRATEGY SELECTION")
                print("=" * 80)

                # Get available strategy categories
                categories = strategy_factory.get_categories()

                print("\nAvailable strategy categories:")
                for i, category in enumerate(categories, 1):
                    strategy_count = len(strategy_factory.get_strategy_names(category))
                    print(f"  {i}: {category.replace('_', ' ').title()} ({strategy_count} strategies)")

                # Select category
                category_choice = input("\nSelect category number (default: 1): ") or "1"
                try:
                    category_index = int(category_choice) - 1
                    if category_index < 0 or category_index >= len(categories):
                        category_index = 0
                    selected_category = categories[category_index]
                except ValueError:
                    selected_category = categories[0]

                # Get strategies in selected category
                strategies = strategy_factory.get_strategy_names(selected_category)

                print(f"\nAvailable {selected_category.replace('_', ' ').title()} strategies:")
                for i, strategy in enumerate(strategies, 1):
                    print(f"  {i}: {strategy}")

                # Select strategy
                strategy_choice = input("\nSelect strategy number (default: 1): ") or "1"
                try:
                    strategy_index = int(strategy_choice) - 1
                    if strategy_index < 0 or strategy_index >= len(strategies):
                        strategy_index = 0
                    selected_strategy = strategies[strategy_index]
                except ValueError:
                    selected_strategy = strategies[0]

                # Create strategy instance
                strategy_instance = strategy_factory.create_strategy(selected_strategy)

                if strategy_instance:
                    print(f"\nSelected strategy: {strategy_instance.name}")
                    print(f"Description: {strategy_instance.description}")

                    # Generate signals
                    print("\nGenerating signals with selected strategy...")
                    market_data = strategy_instance.generate_signals(market_data)

                    active_signal = 'Signal'
                    strategy_name = strategy_instance.name
                else:
                    print("\nError creating strategy. Using default strategy.")
                    logging.error(f"Error creating strategy {selected_strategy}")
                    active_signal = 'Signal'
                    strategy_name = "Default"

                # Option to combine with RL if available
                if use_rl and 'RL_Signal' in market_data.columns:
                    combine_with_rl = input("\nCombine with Reinforcement Learning? (y/n): ").lower() == 'y'

                    if combine_with_rl:
                        # Create hybrid strategy
                        from nq_alpha_elite.strategies.hybrid_strategy import TechnicalRLHybridStrategy

                        print("\nCreating hybrid strategy with RL...")
                        hybrid_strategy = TechnicalRLHybridStrategy(
                            rl_agent=rl_agent if 'rl_agent' in locals() else None,
                            technical_weight=0.6,
                            rl_weight=0.4
                        )

                        # Generate signals
                        market_data = hybrid_strategy.generate_signals(market_data)

                        active_signal = 'Signal'
                        strategy_name = f"Hybrid ({strategy_name} + RL)"
            else:
                print("\nUsing traditional strategy only (no RL integration)")
                logging.info("RL integration skipped")

                # Create a simple default strategy
                from nq_alpha_elite.strategies.trend_following import MovingAverageCrossover

                default_strategy = MovingAverageCrossover()
                market_data = default_strategy.generate_signals(market_data)

                active_signal = 'Signal'
                strategy_name = default_strategy.name

        # Handle specific execution modes
        if mode == "1":
            # Backtesting mode
            print("\n" + "=" * 80)
            print("  BACKTESTING MODE")
            print("=" * 80)

            print(f"\nRunning backtest with {strategy_name} strategy...")
            logging.info(f"Backtesting {strategy_name} strategy")

            # Run backtest using your existing backtest function
            backtest_results = backtest_strategy(market_data, active_signal)

            # Plot backtest results
            print("\nGenerating performance charts...")
            plot_backtest_results(market_data, backtest_results, active_signal)

            # Save backtest results
            results_dir = os.path.join(config.DATA_DIR, "backtest_results")
            os.makedirs(results_dir, exist_ok=True)
            results_file = os.path.join(results_dir,
                                      f"backtest_{symbol}_{timeframe}_{strategy_name.replace(' ', '_')}_{current_time.replace(':', '-').replace(' ', '_')}.pkl")

            try:
                import pickle
                with open(results_file, 'wb') as f:
                    pickle.dump(backtest_results, f)
                print(f"\nBacktest results saved to: {results_file}")
                logging.info(f"Backtest results saved to: {results_file}")
            except Exception as e:
                print(f"Error saving backtest results: {str(e)}")
                logging.error(f"Error saving backtest results: {str(e)}")

        elif mode == "2":
            # Paper trading mode
            print("\n" + "=" * 80)
            print("  PAPER TRADING MODE")
            print("=" * 80)

            print(f"\nStarting paper trading with {strategy_name} strategy...")
            logging.info(f"Paper trading with {strategy_name} strategy")

            # Initialize paper trading parameters
            initial_balance = float(input(f"Enter initial balance (default: {config.TRADING_CONFIG['default_initial_balance']}): ") or
                                  str(config.TRADING_CONFIG['default_initial_balance']))
            position_size = float(input(f"Enter position size % (default: {config.TRADING_CONFIG['default_position_size'] * 100}): ") or
                                str(config.TRADING_CONFIG['default_position_size'] * 100)) / 100

            print("\nPaper trading initialized...")
            print(f"Initial Balance: ${initial_balance:.2f}")
            print(f"Position Size: {position_size * 100:.1f}%")

            # Run paper trading simulation
            try:
                paper_trading_results = run_paper_trading(
                    symbol, timeframe,
                    strategy=strategy_name,
                    signal_column=active_signal,
                    initial_balance=initial_balance,
                    position_size=position_size,
                    rl_agent=rl_agent if use_rl and 'rl_agent' in locals() else None
                )

                print("\nPaper trading completed")
                logging.info("Paper trading completed")

            except KeyboardInterrupt:
                print("\nPaper trading stopped by user")
                logging.info("Paper trading stopped by user")
            except Exception as e:
                print(f"\nError during paper trading: {str(e)}")
                logging.error(f"Paper trading error: {str(e)}")
                traceback.print_exc()

        elif mode == "3":
            # Live trading mode
            print("\n" + "=" * 80)
            print("  LIVE TRADING MODE")
            print("=" * 80)

            # Security confirmation
            print("\n⚠️ WARNING: You are about to start LIVE TRADING with REAL MONEY ⚠️")
            confirm = input("\nType 'CONFIRM' to proceed with live trading: ")

            if confirm != "CONFIRM":
                print("Live trading canceled")
                logging.info("Live trading canceled by user")
                return market_data

            print(f"\nInitializing live trading with {strategy_name} strategy...")
            logging.info(f"Live trading initialized with {strategy_name} strategy")

            # Live trading parameters
            trade_amount = input("Enter trade quantity (or press Enter to use position sizing): ")
            trade_amount = float(trade_amount) if trade_amount else None

            # Run live trading
            try:
                live_trading_results = run_live_trading(
                    symbol, timeframe,
                    strategy=strategy_name,
                    signal_column=active_signal,
                    quantity=trade_amount,
                    rl_agent=rl_agent if use_rl and 'rl_agent' in locals() else None
                )

                print("\nLive trading session completed")
                logging.info("Live trading session completed")

            except KeyboardInterrupt:
                print("\nLive trading stopped by user")
                logging.info("Live trading stopped by user")
            except Exception as e:
                print(f"\nError during live trading: {str(e)}")
                logging.error(f"Live trading error: {str(e)}")
                traceback.print_exc()

        elif mode == "4":
            # Continuous Learning with Paper Trading Mode
            print("\n" + "=" * 80)
            print("  CONTINUOUS LEARNING & PAPER TRADING MODE")
            print("  Training exclusively on live data while paper trading")
            print("=" * 80)

            # Initialize paper trading parameters
            initial_capital = float(input("\nEnter initial paper trading capital (default: 100000): ") or "100000")

            # Configure the live training system
            print("\nConfiguring continuous learning parameters...")
            min_data_points = int(input(f"Minimum data points before trading (default: {config.CONTINUOUS_LEARNING_CONFIG['min_data_points']}): ") or
                                str(config.CONTINUOUS_LEARNING_CONFIG['min_data_points']))
            update_interval = float(input(f"Update interval in seconds (default: {config.CONTINUOUS_LEARNING_CONFIG['update_interval']}): ") or
                                  str(config.CONTINUOUS_LEARNING_CONFIG['update_interval']))
            initial_trade_size = float(input(f"Initial position size % (default: {config.CONTINUOUS_LEARNING_CONFIG['initial_trade_size_pct'] * 100}): ") or
                                     str(config.CONTINUOUS_LEARNING_CONFIG['initial_trade_size_pct'] * 100)) / 100

            # Create Live Trainer configuration
            live_config = {
                'update_interval': update_interval,
                'min_data_points': min_data_points,
                'preferred_data_points': config.CONTINUOUS_LEARNING_CONFIG['preferred_data_points'],
                'initial_trade_size_pct': initial_trade_size,
                'max_trade_size_pct': initial_trade_size * 5,  # Max 5x the initial size
                'print_metrics_interval': config.CONTINUOUS_LEARNING_CONFIG['print_metrics_interval']
            }

            print("\n" + "=" * 80)
            print("  INITIALIZING CONTINUOUS LEARNING SYSTEM")
            print("=" * 80)

            try:
                # Initialize the live trainer
                print("\nCreating Live Trainer with zero historical data...")
                live_trainer = NQLiveTrainer(
                    initial_capital=initial_capital,
                    config=live_config
                )

                # Link the data feed
                live_trainer.market_data_feed = data_feed

                print("\nStarting continuous learning and paper trading...")
                logging.info("Starting continuous learning and paper trading")
                live_trainer.start()

                print("\nSystem is now collecting data and will begin trading after " +
                      f"collecting {min_data_points} data points.")
                print("\nPress Ctrl+C to stop the system.")

                try:
                    # Keep the main thread alive with periodic status updates
                    while True:
                        time.sleep(60)  # Check status every minute

                        # Print periodic status updates
                        metrics = {
                            'data_points': live_trainer.data_points_collected,
                            'capital': live_trainer.capital,
                            'return': (live_trainer.capital / initial_capital - 1) * 100,
                            'trades': len(live_trainer.trades_history),
                            'training_count': live_trainer.training_count
                        }

                        print("\n--- CONTINUOUS LEARNING STATUS UPDATE ---")
                        print(f"Data Points: {metrics['data_points']}")
                        print(f"Paper Capital: ${metrics['capital']:.2f} (Return: {metrics['return']:.2f}%)")
                        print(f"Completed Trades: {metrics['trades']}")
                        print(f"Training Iterations: {metrics['training_count']}")
                        if live_trainer.positions:
                            print(f"Active Positions: {len(live_trainer.positions)}")
                        print("----------------------------------------")

                except KeyboardInterrupt:
                    print("\nStopping continuous learning system...")
                    live_trainer.stop()
                    print("Continuous learning system stopped")
                    logging.info("Continuous learning system stopped by user")

                    # Save final model if needed
                    if hasattr(live_trainer, 'rl_agent'):
                        print("Saving final RL model...")
                        live_trainer.rl_agent.save(os.path.join(config.MODELS_DIR, "rl", "production_rl_agent"))
                        print("Model saved successfully")

            except Exception as e:
                print(f"\nError during continuous learning: {str(e)}")
                logging.error(f"Continuous learning error: {str(e)}")
                traceback.print_exc()

        elif mode == "5":
            # Elite Autonomous Trading (Continuous Learning with Live Execution)
            print("\n" + "=" * 80)
            print("  ELITE AUTONOMOUS TRADING MODE")
            print("  Continuous Learning with Live Order Execution")
            print("=" * 80)

            # Security confirmation for live trading
            print("\n⚠️ WARNING: You are about to start LIVE TRADING with REAL MONEY ⚠️")
            print("⚠️ This mode will execute real orders while learning from live data ⚠️")
            confirm = input("\nType 'CONFIRM LIVE TRADING' to proceed: ")

            if confirm != "CONFIRM LIVE TRADING":
                print("Elite autonomous trading canceled")
                logging.info("Elite autonomous trading canceled by user")
                return None

            print("\nThis feature will be enabled in the next major update.")
            print("Currently, please use Continuous Learning mode (Option 4) for training and")
            print("then switch to Live Trading mode (Option 3) with the trained model.")

            logging.info("Elite autonomous trading not yet implemented")

        else:
            print(f"\nInvalid mode selected: {mode}")
            logging.error(f"Invalid mode selected: {mode}")
            return None

        print("\n" + "=" * 80)
        print("  NQ ALPHA ELITE TRADING SYSTEM EXECUTION COMPLETED")
        print("=" * 80)

        # Return processed market data for further analysis if available
        if 'market_data' in locals():
            return market_data
        else:
            return None

    except Exception as e:
        print(f"\n===== CRITICAL ERROR =====")
        print(f"An error occurred in the main system: {str(e)}")
        print("\nDetailed traceback:")
        traceback.print_exc()
        logging.error(f"Critical error in main: {str(e)}", exc_info=True)
        return None


if __name__ == "__main__":
    try:
        print("\n" + "=" * 80)
        print(f"  NQ ALPHA ELITE TRADING SYSTEM - v{config.VERSION} '{config.VERSION_NAME}'")
        print("  World's Most Advanced AI Trading Platform with Autonomous Learning")
        print(f"  Developed by: {config.DEVELOPER}")
        print("=" * 80)

        # Initialize elite data accumulator first
        print("\nInitializing Elite Data Accumulation System...")
        data_accumulator = EliteDataAccumulator(max_points=10000)
        print("Elite Data Accumulator initialized successfully")

        # Initialize advanced market analysis components
        print("\nInitializing Elite Trading Components...")
        regime_detector = MarketRegimeDetector(lookback=100)
        a3c_agent = NQA3CAgent(state_size=20, action_size=3)
        print("Elite Trading Components initialized successfully")

        # Initialize data feed with proper error handling
        print("\nInitializing Market Data Feed...")
        try:
            # Try to use NQDirectFeed (web scraper class) first
            from nq_alpha_elite.data.web_scraper import NQDirectFeed
            data_feed = NQDirectFeed(clean_start=False)
            print("Using NQDirectFeed for market data...")
        except (ImportError, Exception) as e:
            # Fallback to MarketDataFeed
            try:
                data_feed = MarketDataFeed()
                print("Using MarketDataFeed for market data...")
            except Exception as e:
                print(f"Error initializing data feed: {e}")
                logging.error(f"Fatal error initializing data feed: {e}")
                raise

        # Link data accumulator to data feed
        data_feed.data_accumulator = data_accumulator

        # Initialize accelerated data collection subsystems
        print("\nActivating Elite Data Collection Subsystems...")
        try:
            # Turbo-charge data collection for ultra-fast training
            data_feed.turbo_data_collection(seconds_between_updates=0.2)
            print("Turbo Data Collection activated (5 points per second)")

            # Force periodic updates to ensure continuous data flow
            data_feed.force_update_frequency(minutes=1)
            print("Forced Update System activated (1-minute intervals)")

            # Enable data acceleration for synthetic point generation
            data_feed.enable_data_acceleration()
            print("Data Acceleration System activated")
        except Exception as e:
            print(f"Warning: Some data collection subsystems could not be activated: {e}")
            logging.warning(f"Data collection subsystem initialization error: {e}")

        # Force initial data fetch to seed the system
        print("\nPerforming initial data acquisition...")
        data_feed.update_data()

        # Now we're ready to execute the main trading system
        print("\n" + "=" * 80)
        print("  ELITE SYSTEM INITIALIZATION COMPLETE")
        print("  Starting Main Trading System...")
        print("=" * 80 + "\n")

        # Call the main function to start the trading system
        market_data = main()

    except KeyboardInterrupt:
        print("\n\nSystem shutdown initiated by user...")
        print("NQ Alpha Elite Trading System shutdown complete")

    except Exception as e:
        print(f"\n===== CRITICAL SYSTEM ERROR =====")
        print(f"Fatal error in system initialization: {str(e)}")
        print("\nDetailed error information:")
        traceback.print_exc()
        print("\nSystem shutdown due to critical error")

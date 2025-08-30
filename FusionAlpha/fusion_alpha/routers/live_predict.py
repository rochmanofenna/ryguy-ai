#!/usr/bin/env python3
import argparse
import ast
import logging
import numpy as np
# import pandas as pd  # Removed for live trading performance
import torch

# Import logger setup
try:
    from ..utils.logger_setup import setup_router_logging
    setup_router_logging()
except ImportError:
    logging.basicConfig(level=logging.INFO)
import yfinance as yf
import ta  # Technical analysis library (install via pip install ta)
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import feedparser

# Import your TradingModel from trading_model.py (which encapsulates EncoderTechnical, EncoderSentiment, etc.)
from trading_model import TradingModel  # Ensure this file is available in your project

# Load FinBERT tokenizer and model once.
finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert_model = AutoModel.from_pretrained("yiyanghkust/finbert-tone")
finbert_model.eval()

def get_finbert_cls_embedding(text: str) -> np.ndarray:
    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = finbert_model(**inputs)
    # Return the CLS token embedding (first token) as a tensor (keep on GPU for speed).
    # TODO: Refactor to avoid .cpu().numpy() in live path for better latency
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().detach()
    return cls_embedding  # Return tensor instead of numpy array

def get_live_headlines(ticker: str) -> list:
    # Attempt to fetch live news headlines from Yahoo Finance RSS feed
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    feed = feedparser.parse(url)
    headlines = [entry["title"] for entry in feed.entries[:5]]
    
    if not headlines:
        # Fallback headlines if feed is empty
        headlines = [
            f"{ticker} reports record quarterly earnings amid market turbulence",
            f"Analysts see growth potential in {ticker} despite global uncertainties"
        ]
    return headlines

def compute_simple_rsi(prices: np.ndarray, window: int = 14) -> float:
    """Compute simple RSI for live trading (last value only)."""
    if len(prices) < window + 1:
        return 50.0  # Neutral RSI if insufficient data
    
    deltas = np.diff(prices[-window-1:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_simple_ema(prices: np.ndarray, window: int = 20) -> float:
    """Compute simple EMA for live trading (last value only)."""
    if len(prices) < window:
        return np.mean(prices)  # Simple average if insufficient data
    
    alpha = 2.0 / (window + 1)
    ema = prices[0]
    for price in prices[1:]:
        ema = alpha * price + (1 - alpha) * ema
    return ema

def compute_simple_macd(prices: np.ndarray, fast: int = 12, slow: int = 26) -> float:
    """Compute simple MACD for live trading (last value only)."""
    if len(prices) < slow:
        return 0.0  # Neutral MACD if insufficient data
    
    ema_fast = compute_simple_ema(prices, fast)
    ema_slow = compute_simple_ema(prices, slow)
    return ema_fast - ema_slow

def get_features(ticker: str) -> (np.ndarray, np.ndarray, float):
    # Fetch live OHLCV data for the last 60 days.
    data = yf.download(ticker, period="60d", interval="1d")
    if data.empty:
        raise ValueError(f"No data returned for ticker {ticker}.")
    
    # Extract price arrays for technical indicators
    close_prices = data['Close'].values
    open_prices = data['Open'].values
    high_prices = data['High'].values
    low_prices = data['Low'].values
    volumes = data['Volume'].values
    
    # Compute technical indicators using numpy-only functions
    rsi = compute_simple_rsi(close_prices)
    ema = compute_simple_ema(close_prices)
    macd = compute_simple_macd(close_prices)
    
    # Use the latest values to compute technical features
    tech_features = np.array([
        open_prices[-1], high_prices[-1], low_prices[-1], close_prices[-1],
        volumes[-1], rsi, ema, macd
    ], dtype=np.float32).flatten()
    
    # If your model expects a fixed size (e.g., 10 features), pad with zeros.
    if tech_features.shape[0] < 10:
        tech_features = np.concatenate([tech_features, np.zeros(10 - tech_features.shape[0], dtype=np.float32)])
    
    # Get current price from the latest close.
    current_price = close_prices[-1]
    return tech_features, data, current_price

def main():
    parser = argparse.ArgumentParser(description="Live prediction for contradiction-aware trading model.")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker symbol (e.g., AAPL)")
    parser.add_argument("--current_price", type=float, default=None, help="Current asset price (if not provided, uses latest close)")
    parser.add_argument("--target_mode", type=str, default="normalized", choices=["normalized", "binary", "rolling"], help="Target mode used in training")
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load live OHLCV data and compute technical features.
    tech_features, ohlcv, current_price = get_features(args.ticker)
    if args.current_price is not None:
        current_price = args.current_price
    logger.info(f"Current price for {args.ticker}: {current_price}")
    
    # Pull live news headlines.
    headlines = get_live_headlines(args.ticker)
    logger.info(f"Live headlines: {headlines}")
    
    # Get FinBERT embedding by concatenating headlines.
    combined_text = " ".join(headlines)
    finbert_embedding = get_finbert_cls_embedding(combined_text)
    logger.debug(f"FinBERT embedding (first 5 dims): {finbert_embedding[:5]}")
    
    # Normalize features if needed.
    # (Assume any necessary normalization/scaling is handled within your TradingModel or preprocessor.)
    tech_tensor = torch.tensor(tech_features, dtype=torch.float32).to(device)
    finbert_tensor = torch.tensor(finbert_embedding, dtype=torch.float32).to(device)
    
    # Load pretrained TradingModel.
    model_args = {
        "tech_input_dim": 10,
        "sentiment_input_dim": 768,
        "encoder_hidden_dim": 64,
        "proj_dim": 32,
        "decision_hidden_dim": 64
    }
    model = TradingModel(**model_args).to(device)
    model.load_state_dict(torch.load("models/trading_model.pth", map_location=device))
    model.eval()
    
    # Use torch.no_grad for inference.
    with torch.no_grad():
        # Here we assume the model accepts two inputs: technical and sentiment embeddings.
        # If your architecture requires additional processing, modify accordingly.
        decision, contradiction_score, *_ = model(tech_tensor.unsqueeze(0), finbert_tensor.unsqueeze(0))
        prediction = decision.view(-1).item()
    
    logger.info(f"Predicted next-day return: {prediction}")
    
    # Map predicted return to a projected price change.
    projected_price = current_price * (1 + prediction)
    logger.info(f"Projected price based on predicted return: {projected_price:.2f}")
    
if __name__ == "__main__":
    main()
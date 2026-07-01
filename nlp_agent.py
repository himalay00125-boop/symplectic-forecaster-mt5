import time
import threading
import requests
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from transformers import pipeline

_GLOBAL_FINBERT = None

class FundamentalAgent:
    """
    An autonomous agent that reads financial news headlines and computes
    a live sentiment score using FinBERT.
    Runs asynchronously so it doesn't block the MT5 event loop.
    """
    def __init__(self, update_interval: float = 60.0 * 15):
        self.update_interval = update_interval
        global _GLOBAL_FINBERT
        if _GLOBAL_FINBERT is None:
            print("[NLP] Loading FinBERT model... (this may take a moment on first run)")
            try:
                _GLOBAL_FINBERT = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            except Exception as e:
                print(f"[NLP ERROR] Could not load FinBERT: {e}")
        self.nlp = _GLOBAL_FINBERT

        self._sentiment_cache = {}
        self._last_update = {}
        self._lock = threading.Lock()
        
        # We start a daemon thread to periodically update sentiment
        self._running = False
        self._thread = None

    def start(self, symbols: list):
        self._running = True
        self._thread = threading.Thread(target=self._update_loop, args=(symbols,), daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _update_loop(self, symbols: list):
        while self._running:
            for sym in symbols:
                # Query API using realtime-newsapi structure
                score = self._fetch_and_analyze(sym)
                
                with self._lock:
                    self._sentiment_cache[sym] = score
                    self._last_update[sym] = time.time()
                
                time.sleep(2.0) # Rate limit protection
            
            time.sleep(self.update_interval)

    def _fetch_and_analyze(self, symbol: str) -> float:
        if not self.nlp:
            return 0.0
            
        try:
            # Clean symbol (e.g., remove =X if it has it)
            clean_sym = symbol.replace("=X", "")
            
            # Request format from realtime-newsapi / newsfilter.io
            url = "https://api.newsfilter.io/public/actions"
            payload = {
                "type": "filterArticles",
                "queryString": f"symbols:{clean_sym} OR title:\"{clean_sym}\""
            }
            
            resp = requests.post(url, json=payload, timeout=5)
            if resp.status_code != 200:
                return 0.0
                
            news = resp.json()
            if not news or not isinstance(news, list):
                return 0.0
                
            headlines = [n.get('title', '') for n in news[:5] if n.get('title')]
            if not headlines:
                return 0.0
                
            results = self.nlp(headlines)
            
            # FinBERT returns labels: 'positive', 'negative', 'neutral'
            total_score = 0.0
            for res in results:
                label = res['label']
                score = res['score']
                if label == 'positive':
                    total_score += score
                elif label == 'negative':
                    total_score -= score
                    
            # Average score bounded [-1, 1]
            return total_score / len(results)
            
        except Exception as e:
            print(f"[NLP ERROR] Error analyzing {symbol}: {e}")
            return 0.0

    def get_sentiment(self, mt5_symbol: str) -> float:
        """Returns a cached sentiment score in [-1.0, 1.0]"""
        with self._lock:
            return self._sentiment_cache.get(mt5_symbol, 0.0)

if __name__ == "__main__":
    # Test execution
    agent = FundamentalAgent()
    print("Testing EURUSD sentiment...")
    score = agent._fetch_and_analyze("EURUSD")
    print(f"Sentiment Score: {score}")

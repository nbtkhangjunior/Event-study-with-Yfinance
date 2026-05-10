import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
import datetime
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# CONFIGURATION
# ==========================================
TICKER            = "META"          
EVENT_DATE        = "2022-02-02"    
EST_WINDOW        = 120             
ALPHA             = 0.05            

# Analysis Windows (Relative to t=0)
PRE_WINDOW        = (-5, -1)        
POST_WINDOW       = (2, 5)          
# ==========================================

class EventStudy:
    def __init__(self):
        self.event_date = pd.to_datetime(EVENT_DATE).tz_localize(None)
        self.data = None
        self.event_t0_idx = None
        self.est_residuals = None
        self.baseline_mean = None
        self.sigma_est = None
        self.is_normal = None

    def fetch_data(self):
        print(f"[*] Fetching data for {TICKER}...")
        try:
            from curl_cffi import requests as cffi_requests
            session = cffi_requests.Session(impersonate="chrome", verify=False)
        except ImportError:
            raise ImportError("Please install curl_cffi to continue: pip install curl_cffi")
        
        start_f = (self.event_date - datetime.timedelta(days=EST_WINDOW * 3)).strftime('%Y-%m-%d')
        end_f = (self.event_date + datetime.timedelta(days=30)).strftime('%Y-%m-%d')
        
        ticker_obj = yf.Ticker(TICKER, session=session)
        raw = ticker_obj.history(start=start_f, end=end_f)
        
        if raw.empty: 
            raise ValueError("No data returned from Yahoo Finance.")
        
        df = pd.DataFrame(raw['Close'].dropna())
        df.index = df.index.tz_convert(None) if df.index.tz else df.index.tz_localize(None)
        df['Return'] = df['Close'].pct_change()
        self.data = df.dropna()

        future_days = self.data.index[self.data.index >= self.event_date]

        if len(future_days) == 0:
            raise ValueError("Event date out of bounds.")
            
        t0_date = future_days[0]
        self.event_t0_idx = self.data.index.get_loc(t0_date)
        
        gap = abs(PRE_WINDOW[0]) + 1
        end_est = self.event_t0_idx - gap
        start_est = end_est - EST_WINDOW
        
        est_data = self.data['Return'].iloc[start_est:end_est]
        self.baseline_mean = est_data.mean()
        self.est_residuals = est_data - self.baseline_mean
        self.sigma_est = self.est_residuals.std()
        
        print(f"[*] Timeline locked. t=0 is {t0_date.date()}")

    def test_normality(self):
        _, p_value = stats.shapiro(self.est_residuals)
        self.is_normal = p_value > 0.05
        print("\n--- Diagnostic: Normality Test (For t=0 and t=1) ---")
        if self.is_normal:
            print(f"🟢 Data is Normal (p={p_value:.4f}).")
        else:
            print(f"🔴 Fat Tails Detected (p={p_value:.4f}).")

    def calculation_ttest(self, start_t, end_t):
        s_idx = self.event_t0_idx + start_t
        e_idx = self.event_t0_idx + end_t
        
        window_returns = self.data['Return'].iloc[s_idx:e_idx+1]
        abnormal_returns = window_returns - self.baseline_mean
        car = abnormal_returns.sum()
        
        N = len(window_returns)
        std_error_car = self.sigma_est * np.sqrt(N)
        
        t_stat = car / std_error_car if std_error_car != 0 else 0
        df = EST_WINDOW - 1
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        return {"CAR": car, "Stat": t_stat, "P-Value": p_value, "Sig": "***" if p_value < ALPHA else ""}

    def calculation_single_day(self, t_offset):
        idx = self.event_t0_idx + t_offset
        ret = self.data['Return'].iloc[idx]
        abnormal_return = ret - self.baseline_mean 
        
        if self.is_normal:
            z_stat = abnormal_return / self.sigma_est if self.sigma_est != 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            stat_val = z_stat
        else:
            percentile = stats.percentileofscore(self.est_residuals, abnormal_return)
            p_value = 2 * (1 - (percentile / 100)) if percentile > 50 else 2 * (percentile / 100)
            stat_val = percentile
            
        return {"CAR": abnormal_return, "Stat": stat_val, "P-Value": p_value, "Sig": "***" if p_value < ALPHA else ""}

    def _print_row(self, name, res):
        print(f"{name:<35} | {res['CAR']*100:>7.2f}% | {res['P-Value']:>7.4f} | {res['Stat']:>7.3f} | {res['Sig']}")

    def check_pre_event(self):
        res = self.calculation_ttest(PRE_WINDOW[0], PRE_WINDOW[1])
        self._print_row("Pre-Event Leakage (-5 to -1)", res)

    def check_t0_t1(self):
        res_t0 = self.calculation_single_day(0)
        res_t1 = self.calculation_single_day(1)
        self._print_row("Event Day (t=0)", res_t0)
        self._print_row("Next Day (t=1)", res_t1)

    def check_post_event(self):
        res = self.calculation_ttest(POST_WINDOW[0], POST_WINDOW[1])
        self._print_row("Post-Event Drift (+2 to +5)", res)

    def check_comprehensive_impact(self):
        res = self.calculation_ttest(0, POST_WINDOW[1])
        self._print_row(f"Comprehensive Impact (0 to +{POST_WINDOW[1]})", res)

    def run(self):
        print(f"\n{'='*75}")
        print(f"EVENT STUDY ANALYSIS: {TICKER} on {EVENT_DATE}")
        print(f"{'='*75}")
        
        self.fetch_data()
        self.test_normality()
        
        print("\n" + "="*75)
        print(f"{'PHASE':<35} | {'CAR (%)':>8} | {'P-VALUE':>7} | {'T-VALUE':>7} | SIG")
        print("="*75)
        
        self.check_pre_event()
        self.check_t0_t1()
        self.check_post_event()
        self.check_comprehensive_impact()
        
        print("="*75 + "\n")

if __name__ == "__main__":
    study = EventStudy()
    study.run()
# Event-study-with-Yfinance
Event study step:
1. Fetching data from yfinance, 3 times the determined number of days to avoid weekend, holidays
2. Normality check using Shapiro Test for Step 4
3. Calculate CAR for Pre-event day (for Information Leakage Theory) and Post-event day
4. Check for t0 (Event date) and t1 (in case the event happens while the market is closed)
5. Check for Pre-event day and Post-event day using t-test

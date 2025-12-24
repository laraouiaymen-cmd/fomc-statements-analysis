"""Clean FOMC statements and prepare CSV for ML model training. Also keep data/processed updated for new fomc releases."""

from pathlib import Path
import pandas as pd
import re
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np


def clean_statement(text):
    """
    Remove boilerplate and administrative content to prepare text for NLP models.
    """    
    # Remove "For immediate release" headers that can appear at the top.
    text = re.sub(r'For immediate release.*?\n', '', text, flags=re.IGNORECASE)
    
    # Normalize all whitespace (spaces, tabs, newlines) to a single space.
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def get_sp500_returns(fomc_dates):
    """
    Fetch S&P 500 data and calculate next-day returns for FOMC dates.
    This version is more robust to non-trading days.
    """
    print("📈 Fetching S&P 500 data...")
    
    # Get the date range
    dates = [datetime.strptime(d, '%Y%m%d') for d in fomc_dates]
    start_date = min(dates) - timedelta(days=7)  # Increased buffer
    end_date = max(dates) + timedelta(days=7)
    
    # Download S&P 500 data for the entire range
    sp500 = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
    
    if sp500.empty:
        print("❌ CRITICAL: Failed to download any S&P 500 data. Check network or yfinance.")
        return {date_str: None for date_str in fomc_dates}

    # Newer versions of yfinance return MultiIndex columns like ('Close', '^GSPC')
    # We need to flatten this to just 'Close'
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = sp500.columns.get_level_values(0)
    
    # Normalize the sp500 index to timezone-naive datetime objects for direct comparison
    sp500.index = sp500.index.tz_localize(None).normalize()
    
    returns_dict = {}
    
    for date_str in fomc_dates:
        fomc_date = datetime.strptime(date_str, '%Y%m%d')
        
        try:
            # Find the closing price on the FOMC date or the last available day before it.
            temp_date = fomc_date
            while temp_date not in sp500.index and (fomc_date - temp_date).days < 5:
                temp_date -= timedelta(days=1)
            
            if temp_date not in sp500.index:
                print(f"⚠️ Could not find a valid trading day for FOMC date {date_str}")
                returns_dict[date_str] = None
                continue

            fomc_close = sp500.loc[temp_date, 'Close']

            # Find the next valid trading day's closing price
            next_day = fomc_date + timedelta(days=1)
            while next_day not in sp500.index and (next_day - fomc_date).days < 7:
                next_day += timedelta(days=1)

            if next_day not in sp500.index:
                print(f"⚠️ Could not find next trading day for {date_str}")
                returns_dict[date_str] = None
                continue

            next_close = sp500.loc[next_day, 'Close']
            
            # Calculate return as percentage
            next_day_return = ((next_close - fomc_close) / fomc_close) * 100
            returns_dict[date_str] = next_day_return
            

        except Exception as e:
            print(f"❌ An unexpected error occurred for date {date_str}: {e}")
            returns_dict[date_str] = None
    
    return returns_dict


def process_statements():
    """Process all FOMC statements and create ML-ready CSV."""
    # Use relative paths from the project root
    project_root = Path(__file__).resolve().parent.parent.parent
    interim_dir = project_root / "data" / "interim"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    statements_data = []
    fomc_dates = []
    
    # First pass: collect dates and clean text
    print("🧹 Cleaning statements...")
    for txt_file in sorted(interim_dir.glob("*.txt")):
        date_str = txt_file.stem  # YYYYMMDD format
        fomc_dates.append(date_str)
        
        try:
            # Read the statement
            with open(txt_file, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            
            # Clean the text
            cleaned_text = clean_statement(raw_text)
            
            statements_data.append({
                'statement_id': date_str,
                'date': date_str,
                'clean_text': cleaned_text
            })
            
            print(f"✅ Cleaned: {date_str}")
            
        except Exception as e:
            print(f"❌ Error processing {date_str}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(statements_data)
    
    # Get S&P 500 returns
    returns_dict = get_sp500_returns(fomc_dates)
    
    # Add returns and labels
    df['next_day_return'] = df['statement_id'].map(returns_dict)
    
    # Convert to numeric, coercing errors (like None) to NaN (Not a Number)
    df['next_day_return'] = pd.to_numeric(df['next_day_return'], errors='coerce')
    
    # Use np.select for conditional labeling, which is vectorized and robust
    conditions = [
        df['next_day_return'] > 0,
        df['next_day_return'] <= 0
    ]
    choices = [1, 0]
    df['label'] = np.select(conditions, choices, default=np.nan)
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Save to CSV
    output_file = processed_dir / "fomc_statements.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"✅ Processed {len(df)} statements")
    print(f"📊 Valid returns: {df['next_day_return'].notna().sum()}")
    print(f"📈 Positive returns: {(df['label'] == 1).sum()}")
    print(f"📉 Negative returns: {(df['label'] == 0).sum()}")
    print(f"📁 Saved to: {output_file}")
    print(f"{'='*60}")
    
    return df


if __name__ == "__main__":
    df = process_statements()
    print("\nDataset preview:")
    print(df[['statement_id', 'date', 'next_day_return', 'label']].head(10))
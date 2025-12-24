"""
Incrementally updates the local FOMC statement dataset.

This script:
1. Detects which FOMC statements already exist in data/interim/.
2. Scrapes the Federal Reserve website to retrieve all available statements.
3. Downloads only missing statements, saves them in data/raw/ as .html file.
4. Extracts the main text, storing it as .txt in data/interim/.

The process is idempotent: existing files are preserved, and only new or lost
statements are added.
"""


from pathlib import Path
import requests
from bs4 import BeautifulSoup
import re
from dateutil import parser
import textwrap
import ftfy

# --- Configuration ---
BASE_URL = "https://www.federalreserve.gov"
CALENDAR_URL = f"{BASE_URL}/monetarypolicy/fomccalendars.htm"

# Use relative paths from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# --- Helper Functions ---

def get_existing_statements(directory):
    """Scans the target directory and returns a set of existing statement IDs."""
    directory.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return {file.stem for file in directory.glob("*.txt")}

def fetch_page(url):
    """Fetches the content of a URL."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"❌ Error fetching {url}: {e}")
        return None

def extract_statement_text(html_content):
    """Extracts and cleans the core statement text from HTML."""
    if not html_content:
        return None
        
    soup = BeautifulSoup(html_content, 'html.parser')
    article_div = soup.find('div', id='article')
    
    if not article_div:
        return None
    
    paragraphs = article_div.find_all('p')
    statement_paragraphs = []
    for p in paragraphs:
        text = p.get_text(strip=True)
        text = ftfy.fix_text(text)
        
        # Filter out metadata and short, irrelevant paragraphs
        if (text and not text.startswith('For release at') and 
            'Last Update' not in text and len(text) > 50):
            wrapped_text = textwrap.fill(
                text, width=80, break_long_words=False, break_on_hyphens=False
            )
            statement_paragraphs.append(wrapped_text)
            
    return '\n\n'.join(statement_paragraphs) if statement_paragraphs else None

# --- Main Update Function ---

def update_fomc_statements():
    """
    Scrapes the Federal Reserve website and downloads/processes any missing
    FOMC statements, saving them to the interim directory.
    """
    print("🚀 Starting FOMC statement update process...")
    
    existing_ids = get_existing_statements(INTERIM_DIR)
    print(f"🔎 Found {len(existing_ids)} existing statements in {INTERIM_DIR}.")
    
    print(" scraping Federal Reserve website for all statements...")
    calendar_html = fetch_page(CALENDAR_URL)
    if not calendar_html:
        print("❌ Could not fetch FOMC calendar. Aborting.")
        return

    soup = BeautifulSoup(calendar_html, 'html.parser')
    panels = soup.find_all("div", {"class": "panel panel-default"})
    
    new_statements_processed = 0
    
    for panel in panels:
        year_text = panel.find("div", {"class": "panel-heading"}).text
        year = re.findall(r"\d+", year_text)[-1]

        for row in panel.select('div[class*="row fomc-meeting"]'):
            try:
                month_text = row.find("div", {"class": "fomc-meeting__month"}).text.split('/')[-1]
                date_text = re.findall(r"\d+", row.find("div", {"class": "fomc-meeting__date"}).text)[-1]
                meeting_timestamp = parser.parse(f"{year} {month_text} {date_text}")
                statement_id = meeting_timestamp.strftime("%Y%m%d")

                # If we already have this statement, skip it
                if statement_id in existing_ids:
                    continue

                # --- FIX STARTS HERE: More robust link finding ---
                # Find the container for the statement link by searching for its text.
                statement_container = row.find(
                    lambda tag: tag.name == 'div' and 'Statement' in tag.get_text()
                )

                if not statement_container:
                    continue

                # First, try to find a specific "HTML" link.
                html_link_tag = statement_container.find("a", string=re.compile(r"\s*HTML\s*"))
                
                # If no "HTML" link is found, fall back to the first available link,
                # which is common for older statements.
                if not html_link_tag:
                    html_link_tag = statement_container.find("a")

                # If still no link, skip this entry.
                if not html_link_tag or not html_link_tag.get("href"):
                    continue
                # --- FIX ENDS HERE ---

                statement_url = BASE_URL + html_link_tag.get("href")
                
                print(f"✨ New statement found: {statement_id}. Downloading...")
                
                # Download, process, and save the new statement
                html_content = fetch_page(statement_url)
                
                if html_content:
                    # --- FIX STARTS HERE: Save raw HTML for traceability ---
                    raw_html_path = RAW_DIR / f"{statement_id}.html"
                    with open(raw_html_path, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    # --- FIX ENDS HERE ---

                    clean_text = extract_statement_text(html_content)
                    if clean_text:
                        output_path = INTERIM_DIR / f"{statement_id}.txt"
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(clean_text)
                        print(f"✅ Successfully processed and saved to {output_path.name}")
                        new_statements_processed += 1
                    else:
                        print(f"⚠️ Could not extract text for {statement_id}.")
                else:
                    print(f"⚠️ Failed to download HTML for {statement_id}.")

            except Exception as e:
                # This will catch errors in parsing a specific row
                print(f"⚠️ Error processing a meeting row in year {year}: {e}")
                continue

    print("\n" + "="*60)
    if new_statements_processed > 0:
        print(f"🎉 Update complete. Added {new_statements_processed} new statements.")
    else:
        print("✅ All local statements are up-to-date. No new statements found.")
    print("="*60)


if __name__ == "__main__":
    update_fomc_statements()

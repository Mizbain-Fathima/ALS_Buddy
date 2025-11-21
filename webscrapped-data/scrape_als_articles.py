import requests
from bs4 import BeautifulSoup
import json
import os
import time  # For delays between requests

urls = [
    "https://www.ninds.nih.gov/health-information/disorders/amyotrophic-lateral-sclerosis-als",
    "https://www.cdc.gov/als/index.html",
    "https://medlineplus.gov/amyotrophiclateralsclerosis.html",
    "https://www.mda.org/disease/amyotrophic-lateral-sclerosis",
    "https://www.als.net/news/approved-drugs-for-als-progression/",
    "https://www.iamals.org/what-is-als/",
    "https://www.ninds.nih.gov/health-information/disorders/amyotrophic-lateral-sclerosis-als",
    "https://rarediseases.info.nih.gov/diseases/5786/amyotrophic-lateral-sclerosis",
    "https://lesturnerals.org/what-is-als/",
    "https://lesturnerals.org/support-services/national-als-registry/",
    "https://lesturnerals.org/research-patient-center/",
    "https://lesturnerals.org/als-participation-in-clinical-research/",
    "https://lesturnerals.org/als-breathing-guide/",
    "https://lesturnerals.org/als-nutrition-guide/",
    "https://lesturnerals.org/als-and-genetics/",
    "https://lesturnerals.org/als-genetic-counseling-and-testing-for-family-members/",
    "https://lesturnerals.org/als-communication/",
    "https://lesturnerals.org/als-mobility-guide/",
    "https://lesturnerals.org/als-home-modifications-guide/",
    "https://lesturnerals.org/als-activities-of-daily-living-guide/",
    "https://lesturnerals.org/caregiver/",
    "https://lesturnerals.org/caregiver-rights/",
    "https://lesturnerals.org/als-caregiver-self-care-guide/",
    "https://lesturnerals.org/als-relationships-sex-and-intimacy-guide/",
    "https://lesturnerals.org/als-children-guide/",
    "https://www.als-mnd.org/what-is-alsmnd/",
    "https://www.answerals.org/research/research-progress/",
    "https://www.answerals.org/research/research-approach/",
    "https://www.answerals.org/research/understanding-als/",
    "https://www.answerals.org/research/als-myths/",
    "https://www.mayoclinic.org/diseases-conditions/amyotrophic-lateral-sclerosis"
]

data = []

for url in urls:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Referer': 'https://www.google.com/',  # Pretend referral from google search
        }
        r = requests.get(url, headers=headers, timeout=100)  # Increased timeout to 100 seconds
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')

        # Apollo API
        # Expanded content selectors for more comprehensive scraping
        content_selectors = [
        'p',  # Standard paragraphs
        '.content p', '.article-body p', '.main-content p', '.entry-content p', '.post-content p',  # Class-based content
        'main p', 'article p', 'section p', 'div p', 'span p',  # Semantic and div-based
        '[data-testid*="content"] p', '[id*="content"] p',  # Dynamic/ID-based
        'body p',  # Fallback for entire body (use cautiously)
        ]
        
        # If no paragraphs, try other elements for more data (lists, definitions, etc.)
        fallback_selectors = [
        'li',  # List items (bullets/numbers, e.g., symptoms lists)
        'ul li', 'ol li',  # Unordered/ordered list items
        'dt', 'dd',  # Definition terms/descriptions (dictionaries/glossaries)
        'dl dt', 'dl dd',  # Full definition lists
        'details summary', 'details p',  # Accordions/expandable sections (dropboxes-like)
        'select option',  # Dropdown options (if text-based)
        'a',  # Links (pointers to related info, e.g., "Learn more" links)
        'h1', 'h2', 'h3', 'h4', 'h5', 'h6',  # Headings (for structure)
        'strong', 'em', 'b', 'i',  # Emphasized text
        'blockquote',  # Quotes
        'table td', 'table th',  # Table cells (for data tables)
        ]
        
        paragraphs = []
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                paragraphs.extend([elem.get_text(strip=True) for elem in elements])
                break  # Use the first matching selector
        
        # If no paragraphs, try other elements like lists and headings for more data
        if not paragraphs:
            for selector in fallback_selectors:
                elements = soup.select(selector)
                if elements:
                    paragraphs.extend([elem.get_text(strip=True) for elem in elements])
                    break
        
        content = ' '.join(paragraphs) if paragraphs else "No content extracted"
        char_count = len(content)
        print(f"Scraped {char_count} characters from {url}")
        if char_count < 500:
            print(f"Warning: Low content ({char_count} chars) - page may be minimal or blocked.")
        
    except requests.exceptions.RequestException as e:
        content = f"Error fetching URL: {str(e)}"
        print(f"Error for {url}: {e}")
    
    data.append({"url": url, "content": content})
    time.sleep(1)  # 1-second delay between requests to avoid rate limits

# Save data
with open('als_articles_expanded.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4)

print("Scraping complete. Check 'als_articles_expanded.json' for results.")

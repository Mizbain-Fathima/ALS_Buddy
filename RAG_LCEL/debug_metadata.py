import json
from pathlib import Path

DATA_PATH = Path("webscrapped-data/als_articles_expanded.json")

data = json.loads(DATA_PATH.read_text(encoding="utf-8"))

print("\nTotal items:", len(data))
print("\n--- First 5 metadata values ---")

for i, item in enumerate(data[:5]):
    print(f"\nItem {i}:")
    print("meta =", item.get("meta"))

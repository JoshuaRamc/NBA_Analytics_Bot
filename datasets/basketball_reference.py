import pandas as pd

# AUTHORED BY MARK RAMCHARAN

url = "https://www.basketball-reference.com/leagues/NBA_2025_per_game.html"

# Read with correct encoding
tables = pd.read_html(url, flavor="lxml")

df = tables[0]

# Remove duplicate headers inside the table
df = df[df["Rk"] != "Rk"].reset_index(drop=True)

# Save with UTF-8 encoding
df.to_csv("nba_2025_per_game.csv", index=False, encoding="utf-8")

print(df.head())

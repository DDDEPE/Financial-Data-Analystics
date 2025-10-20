from pathlib import Path

# ===== Settings =====
START_DATE = "2020-03-01"
END_DATE   = "2024-12-31"

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

COUNTRY_NAME_TO_ISO3 = {
    "Australia": "AUS", "France": "FRA", "Germany": "DEU",
    "Japan": "JPN", "United Kingdom": "GBR",
    "Brazil": "BRA", "China": "CHN", "India": "IND",
    "South Africa": "ZAF", "Turkey": "TUR",
}
TARGET_ISO3 = {"AUS","FRA","DEU","JPN","GBR","BRA","CHN","IND","ZAF","TUR"}
DEV_ISO3    = {"AUS","FRA","DEU","JPN","GBR"}

ISO2_TO_ISO3 = {
    "AU":"AUS","FR":"FRA","DE":"DEU","JP":"JPN","GB":"GBR",
    "BR":"BRA","CN":"CHN","IN":"IND","ZA":"ZAF","TR":"TUR"
}
ISO3_TO_ISO2 = {v:k for k,v in ISO2_TO_ISO3.items()}

import json

with open("ml/models/cv_results.json", "r", encoding="utf-8") as f:
    d = json.load(f)

for m in d["models"]:
    print("Model:", m)
    print("Keys:", list(d["models"][m].keys()))

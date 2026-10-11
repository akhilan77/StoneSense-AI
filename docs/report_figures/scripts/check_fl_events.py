import re

with open("backend/app/services/federated_coordinator.py", "r", encoding="utf-8") as f:
    text = f.read()

events = []
for line in text.splitlines():
    m = re.findall(r'["\']([A-Z][A-Z_]+)["\']', line)
    for match in m:
        if any(term in match for match in [match] for term in ["ROUND", "MODEL", "HOSPITAL", "CLIENT", "FEDAVG", "GATE"]):
            if match not in events and not match.startswith("IMAGE") and not match.startswith("PROJECT"):
                events.append(match)

print("FL Events found in backend/app/services/federated_coordinator.py:")
for e in events:
    print(f"  {e}")

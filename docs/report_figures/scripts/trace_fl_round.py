with open("backend/app/services/federated_coordinator.py", "r", encoding="utf-8") as f:
    text = f.read()

lines = text.splitlines()
start = False
events = []
for i, l in enumerate(lines):
    if "def _run_round_workflow(" in l:
        start = True
    if start and ("_emit_event(" in l or "_set_hospital_state(" in l):
        events.append((i+1, l.strip()))
    if start and "def get_live_round_status(" in l:
        break

print("Events in _run_round_workflow:")
for num, ev in events:
    print(f"{num}: {ev}")

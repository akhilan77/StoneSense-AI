# Bug Tracking & Resolution Log

Below is the verified QA bug list tracking interface issues and routing discrepancies found during integration phases.

---

| Bug ID | Component | Description | Severity | Resolution Status |
| --- | --- | --- | --- | --- |
| BUG-01 | API Routing | Endpoint paths prefix mismatches (Starlette 404 on backend tests) | Critical | ✅ Resolved (prefixed version `/api/v1` routes in `main.py`) |
| BUG-02 | Schemas | Missing Optionals in response models causing payload parse errors | Major | ✅ Resolved (updated `responses.py` Pydantic models) |
| BUG-03 | Frontend UI | Form submits targeting mock paths rather than live api clients | Major | ✅ Resolved (integrated `assess` and `predictImage` form data calls) |
| BUG-04 | ML Pip Warnings | XGBoost serialization pickle compatibility warning outputted on boot | Minor | ✅ Accepted (No impact on execution prediction scores) |

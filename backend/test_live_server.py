import urllib.request
import json
import io
from pathlib import Path

def create_multipart_form(field_name: str, file_name: str, file_bytes: bytes, mime_type: str, form_data: dict) -> tuple[bytes, str]:
    boundary = "----WebKitFormBoundaryStoneSenseTest2026"
    body = io.BytesIO()
    for k, v in form_data.items():
        body.write(f"--{boundary}\r\n".encode("utf-8"))
        body.write(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode("utf-8"))
        body.write(f"{v}\r\n".encode("utf-8"))
    
    body.write(f"--{boundary}\r\n".encode("utf-8"))
    body.write(f'Content-Disposition: form-data; name="{field_name}"; filename="{file_name}"\r\n'.encode("utf-8"))
    body.write(f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"))
    body.write(file_bytes)
    body.write(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    content_type = f"multipart/form-data; boundary={boundary}"
    return body.getvalue(), content_type

def test_live_backend():
    print("=== 1. TEST /docs ===")
    with urllib.request.urlopen("http://127.0.0.1:8000/docs") as resp:
        print("GET /docs status:", resp.status)

    print("\n=== 2. TEST /api/v1/health ===")
    with urllib.request.urlopen("http://127.0.0.1:8000/api/v1/health") as resp:
        print("GET /api/v1/health status:", resp.status, resp.read().decode("utf-8"))

    print("\n=== 3. TEST /api/v1/auth/login ===")
    login_payload = json.dumps({"email": "admin@stonesense.ai", "password": "StoneSenseAdmin!2026"}).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:8000/api/v1/auth/login", data=login_payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        login_data = json.loads(resp.read().decode("utf-8"))
        print("POST /api/v1/auth/login status:", resp.status, "| User:", login_data["user"]["email"], "| Role:", login_data["user"]["role"])
        token = login_data["access_token"]

    print("\n=== 4. TEST /api/v1/models/dl ===")
    req_dl = urllib.request.Request("http://127.0.0.1:8000/api/v1/models/dl", headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req_dl) as resp:
        dl_data = json.loads(resp.read().decode("utf-8"))
        print("GET /api/v1/models/dl status:", resp.status)
        for m in dl_data["models"]:
            print(f"   - {m['id']}: name='{m['name']}', status={m['status']}, is_ready={m['is_ready']}, accuracy={m.get('accuracy')}")

    print("\n=== 5. TEST /api/v1/predict/risk ===")
    risk_payload = json.dumps({
        "age": 45,
        "gender": "male",
        "bmi": 26.5,
        "blood_pressure": 120.0,
        "diabetes": False,
        "family_history": True,
        "water_intake": 2.0,
        "urine_ph": 5.8,
        "urine_specific_gravity": 1.021,
        "calcium": 4.2,
        "uric_acid": 5.5,
        "creatinine": 1.1,
        "osmolality": 640.0,
        "conductivity": 24.5,
        "urea": 380.0,
        "hospital_id": 1
    }).encode("utf-8")
    req_risk = urllib.request.Request("http://127.0.0.1:8000/api/v1/predict/risk", data=risk_payload, headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req_risk) as resp:
        risk_data = json.loads(resp.read().decode("utf-8"))
        print("POST /api/v1/predict/risk status:", resp.status, "| Risk:", risk_data.get("risk_level"), "| Prob:", risk_data.get("probability"))

    print("\n=== 6. TEST /api/v1/predict/image with all 4 DL models ===")
    sample_img_path = Path("dl/datasets/partitions/hospital_1/test/Cyst/Cyst- (1372).jpg")
    with open(sample_img_path, "rb") as f:
        img_bytes = f.read()

    for model_family in ["resnet18", "yolo26", "dinov3", "qknn"]:
        body, ct_header = create_multipart_form(
            field_name="image",
            file_name="test_ct.jpg",
            file_bytes=img_bytes,
            mime_type="image/jpeg",
            form_data={"model_family": model_family, "hospital_id": "1"}
        )
        req_img = urllib.request.Request(
            "http://127.0.0.1:8000/api/v1/predict/image",
            data=body,
            headers={"Content-Type": ct_header, "Authorization": f"Bearer {token}"}
        )
        with urllib.request.urlopen(req_img) as resp:
            pred_data = json.loads(resp.read().decode("utf-8"))
            print(f"   [{model_family}] status: {resp.status} | Pred: {pred_data.get('class_name')} | Conf: {pred_data.get('confidence')} | Expl: {pred_data.get('gradcam', {}).get('available')}")

    print("\n=== 7. TEST CORS Preflight from http://localhost:5173 ===")
    req_cors = urllib.request.Request("http://127.0.0.1:8000/api/v1/models/dl", headers={
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "Authorization"
    }, method="OPTIONS")
    with urllib.request.urlopen(req_cors) as resp:
        print("OPTIONS status:", resp.status, "| Access-Control-Allow-Origin:", resp.headers.get("Access-Control-Allow-Origin"))

if __name__ == "__main__":
    test_live_backend()

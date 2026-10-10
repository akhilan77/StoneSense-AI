import sqlite3
import json

con = sqlite3.connect("stonesense.db")
cur = con.cursor()

# Deactivate old test deployed model
cur.execute("UPDATE model_versions SET is_deployed = 0, status = 'archived' WHERE is_deployed = 1")

# Check if resnet18_grouped_audit_v1 exists
cur.execute("SELECT id FROM model_versions WHERE version_tag = 'resnet18_grouped_audit_v1'")
row = cur.fetchone()

gate_report = json.dumps({
    "audited": True,
    "data_source": "grouped_split_pHash_d2",
    "slice_accuracy": 0.8694,
    "slice_f1": 0.8251,
    "cluster_accuracy": 0.9648,
    "cluster_f1": 0.9547,
    "stone_recall": 0.6570,
    "tumor_recall": 1.0,
    "patient_level_verified": False,
    "disclaimer": "Research Prototype Only — Not for Clinical Diagnosis."
})

if row:
    cur.execute("""
        UPDATE model_versions 
        SET accuracy = 0.8694, f1_score = 0.8251, precision = 0.8319, recall = 0.8378,
            is_deployed = 0, status = 'pending_review', gate_report = ?
        WHERE version_tag = 'resnet18_grouped_audit_v1'
    """, (gate_report,))
else:
    cur.execute("""
        INSERT INTO model_versions (
            model_family, version_tag, accuracy, f1_score, precision, recall,
            is_deployed, status, gate_report, artifact_path, trained_at
        ) VALUES (
            'resnet18_ct', 'resnet18_grouped_audit_v1', 0.8694, 0.8251, 0.8319, 0.8378,
            0, 'pending_review', ?, 'dl/models/kidney_resnet18.pth', datetime('now')
        )
    """, (gate_report,))

con.commit()
print("stonesense.db model_versions updated successfully.")
con.close()

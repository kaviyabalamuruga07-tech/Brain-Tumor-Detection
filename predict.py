# ================================================================
#  NeuroScan V3.0 — predict.py with Medicine Report
# ================================================================
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

IMG_SIZE   = 150
MODEL_FILE = "model.h5"

MEDICINE_REPORT = {
    "tumor": {
        "immediate_actions": [
            "Consult a Neurologist or Neurosurgeon immediately",
            "Get admitted to a hospital for further evaluation",
            "Request a contrast-enhanced MRI for confirmation",
            "Blood tests: CBC, LFT, RFT, coagulation profile",
            "Biopsy may be required to determine tumor type"
        ],
        "diagnostic_tests": [
            "Contrast-Enhanced MRI Brain",
            "CT Scan of Brain with contrast",
            "PET Scan",
            "EEG (Electroencephalogram)",
            "Neurological examination",
            "Ophthalmology exam (vision check)"
        ],
        "medications": [
            {"name":"Dexamethasone (Decadron)","use":"Reduces brain swelling and edema around tumor","dose":"4-16 mg/day as prescribed","type":"Corticosteroid"},
            {"name":"Temozolomide (Temodar)","use":"Chemotherapy drug for brain tumors","dose":"75 mg/m2 daily during radiotherapy","type":"Chemotherapy"},
            {"name":"Levetiracetam (Keppra)","use":"Prevents seizures caused by brain tumors","dose":"500-1500 mg twice daily","type":"Anticonvulsant"},
            {"name":"Bevacizumab (Avastin)","use":"Blocks blood supply to tumor","dose":"10 mg/kg every 2 weeks IV","type":"Targeted Therapy"},
            {"name":"Mannitol","use":"Reduces intracranial pressure in emergency","dose":"0.25-1 g/kg IV hospital use only","type":"Osmotic Diuretic"},
            {"name":"Ondansetron (Zofran)","use":"Controls nausea from chemotherapy","dose":"8 mg twice daily as needed","type":"Antiemetic"}
        ],
        "treatments": [
            {"name":"Surgical Resection","desc":"Removal of tumor by neurosurgeon. Most effective for accessible tumors.","icon":"🔪"},
            {"name":"Radiation Therapy","desc":"High-energy X-rays target and destroy tumor cells. Usually 30 sessions.","icon":"☢️"},
            {"name":"Chemotherapy","desc":"Drug treatment to kill cancer cells. Often combined with radiation.","icon":"💊"},
            {"name":"Stereotactic Radiosurgery","desc":"Gamma Knife or CyberKnife — precise radiation without open surgery.","icon":"🎯"},
            {"name":"Immunotherapy","desc":"Boosts immune system to fight tumor. Newer treatment approach.","icon":"🛡️"}
        ],
        "lifestyle": [
            "Rest adequately and avoid physical strain",
            "Eat a balanced diet rich in antioxidants",
            "Avoid alcohol and smoking completely",
            "Reduce stress with meditation or yoga",
            "Keep family support system active",
            "Attend all follow-up appointments",
            "Never skip prescribed medications",
            "Avoid driving if seizures have occurred"
        ],
        "warning_signs": [
            "Sudden severe headache",
            "Vomiting without nausea",
            "Vision or speech problems",
            "Sudden confusion or memory loss",
            "Weakness on one side of body",
            "Seizures or convulsions"
        ]
    },
    "no_tumor": {
        "immediate_actions": [
            "No immediate medical action required",
            "Schedule routine annual MRI check-up",
            "Follow up with your doctor to discuss results",
            "Continue any existing prescribed medications",
            "Maintain healthy lifestyle habits"
        ],
        "medications": [
            {"name":"Vitamin D3","use":"Supports brain and nervous system health","dose":"1000-2000 IU daily","type":"Supplement"},
            {"name":"Omega-3 Fatty Acids","use":"Reduces inflammation, supports brain function","dose":"1000 mg daily with food","type":"Supplement"},
            {"name":"Magnesium Glycinate","use":"Reduces headaches and supports nerve function","dose":"200-400 mg daily at bedtime","type":"Mineral"}
        ],
        "lifestyle": [
            "Maintain regular health check-ups",
            "Exercise regularly at least 30 min per day",
            "Eat brain-healthy foods: berries, nuts, fish",
            "Stay hydrated — drink 8 glasses of water daily",
            "Get 7-8 hours of quality sleep every night",
            "Keep brain active with reading and puzzles",
            "Avoid excessive screen time",
            "Manage stress with yoga or meditation"
        ]
    }
}

def preprocess(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read: {image_path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def predict_tumor(image_path):
    model = load_model(MODEL_FILE)
    img   = preprocess(image_path)
    pred  = model.predict(img, verbose=0)[0]
    no_tumor_prob = float(pred[0]) * 100
    tumor_prob    = float(pred[1]) * 100
    cls           = int(np.argmax(pred))
    confidence    = float(np.max(pred)) * 100
    if cls == 1:
        result     = "Tumor Detected"
        label      = "tumor"
        risk_level = "High Risk" if confidence >= 90 else ("Medium Risk" if confidence >= 70 else "Low Risk")
    else:
        result     = "No Tumor Detected"
        label      = "no_tumor"
        risk_level = "Low Risk"
    return {
        "result":      result,
        "confidence":  round(confidence, 2),
        "label":       label,
        "tumor_prob":  round(tumor_prob, 2),
        "normal_prob": round(no_tumor_prob, 2),
        "risk_level":  risk_level,
        "medicine":    MEDICINE_REPORT[label]
    }
TARGETS = [
    "Pneumothorax",
    "Pleural Effusion",
    "Support Devices",
    "Consolidation",
    "Cardiomegaly",
    "Atelectasis",
    "Edema",
    "No Finding",
]

PROMPTS = {
    "Pneumothorax": {
        "pos": [
            "There is a pneumothorax.",
            "Findings are consistent with pneumothorax.",
        ],
        "neg": [
            "There is no pneumothorax.",
            "No evidence of pneumothorax.",
        ],
    },
    "Pleural Effusion": {
        "pos": [
            "There is a pleural effusion.",
            "Findings are consistent with pleural effusion.",
        ],
        "neg": [
            "There is no pleural effusion.",
            "No pleural effusion is seen.",
        ],
    },
    "Support Devices": {
        "pos": [
            "Support devices are present.",
            "Lines and tubes are present.",
        ],
        "neg": [
            "No support devices are present.",
            "No lines and tubes are present.",
        ],
    },
    "Consolidation": {
        "pos": [
            "There is consolidation.",
            "Findings are consistent with consolidation.",
        ],
        "neg": [
            "There is no focal consolidation.",
            "No focal consolidation is seen.",
        ],
    },
    "Cardiomegaly": {
        "pos": [
            "There is cardiomegaly.",
            "The cardiac silhouette is enlarged.",
        ],
        "neg": [
            "There is no cardiomegaly.",
            "Heart size is normal.",
        ],
    },
    "Atelectasis": {
        "pos": [
            "There is atelectasis.",
            "Findings are consistent with atelectasis.",
        ],
        "neg": [
            "There is no atelectasis.",
            "No atelectatic change is seen.",
        ],
    },
    "Edema": {
        "pos": [
            "There is pulmonary edema.",
            "Findings are consistent with pulmonary edema.",
        ],
        "neg": [
            "There is no pulmonary edema.",
            "No edema is seen.",
        ],
    },
    "No Finding": {
        "pos": [
            "No acute cardiopulmonary abnormality.",
            "No acute disease is seen on this chest radiograph.",
        ],
        "neg": [
            "There is an acute cardiopulmonary abnormality.",
            "This is an abnormal chest radiograph.",
        ],
    },
}
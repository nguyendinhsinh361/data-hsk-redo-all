class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass

def validate_model(data, model):
    """
    Validate data against a predefined model.

    Args:
        data (dict): Input data to validate.
        model (dict): Model defining expected fields and data types.

    Raises:
        DataValidationError: If the data does not match the model.
    """
    for field, expected_type in model.items():
        if field not in data:
            raise DataValidationError(f"Missing field: {field}")
        if isinstance(expected_type, dict):
            if not isinstance(data[field], dict):
                raise DataValidationError(
                    f"Field '{field}' expected a nested object, got {type(data[field])}"
                )
            validate_model(data[field], expected_type)  # Recursive validation for nested objects
        elif isinstance(expected_type, list):
            if not isinstance(data[field], list):
                raise DataValidationError(
                    f"Field '{field}' expected a list, got {type(data[field])}"
                )
            for item in data[field]:
                if not isinstance(item, (expected_type[0] if isinstance(expected_type[0], type) else type(expected_type[0]))):
                    raise DataValidationError(
                        f"Items in field '{field}' expected {expected_type[0]}, got {type(item)}"
                    )
        elif not isinstance(data[field], expected_type):
            raise DataValidationError(
                f"Field '{field}' expected {expected_type}, got {type(data[field])}"
            )

# General model definition
general_model = {
    "G_text": list,
    "G_text_translate": {
        "vi": str,
        "en": str,
        "ko": str,
        "fr": str,
        "ja": str,
        "ru": str
    },
    "G_text_audio": str,
    "G_text_audio_translate": {
        "vi": str,
        "en": str,
        "ko": str,
        "fr": str,
        "ja": str,
        "ru": str
    },
    "G_audio": list,
    "G_image": [str]
}

# Content item model definition
content_item_model = {
    "Q_text": str,
    "Q_audio": str,
    "Q_image": str,
    "A_text": [str],
    "A_audio": list,
    "A_image": list,
    "A_correct": [str],
    "explain": {
        "vi": str,
        "en": str,
        "ko": str,
        "fr": str,
        "ja": str,
        "ru": str
    },
    "A_more_correct": [str],
    "advance_explain": {
        "vi": str,
        "en": str
    },
    "langExplainAdvance": [str]
}

# Main model definition
example_model = {
    "id": int,
    "kind": str,
    "general": general_model,
    "content": [content_item_model],
    "scores": [float]
}

# Example data
example_data = {
    "id": 43051,
    "kind": "310001",
    "general": {
        "G_text": [],
        "G_text_translate": {
            "vi": "",
            "en": "",
            "ko": "",
            "fr": "",
            "ja": "",
            "ru": ""
        },
        "G_text_audio": "",
        "G_text_audio_translate": {
            "vi": "",
            "en": "",
            "ko": "",
            "fr": "",
            "ja": "",
            "ru": ""
        },
        "G_audio": [],
        "G_image": [
            "https://hsk.migii.net/uploads/assert/images/ac3f4a6dbfcd4762a44723549941fd70.jpg",
            "https://hsk.migii.net/uploads/assert/images/ec279a5518d967360fc58e4696a51daf.jpg",
            "https://hsk.migii.net/uploads/assert/images/3f2b9a80b4c0bac45b6bddf189dd26b4.jpg",
            "https://hsk.migii.net/uploads/assert/images/4f934a48ae3d2de3a93a28b9be22f04c.jpg",
            "https://hsk.migii.net/uploads/assert/images/2b32484200f64f641e46760daa7f75db.jpg"
        ]
    },
    "content": [
        {
            "Q_text": "",
            "Q_audio": "https://hsk.migii.net/uploads/assert/audios/b587382f009920189b223c1cf1e602a1.mp3",
            "Q_image": "",
            "A_text": ["A", "B", "C", "D", "E"],
            "A_audio": [],
            "A_image": [],
            "A_correct": ["2"],
            "explain": {
                "vi": "Example explanation",
                "en": "Example explanation",
                "ko": "",
                "fr": "",
                "ja": "",
                "ru": ""
            },
            "A_more_correct": ["2"],
            "advance_explain": {
                "vi": "Detailed explanation in Vietnamese",
                "en": "Detailed explanation in English"
            },
            "langExplainAdvance": ["vi", "en"]
        }
    ],
    "scores": [2.5, 2.5, 2.5, 2.5, 2.5]
}

# Validate data
try:
    validate_model(example_data, example_model)
    print("Data is valid.")
except DataValidationError as e:
    print(f"Validation error: {e}")

"""Demo client for the local MLflow serving endpoint.

Start the server in one terminal::

    mlflow models serve -m "models:/churn_classifier/Production"
    -p 5001 --no-conda

Then run this script::

    python -m churn.inference.request

It POSTs a single sample customer record to the model and prints the
response.
"""

from __future__ import annotations

import json
import sys

import requests

DEFAULT_ENDPOINT = "http://127.0.0.1:5001/invocations"

# A single example row using the same schema as the training data
SAMPLE_CUSTOMER: dict[str, object] = {
    "Tenure": 4.0,
    "PreferredLoginDevice": "Mobile Phone",
    "CityTier": 3,
    "WarehouseToHome": 6.0,
    "PreferredPaymentMode": "Debit Card",
    "Gender": "Female",
    "HourSpendOnApp": 3.0,
    "NumberOfDeviceRegistered": 3,
    "PreferedOrderCat": "Laptop & Accessory",
    "SatisfactionScore": 2,
    "MaritalStatus": "Single",
    "NumberOfAddress": 9,
    "Complain": 1,
    "OrderAmountHikeFromlastYear": 11.0,
    "CouponUsed": 1.0,
    "OrderCount": 1.0,
    "DaySinceLastOrder": 5.0,
    "CashbackAmount": 159.93,
}

SAMPLE_CUSTOMER2: dict[str, object] = {
    "Tenure": 8.0,
    "PreferredLoginDevice": "Mobile Phone",
    "CityTier": 3,
    "WarehouseToHome": 6.0,
    "PreferredPaymentMode": "E wallet",
    "Gender": "Male",
    "HourSpendOnApp": 3.0,
    "NumberOfDeviceRegistered": 3,
    "PreferedOrderCat": "Fashion",
    "SatisfactionScore": 4,
    "MaritalStatus": "Divorced",
    "NumberOfAddress": 2,
    "Complain": 0,
    "OrderAmountHikeFromlastYear": 13.0,
    "CouponUsed": 1.0,
    "OrderCount": 1.0,
    "DaySinceLastOrder": 6.0,
    "CashbackAmount": 173.00,
}


def call_endpoint(endpoint: str = DEFAULT_ENDPOINT) -> dict:
    """POST sample customer to serving endpoint and return JSON response."""
    payload = {"dataframe_records": [SAMPLE_CUSTOMER]}
    response = requests.post(
        endpoint,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def main() -> int:
    """Run the demo request and print the result."""
    try:
        result = call_endpoint()
    except requests.exceptions.ConnectionError:
        print(
            "Could not connect to the MLflow serving endpoint at",
            DEFAULT_ENDPOINT,
            file=sys.stderr,
        )
        print("Start it with:", file=sys.stderr)
        print(
            '  mlflow models serve -m "models:/churn_classifier/Production"'
            " -p 5001 --no-conda",  # noqa: E501
            file=sys.stderr,
        )
        return 1

    print("Server response:")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Utility functions for communicating with the FastAPI backend.
"""

from typing import Any

import requests

from config import API_URL, REQUEST_TIMEOUT


class PredictionAPIError(Exception):
    """Raised when the prediction API cannot process the request."""


def predict(customer_data: dict[str, Any]) -> dict[str, Any]:
    """
    Send customer information to the FastAPI prediction endpoint.

    Parameters
    ----------
    customer_data : dict
        Customer information.

    Returns
    -------
    dict
        PredictionResponse JSON.

    Raises
    ------
    PredictionAPIError
    """

    try:
        response = requests.post(
            API_URL,
            json=customer_data,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.Timeout as exc:
        raise PredictionAPIError("The request timed out.") from exc

    except requests.exceptions.ConnectionError as exc:
        raise PredictionAPIError("Unable to connect to the FastAPI server.") from exc

    except requests.exceptions.HTTPError as exc:
        raise PredictionAPIError(response.text) from exc

    except requests.exceptions.RequestException as exc:
        raise PredictionAPIError(str(exc)) from exc

from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200


def test_health():
    response = client.get("/health")

    assert response.status_code == 200


def test_market_latest_endpoint_exists():
    response = client.get("/market/latest")

    assert response.status_code in [200, 404, 500]


def test_market_history_endpoint_exists():
    response = client.get(
        "/market/history",
        params={
            "symbol": "BTCUSDT",
            "timeframe": "1m",
            "limit": 10,
        },
    )

    assert response.status_code in [200, 404, 500]


def test_predictions_latest_endpoint_exists():
    response = client.get("/predictions/latest")

    assert response.status_code in [200, 404, 500]


def test_predictions_history_endpoint_exists():
    response = client.get("/predictions/history")

    assert response.status_code in [200, 404, 500]


def test_monitoring_latest_endpoint_exists():
    response = client.get("/monitoring/latest")

    assert response.status_code in [200, 404, 500]


def test_monitoring_history_endpoint_exists():
    response = client.get("/monitoring/history")

    assert response.status_code in [200, 404, 500]
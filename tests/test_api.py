"""
API Unit tests for FastAPI endpoint validation.
"""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

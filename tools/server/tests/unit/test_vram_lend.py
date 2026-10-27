import pytest
import re
from utils import *

server = ServerPreset.tinyllama2()

TEST_API_KEY = "sk-this-is-the-secret-key"

# the answer form: lower case, a 4-digit domain and the function
PCI_ADDR_RE = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]$")


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def test_state_no_cap():
    global server
    server.start()
    res = server.make_request("GET", "/vram/state")
    assert res.status_code == 200
    assert set(res.body.keys()) == {"gpus", "leases"}
    gpus = res.body["gpus"]
    # every GPU the model uses is listed, with or without a cap (a build without a GPU backend lists none)
    assert len(gpus) >= 1
    for gpu in gpus:
        assert set(gpu.keys()) == {"gpu", "lendable", "lent"}
        assert PCI_ADDR_RE.match(gpu["gpu"]), gpu["gpu"]
        # without a cap no GPU lends
        assert gpu["lendable"] == 0
        assert gpu["lent"] == 0
    # every GPU is listed once
    assert len(set(gpu["gpu"] for gpu in gpus)) == len(gpus)
    assert res.body["leases"] == []


def test_state_api_key():
    global server
    server.api_key = TEST_API_KEY
    server.start()
    res = server.make_request("GET", "/vram/state")
    assert res.status_code == 401
    res = server.make_request("GET", "/vram/state", headers={
        "Authorization": f"Bearer {TEST_API_KEY}",
    })
    assert res.status_code == 200
    assert "gpus" in res.body
    assert "leases" in res.body

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


def first_gpu(server: ServerProcess) -> str:
    res = server.make_request("GET", "/vram/state")
    assert res.status_code == 200
    assert len(res.body["gpus"]) >= 1
    return res.body["gpus"][0]["gpu"]


# a PCI address in the answer form that no server has
NO_SUCH_GPU = "0000:ff:1f.7"


@pytest.mark.parametrize("body", [
    "not an object",                        # the body is not a JSON object
    {"gpu": "GPU"},                         # mib missing
    {"gpu": "GPU", "mib": True},            # mib a boolean
    {"gpu": "GPU", "mib": 1.5},             # mib not a whole number
    {"gpu": "GPU", "mib": "64"},            # mib a string
    {"gpu": "GPU", "mib": 0},               # mib 0
    {"gpu": "GPU", "mib": -64},             # mib negative
    {"gpu": "bus 1", "mib": 64},            # gpu not a PCI address
    {"mib": 64},                            # gpu missing
    {"gpu": NO_SUCH_GPU, "mib": 64},        # gpu not a GPU of this server
])
def test_lend_bad_request(body):
    global server
    server.start()
    gpu = first_gpu(server)
    if isinstance(body, dict) and body.get("gpu") == "GPU":
        body["gpu"] = gpu
    res = server.make_request("POST", "/vram/lend", data=body)
    assert res.status_code == 400
    assert isinstance(res.body["error"], str)
    assert res.body["error"] != ""


def test_lend_no_cap():
    global server
    server.start()
    gpu = first_gpu(server)
    res = server.make_request("POST", "/vram/lend", data={"gpu": gpu, "mib": 64})
    assert res.status_code == 501
    assert isinstance(res.body["error"], str)


def test_lend_non_lending_gpu(monkeypatch):
    # with a cap on every GPU the server lends, but this model keeps no expert cache on any device,
    # so each GPU can lend 0
    monkeypatch.setenv("LLAMA_ARG_VRAM_LEND_MAX_MIB", "64")
    global server
    server.start()
    gpu = first_gpu(server)
    # the upper-case CUDA form and the NVML form name the same GPU
    for name in [gpu, gpu.upper(), "0000" + gpu]:
        res = server.make_request("POST", "/vram/lend", data={"gpu": name, "mib": 64})
        assert res.status_code == 409
        assert res.body == {"max_mib": 0}
    # nothing was lent
    res = server.make_request("GET", "/vram/state")
    assert res.status_code == 200
    assert res.body["leases"] == []
    assert all(g["lent"] == 0 for g in res.body["gpus"])

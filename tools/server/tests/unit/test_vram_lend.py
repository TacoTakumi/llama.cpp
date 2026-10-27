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


@pytest.mark.parametrize("body", [
    [],                  # the body is not a JSON object
    {},                  # id missing
    {"id": "a b"},       # id not one word
    {"id": ""},          # id empty
    {"id": True},        # id a boolean
    {"id": None},        # id null
    {"id": [1]},         # id an array
])
def test_return_bad_request(body):
    global server
    server.start()
    res = server.make_request("POST", "/vram/return", data=body)
    assert res.status_code == 400
    assert isinstance(res.body["error"], str)
    assert res.body["error"] != ""


def test_return_no_cap():
    global server
    server.start()
    res = server.make_request("POST", "/vram/return", data={"id": "1"})
    assert res.status_code == 501
    assert isinstance(res.body["error"], str)


def test_return_unknown_id(monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_VRAM_LEND_MAX_MIB", "64")
    global server
    server.start()
    # nothing was lent, so no id names a lease; a JSON number is read like the string
    for lease_id in ["1", 1, "7"]:
        res = server.make_request("POST", "/vram/return", data={"id": lease_id})
        assert res.status_code == 404
        assert isinstance(res.body["error"], str)
    res = server.make_request("GET", "/vram/state")
    assert res.status_code == 200
    assert res.body["leases"] == []


def test_lend_wait_option(monkeypatch):
    # the option is accepted; on this model no GPU lends, so the wait is never reached
    monkeypatch.setenv("LLAMA_ARG_VRAM_LEND_MAX_MIB", "64")
    monkeypatch.setenv("LLAMA_ARG_VRAM_LEND_WAIT_MS", "1")
    global server
    server.start()
    gpu = first_gpu(server)
    res = server.make_request("POST", "/vram/lend", data={"gpu": gpu, "mib": 64})
    assert res.status_code == 409
    assert res.body == {"max_mib": 0}


# a small MoE model with its experts in the CPU keeps an expert cache on the GPUs, which can lend
def moe_lend_server(monkeypatch) -> ServerProcess:
    monkeypatch.setenv("LLAMA_ARG_CPU_MOE", "1")
    monkeypatch.setenv("LLAMA_ARG_MOE_CACHE_MIB", "64")
    monkeypatch.setenv("LLAMA_ARG_VRAM_LEND_MAX_MIB", "64")
    return ServerPreset.stories15m_moe()


def lending_gpu(server: ServerProcess) -> str:
    res = server.make_request("GET", "/vram/state")
    assert res.status_code == 200
    gpus = res.body["gpus"]
    if len(gpus) == 0:
        pytest.skip("no GPU in this build")  # ty: ignore[too-many-positional-arguments]
    best = max(gpus, key=lambda g: g["lendable"])
    assert best["lendable"] >= 1
    return best["gpu"]


def assert_nothing_lent(server: ServerProcess):
    res = server.make_request("GET", "/vram/state")
    assert res.status_code == 200
    assert res.body["leases"] == []
    assert all(g["lent"] == 0 for g in res.body["gpus"])


def test_lend_and_return_moe(monkeypatch):
    global server
    server = moe_lend_server(monkeypatch)
    server.start()
    gpu = lending_gpu(server)

    res = server.make_request("POST", "/vram/lend", data={"gpu": gpu, "mib": 1})
    assert res.status_code == 200
    lease = res.body
    assert set(lease.keys()) == {"id", "gpu", "mib"}
    assert lease["gpu"] == gpu
    assert lease["mib"] >= 1

    res = server.make_request("GET", "/vram/state")
    assert res.body["leases"] == [lease]
    lent = {g["gpu"]: g["lent"] for g in res.body["gpus"]}
    assert lent[gpu] == lease["mib"]

    # the server keeps serving while the memory is lent
    res = server.make_request("POST", "/completion", data={"prompt": "Once upon a time", "n_predict": 8, "temperature": 0})
    assert res.status_code == 200
    assert len(res.body["content"]) > 0

    # a second lease on the same GPU, when it still lends, is independent of the first
    res = server.make_request("GET", "/vram/state")
    lendable = {g["gpu"]: g["lendable"] for g in res.body["gpus"]}
    if lendable[gpu] >= 1:
        res = server.make_request("POST", "/vram/lend", data={"gpu": gpu, "mib": 1})
        assert res.status_code == 200
        second = res.body
        assert second["id"] != lease["id"]
        res = server.make_request("POST", "/vram/return", data={"id": second["id"]})
        assert res.status_code == 200
        assert res.body == second
        res = server.make_request("GET", "/vram/state")
        assert res.body["leases"] == [lease]

    # more than the GPU can lend now
    res = server.make_request("POST", "/vram/lend", data={"gpu": gpu, "mib": 1 << 30})
    assert res.status_code == 409
    assert "max_mib" in res.body

    # the return, as a JSON number id; a second return finds no lease
    res = server.make_request("POST", "/vram/return", data={"id": int(lease["id"])})
    assert res.status_code == 200
    assert res.body == lease
    res = server.make_request("POST", "/vram/return", data={"id": lease["id"]})
    assert res.status_code == 404
    assert_nothing_lent(server)


def test_lend_wait_timeout(monkeypatch):
    # with a 1 ms wait the request can lose the race against the main loop; either way the borrower
    # never keeps memory it was not told about: a lend answered 503 is returned by the server itself
    global server
    server = moe_lend_server(monkeypatch)
    monkeypatch.setenv("LLAMA_ARG_VRAM_LEND_WAIT_MS", "1")
    server.start()
    gpu = lending_gpu(server)

    seen = set()
    for _ in range(10):
        res = server.make_request("POST", "/vram/lend", data={"gpu": gpu, "mib": 1})
        assert res.status_code in (200, 503)
        seen.add(res.status_code)
        if res.status_code == 200:
            lease = res.body
            res = server.make_request("POST", "/vram/return", data={"id": lease["id"]})
            assert res.status_code == 200
        else:
            assert "did not complete within 1 ms" in res.body["error"]
            # the 503 can come before the lend is even recorded, and the self-return follows the lend within
            # milliseconds; the state is settled once it stays empty over a few polls
            deadline = time.time() + 10
            empty_polls = 0
            while time.time() < deadline and empty_polls < 3:
                res = server.make_request("GET", "/vram/state")
                empty_polls = empty_polls + 1 if res.body["leases"] == [] else 0
                time.sleep(0.05)
        assert_nothing_lent(server)
    print(f"lend answers seen with a 1 ms wait: {sorted(seen)}")

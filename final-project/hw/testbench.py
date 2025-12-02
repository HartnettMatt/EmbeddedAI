import cocotb
from pathlib import Path

import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ReadOnly, RisingEdge
from cocotb.result import TestFailure
from cocotb.triggers import Event

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "sw" / "dataset" / "digits.npz"
GOLDEN_PATH = REPO_ROOT / "sw" / "artifacts" / "golden_vectors.npz"
SEED = 42


async def reset_dut(dut, cycles: int = 2) -> None:
    dut.rst_n.value = 0
    dut.s_axis_tvalid.value = 0
    dut.s_axis_tdata.value = 0
    dut.s_axis_tid.value = 0
    dut.m_axis_tready.value = 0
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def s_axis_write(dut, tdata: int, tid: int) -> None:
    dut.s_axis_tdata.value = tdata
    dut.s_axis_tid.value = tid
    dut.s_axis_tvalid.value = 1
    while True:
        await RisingEdge(dut.clk)
        if dut.s_axis_tready.value:
            break
    dut.s_axis_tvalid.value = 0


def image_to_tdata(image: np.ndarray, data_width: int) -> int:
    """Convert 8x8 digit image to binarized bit-packed tdata."""
    img_f = image.astype(np.float32)
    if img_f.max() <= 1.0 and img_f.min() >= -1.0:
        normalized = img_f
    else:
        normalized = img_f / 16.0 * 2.0 - 1.0
    bits = (normalized >= 0).astype(np.uint8).flatten()
    if len(bits) > data_width:
        raise TestFailure(f"Image bits ({len(bits)}) exceed DATA_WIDTH ({data_width})")
    tdata = 0
    for idx, bit in enumerate(bits):
        tdata |= int(bit) << idx
    return tdata


def generate_tids(count: int, tid_width: int, seed: int = 123) -> np.ndarray:
    """Generate a sequence of TIDs with no consecutive duplicates."""
    rng = np.random.default_rng(seed)
    tid_space = 1 << tid_width
    tids = np.zeros((count,), dtype=np.int32)
    last = None
    for i in range(count):
        if tid_space < 2:
            tids[i] = 0
            continue
        while True:
            candidate = int(rng.integers(0, tid_space))
            if last is None or candidate != last:
                tids[i] = candidate
                last = candidate
                break
    return tids


@cocotb.test()
async def test_reset_condition(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)

    # Hold a few cycles after reset and check outputs are idle.
    for _ in range(3):
        await RisingEdge(dut.clk)
        assert dut.m_axis_tvalid.value == 0, "m_axis_tvalid should be low after reset"
        assert dut.m_axis_tdata.value == 0, "m_axis_tdata should reset to zero"
        assert dut.m_axis_tid.value == 0, "m_axis_tid should reset to zero"


@cocotb.test()
async def test_single_image_flow(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)

    if not DATASET_PATH.exists():
        raise TestFailure(f"Dataset not found at {DATASET_PATH}. Run sw/model.py to generate it.")

    golden_path = REPO_ROOT / "sw" / "artifacts" / "golden_vectors.npz"
    if golden_path.exists():
        golden = np.load(golden_path, allow_pickle=False)
        images = golden["images_8x8"]
        expected_labels = golden["predictions_hw"] if "predictions_hw" in golden else golden["predictions"]
    else:
        digits = np.load(DATASET_PATH, allow_pickle=False)
        images = digits["images"]
        expected_labels = digits["target"]
    first_image = images[0]
    expected_label = int(expected_labels[0])

    tdata = image_to_tdata(first_image, int(dut.DATA_WIDTH))

    dut.m_axis_tready.value = 1
    tid = 0x5

    await s_axis_write(dut, tdata=tdata, tid=tid)

    # Wait for result to appear.
    while not dut.m_axis_tvalid.value:
        await RisingEdge(dut.clk)

    await ReadOnly()
    predicted = dut.m_axis_tdata.value.integer
    assert dut.m_axis_tid.value.integer == tid, "Output TID should match input"
    assert dut.s_axis_tready.value == 1, "Input ready should return high after result latched"
    assert predicted == expected_label, f"Predicted class {predicted} != expected {expected_label}"

    # Consume result.
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_basic_backpressure(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)

    tid = 0x3
    tdata = 0x1234
    dut.m_axis_tready.value = 0

    await s_axis_write(dut, tdata=tdata, tid=tid)

    # Wait for DUT to present data.
    while not dut.m_axis_tvalid.value:
        await RisingEdge(dut.clk)

    # While backpressured, m_axis_tvalid should stay asserted and s_axis_tready should drop.
    for _ in range(3):
        await RisingEdge(dut.clk)
        assert dut.m_axis_tvalid.value == 1, "m_axis_tvalid should hold under backpressure"
        assert dut.s_axis_tready.value == 0, "s_axis_tready should deassert when holding output"

    # Release backpressure and ensure handshake completes.
    dut.m_axis_tready.value = 1
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert dut.m_axis_tvalid.value == 0, "m_axis_tvalid should clear after ready asserted"
    assert dut.s_axis_tready.value == 1, "s_axis_tready should return high after drain"


@cocotb.test()
async def test_batch_dataset_predictions(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)

    if not DATASET_PATH.exists():
        raise TestFailure(f"Dataset not found at {DATASET_PATH}. Run sw/model.py to generate it.")

    if GOLDEN_PATH.exists():
        golden = np.load(GOLDEN_PATH, allow_pickle=False)
        images = golden["images_8x8"]
        expected_labels = golden["predictions_hw"] if "predictions_hw" in golden else golden["predictions"]
    else:
        digits = np.load(DATASET_PATH, allow_pickle=False)
        images = digits["images"]
        expected_labels = digits["target"]

    dut.m_axis_tready.value = 1
    tids = generate_tids(len(images), int(dut.TID_WIDTH), seed=SEED)

    for idx in range(len(images)):
        image = images[idx]
        expected = int(expected_labels[idx])
        tdata = image_to_tdata(image, int(dut.DATA_WIDTH))

        tid = int(tids[idx])

        # Issue next image as soon as ready deasserts.
        await s_axis_write(dut, tdata=tdata, tid=tid)

        while not dut.m_axis_tvalid.value:
            await RisingEdge(dut.clk)

        await ReadOnly()
        predicted = dut.m_axis_tdata.value.integer
        assert dut.m_axis_tid.value.integer == tid, f"TID mismatch on sample {idx}"
        assert predicted == expected, f"Predicted class {predicted} != expected {expected} on sample {idx}"

        await RisingEdge(dut.clk)


@cocotb.test()
async def test_latency(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)

    if not GOLDEN_PATH.exists():
        raise TestFailure("Golden vectors missing; run sw/model.py")
    golden = np.load(GOLDEN_PATH, allow_pickle=False)
    image = golden["images_8x8"][0]
    expected = int((golden["predictions_hw"] if "predictions_hw" in golden else golden["predictions"])[0])
    tdata = image_to_tdata(image, int(dut.DATA_WIDTH))
    tid = 1

    dut.m_axis_tready.value = 1
    start_cycles = 0
    end_cycles = 0

    await s_axis_write(dut, tdata=tdata, tid=tid)
    while not dut.m_axis_tvalid.value:
        start_cycles += 1
        await RisingEdge(dut.clk)
    await ReadOnly()
    end_cycles = start_cycles
    predicted = dut.m_axis_tdata.value.integer
    dut._log.info("Latency cycles: %d, predicted=%d expected=%d", end_cycles, predicted, expected)


@cocotb.test()
async def test_throughput(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)

    if not GOLDEN_PATH.exists():
        raise TestFailure("Golden vectors missing; run sw/model.py")
    golden = np.load(GOLDEN_PATH, allow_pickle=False)
    images = golden["images_8x8"]
    expected_labels = golden["predictions_hw"] if "predictions_hw" in golden else golden["predictions"]

    count = min(20, len(images))
    tids = generate_tids(count, int(dut.TID_WIDTH), seed=SEED)
    dut.m_axis_tready.value = 1

    done_event = Event()
    pending = []

    async def feed_images():
        for idx in range(count):
            tdata = image_to_tdata(images[idx], int(dut.DATA_WIDTH))
            tid = int(tids[idx])
            pending.append((tid, int(expected_labels[idx])))
            await s_axis_write(dut, tdata=tdata, tid=tid)
        done_event.set()

    cocotb.start_soon(feed_images())

    received = 0
    cycles = 0
    busy_cycles = 0
    while received < count:
        await RisingEdge(dut.clk)
        cycles += 1
        if not dut.m_axis_tvalid.value:
            continue
        busy_cycles += 1
        await ReadOnly()
        predicted = dut.m_axis_tdata.value.integer
        tid = dut.m_axis_tid.value.integer
        assert pending, "Output without pending input"
        exp_tid, expected = pending.pop(0)
        assert tid == exp_tid, f"TID mismatch: expected {exp_tid} got {tid} at index {received}"
        assert predicted == expected, f"Predicted class {predicted} != expected {expected} on sample {received}"
        received += 1

    await done_event.wait()
    util = (busy_cycles / cycles) * 100.0 if cycles else 0.0
    ips = received / cycles if cycles else 0.0
    dut._log.info("Throughput: util=%.2f%%, images/cycle=%.4f", util, ips)

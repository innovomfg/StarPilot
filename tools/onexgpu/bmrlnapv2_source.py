#!/usr/bin/env python3
"""Source-contract validation only; no swap or service-control code."""
import base64
import hashlib
import io
import json
import os
import pickle

EXPECTED = json.loads(r'''{"model_checkpoint":"594c0fe7-649a-4e06-5239-f3cc64a67aac/56320/2004ef3b-0461-b0f9-a1c6-23f444b6ef71/12864","output_slices":{"lane_lines":{"slice":[0,528,null]},"lane_lines_prob":{"slice":[528,536,null]},"road_edges":{"slice":[536,800,null]},"meta":{"slice":[800,855,null]},"desire_pred":{"slice":[855,887,null]},"pose":{"slice":[887,899,null]},"wide_from_device_euler":{"slice":[899,905,null]},"road_transform":{"slice":[905,917,null]},"plan":{"slice":[917,1907,null]},"lead":{"slice":[1907,2051,null]},"lead_prob":{"slice":[2051,2054,null]},"desire_state":{"slice":[2054,2062,null]},"action":{"slice":[2062,2066,null]},"hidden_state":{"slice":[2066,18450,null]},"pad":{"slice":[18450,18452,null]}},"input_shapes":{"img":[1,12,128,256],"big_img":[1,12,128,256],"desire_pulse":[1,33,8],"traffic_convention":[1,2],"action_t":[1,2],"features_buffer":[1,32,32,512]},"output_shapes":{"outputs":[1,18452]}}''')
SOURCE_SHA = "6afb28d58bfa7d7639832db8a61ff9209ef704ce1a9de2cdc20dd9004ec42bec"
SOURCE_SIZE = 1757861459
def say(*args):
    try:
        print(*args, flush=True)
    except BrokenPipeError:
        pass

class SlicesOnly(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("builtins", "slice"):
            return slice
        raise ValueError("Unexpected metadata object: " + module + "." + name)

def normalize(value):
    if isinstance(value, slice):
        return {"slice": [value.start, value.stop, value.step]}
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    return value

def validate_source(stage):
    os.environ["DEV"] = "CPU:LLVM"
    from tinygrad.helpers import DEV
    DEV.value = "CPU:LLVM"
    source = stage / "big_driving_supercombo.onnx"
    assert source.stat().st_size == SOURCE_SIZE, "Source size mismatch or download incomplete"
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == SOURCE_SHA, "Source checksum mismatch"
    from openpilot.selfdrive.modeld.get_model_metadata import (
        MetadataOnnxPBParser, get_metadata_value_by_name, get_name_and_shape)
    model = MetadataOnnxPBParser(source).parse()
    slices = get_metadata_value_by_name(model, "output_slices")
    actual = {
        "model_checkpoint": get_metadata_value_by_name(model, "model_checkpoint"),
        "output_slices": SlicesOnly(io.BytesIO(base64.b64decode(slices))).load(),
        "input_shapes": dict(get_name_and_shape(v) for v in model["graph"]["input"]),
        "output_shapes": dict(get_name_and_shape(v) for v in model["graph"]["output"]),
    }
    assert normalize(actual) == EXPECTED, "Source contract differs from published BMRLNAP V2"
    from tinygrad.device import Device
    assert "AMD" not in Device._opened_devices, "Metadata process unexpectedly owns the GPU"
    say("SOURCE_CONTRACT_VERIFIED", actual["model_checkpoint"])
    say("SOURCE_VALIDATION_CPU_ONLY")
    return source

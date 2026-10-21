"""Verify integration, gradient flow, pretrained transfer and checkpoint fidelity."""
import copy
import io
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
(ROOT / ".ultralytics").mkdir(exist_ok=True)

import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from attention import MODULES, TRAINERS, attach_attention


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    source = YOLO(str(ROOT / "yolov8n.pt")).model
    base = DetectionModel(copy.deepcopy(source.yaml), nc=2, verbose=False)
    base.load(source, verbose=False)
    base_state = {k: v.clone() for k, v in base.state_dict().items()}
    report = {}
    for name, module_type in MODULES.items():
        module = module_type(128).eval()
        for p in module.parameters():
            p.data.zero_()
        sample = torch.randn(2, 128, 7, 11)
        expected_scale = 0.5 if name == "se" else 0.25
        torch.testing.assert_close(module(sample), sample * expected_scale)

        model = attach_attention(copy.deepcopy(base), name)
        for key, tensor in base_state.items():
            torch.testing.assert_close(model.state_dict()[key], tensor, rtol=0, atol=0)
        new_keys = set(model.state_dict()) - set(base_state)
        assert new_keys and all(k.startswith("model.6.attention.") for k in new_keys)
        # Non-square inputs exercise Coordinate Attention's independent axes.
        model.train()
        outputs = model(torch.randn(2, 3, 96, 128))
        loss = sum(y.square().mean() for y in outputs)
        loss.backward()
        params = list(model.model[6].attention.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        assert any(p.grad.abs().sum() > 0 for p in params)
        model.eval()
        x = torch.randn(1, 3, 96, 128)
        with torch.no_grad():
            expected = model(x)[0]
            memory = io.BytesIO()
            torch.save(model, memory)
            memory.seek(0)
            restored = torch.load(memory, weights_only=False).eval()
            torch.testing.assert_close(restored(x)[0], expected, rtol=0, atol=0)
            restored.fuse(verbose=False)
            torch.testing.assert_close(restored(x)[0], expected, rtol=1e-4, atol=1e-4)

        trainer = TRAINERS[name].__new__(TRAINERS[name])
        trainer.data = {"nc": 2, "channels": 3}
        torch.manual_seed(123)
        reference = DetectionModel(copy.deepcopy(source.yaml), nc=2, verbose=False)
        reference.load(source, verbose=False)
        reference_rng = torch.get_rng_state().clone()
        torch.manual_seed(123)
        initial = trainer.get_model(cfg=copy.deepcopy(source.yaml), weights=source, verbose=False)
        for key, tensor in reference.state_dict().items():
            torch.testing.assert_close(initial.state_dict()[key], tensor, rtol=0, atol=0)
        assert torch.equal(torch.get_rng_state(), reference_rng)
        rebuilt = trainer.get_model(cfg=copy.deepcopy(model.yaml), weights=model, verbose=False)
        for key, tensor in model.state_dict().items():
            torch.testing.assert_close(rebuilt.state_dict()[key], tensor, rtol=0, atol=0)
        report[name] = {"parameters": sum(p.numel() for p in model.parameters()),
                        "attention_parameters": sum(p.numel() for p in params),
                        "shared_weights_preserved": True, "gradient_check": "passed",
                        "fresh_trainer_initialization_matches_baseline": True,
                        "checkpoint_roundtrip": "passed", "fused_inference": "passed",
                        "trainer_reconstruction": "passed"}
    (ROOT / "attention_checks.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

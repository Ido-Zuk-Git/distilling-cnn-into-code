"""Export the torch state_dict saved by `cnn_small.py --save PATH` to the numpy teacher package.

Usage (from the repo root): python3 baseline/export_teacher.py cnn_505.pt [OUT_DIR]
OUT_DIR defaults to workspace/teacher/; writes cnn_505.npz and arch.json there (teacher.py and README.md are
hand-written and not touched). Requires torch.
"""
import json, os, sys, numpy as np, torch
sd = torch.load(sys.argv[1], map_location="cpu")
out = os.path.join(sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "workspace", "teacher"), "")
arr = {}
for k, v in sd.items():
    if k.endswith("num_batches_tracked"): continue
    head, rest = k.split(".", 1); i = int(head)
    nk = (f"features.{i}." if i < 20 else "fc.") + rest
    arr[nk] = v.numpy().astype(np.float32)
np.savez(out + "cnn_505.npz", **arr)
CLASSES = ["tench", "english_springer", "cassette_player", "chain_saw", "church", "french_horn", "garbage_truck", "gas_pump", "golf_ball", "parachute"]
chans = [32, 64, 128, 256, 256]; blocks = []; layers = []; cin = 3
for b, o in enumerate(chans):
    i = 4 * b; n = b + 1
    layers += [dict(index=i, torch="Conv2d", key=f"features.{i}", in_channels=cin, out_channels=o, kernel=3, stride=1, padding=1, bias=False, torch_state_dict_key=f"{i}"),
               dict(index=i+1, torch="BatchNorm2d", key=f"features.{i+1}", num_features=o, eps=1e-5, momentum=0.1, torch_state_dict_key=f"{i+1}"),
               dict(index=i+2, torch="ReLU"), dict(index=i+3, torch="MaxPool2d", kernel=2, stride=2)]
    blocks.append(dict(name=f"conv{n}", kind=f"conv3x3({cin}->{o})+BN+ReLU", params=[f"features.{i}.weight", f"features.{i+1}.weight", f"features.{i+1}.bias", f"features.{i+1}.running_mean", f"features.{i+1}.running_var"],
        ops=[dict(type="conv", key=f"features.{i}", kernel=3, padding=1, bias=False), dict(type="bn", key=f"features.{i+1}", eps=1e-5), dict(type="relu")]))
    blocks.append(dict(name=f"pool{n}", kind="maxpool2x2", params=[], ops=[dict(type="maxpool", kernel=2)]))
    cin = o
layers += [dict(index=20, torch="AdaptiveAvgPool2d", output_size=1), dict(index=21, torch="Flatten"),
           dict(index=22, torch="Linear", key="fc", in_features=256, out_features=10, torch_state_dict_key="22")]
blocks += [dict(name="gap", kind="global_avg_pool", params=[], ops=[dict(type="gap")]),
           dict(name="fc", kind="linear(256->10) = logits", params=["fc.weight", "fc.bias"], ops=[dict(type="linear", key="fc")]),
           dict(name="probs", kind="softmax", params=[], ops=[dict(type="softmax")])]
arch = dict(description="small CNN, 5x[conv3x3-BN-ReLU-maxpool2], GAP, linear; torch nn.Sequential from baseline/cnn_small.py net()",
    input="RGB image -> float32 tensor (3,128,128)",
    preprocess=dict(load="PIL Image.open(path).convert('RGB')", resize_shortest_side=144, resize_rule="s=144/min(w,h); size=(max(144,round(w*s)), max(144,round(h*s))); PIL Image.BILINEAR",
        center_crop=128, crop_rule="i=(h-128)//2, j=(w-128)//2", scale="divide by 255", channel_order="RGB, CHW", mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        normalise="(x/255 - mean)/std per channel"),
    classes=CLASSES, class_order_note="logit index i = CLASSES[i]; cnn_small.py builds labels via score_preds.list_images, enumerating the CLASSES list of scripts/score.py in this exact order",
    param_key_note="npz keys: features.N.* for the 20 layers of the feature stack (torch Sequential indices 0..19; torch state_dict keys are just 'N.*'), fc.weight/fc.bias for the final Linear (torch index 22). BatchNorm num_batches_tracked is omitted. Conv layers have no bias.",
    layers=layers, blocks=blocks)
json.dump(arch, open(out + "arch.json", "w"), indent=1)
print({k: v.shape for k, v in arr.items()})

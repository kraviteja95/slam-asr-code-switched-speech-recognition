# Reassembling `adapter_model.safetensors`

GitHub's web UI caps single-file uploads at 25 MB, so the 70 MB trained
LoRA adapter was split into ~18 MB parts:

    adapter_model.safetensors.part_00
    adapter_model.safetensors.part_01
    adapter_model.safetensors.part_02
    adapter_model.safetensors.part_03

## macOS / Linux — one command

```bash
cd checkpoints/slam_asr/final/lora
cat adapter_model.safetensors.part_* > adapter_model.safetensors
# optional cleanup once verified:
# rm adapter_model.safetensors.part_*
```

## Windows (PowerShell)

```powershell
cd checkpoints\slam_asr\final\lora
Get-Content adapter_model.safetensors.part_* -Encoding Byte -ReadCount 0 |
    Set-Content adapter_model.safetensors -Encoding Byte
```

## Verify

The reassembled file must be exactly **73 911 112 bytes** (~70 MB) and
loadable via safetensors:

```python
from safetensors.torch import load_file
weights = load_file('adapter_model.safetensors')
print(f'Loaded {len(weights)} tensors')
```

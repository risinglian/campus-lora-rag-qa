from pathlib import Path
import json
import math

ROOT = Path('/root/autodl-tmp/llm_workspace')
config_path = ROOT / 'configs' / 'lora_sft.yaml'
adapter_dir = ROOT / 'outputs' / 'qwen25-1.5b-campus-lora'
adapter_model = adapter_dir / 'adapter_model.safetensors'
adapter_config = adapter_dir / 'adapter_config.json'

try:
    from safetensors import safe_open
except Exception as exc:
    raise SystemExit(f'safetensors import failed: {exc!r}')

def parse_simple_yaml(path):
    result = {}
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line or ':' not in line:
            continue
        key, value = line.split(':', 1)
        result[key.strip()] = value.strip().strip('"').strip("'")
    return result

def count_safetensors(paths):
    total = 0
    tensor_count = 0
    examples = []
    for path in paths:
        with safe_open(str(path), framework='pt', device='cpu') as handle:
            for key in handle.keys():
                shape = tuple(handle.get_slice(key).get_shape())
                count = math.prod(shape)
                total += count
                tensor_count += 1
                if len(examples) < 30:
                    examples.append((path.name, key, shape, count))
    return total, tensor_count, examples

cfg = parse_simple_yaml(config_path)
adapter_cfg = json.loads(adapter_config.read_text(encoding='utf-8'))
model_path = Path(cfg.get('model_name_or_path') or adapter_cfg.get('base_model_name_or_path') or '')
if not model_path.is_absolute():
    model_path = ROOT / model_path
base_files = sorted(model_path.glob('*.safetensors'))
adapter_params, adapter_tensor_count, adapter_examples = count_safetensors([adapter_model])
base_params = None
base_tensor_count = None
if base_files:
    base_params, base_tensor_count, _ = count_safetensors(base_files)

print('source: safetensors tensor shape statistics, no fabricated values')
print('base_model_path:', model_path)
print('base_safetensors_files:', len(base_files))
if base_params is not None:
    print('base_model_params:', base_params)
    print('base_model_params_B:', round(base_params / 1_000_000_000, 6))
    print('base_tensor_count:', base_tensor_count)
else:
    print('base_model_params: unavailable, no *.safetensors found')
print('adapter_model_path:', adapter_model)
print('adapter_tensors:', adapter_tensor_count)
print('adapter_params_trainable_lora:', adapter_params)
print('adapter_params_M:', round(adapter_params / 1_000_000, 6))
print('adapter_size_MB:', round(adapter_model.stat().st_size / 1024 / 1024, 4))
if base_params:
    print('trainable_ratio_vs_base_percent:', round(adapter_params / base_params * 100, 6))
    print('trainable_ratio_vs_base_plus_adapter_percent:', round(adapter_params / (base_params + adapter_params) * 100, 6))
print('peft_r:', adapter_cfg.get('r'))
print('peft_lora_alpha:', adapter_cfg.get('lora_alpha'))
print('peft_lora_dropout:', adapter_cfg.get('lora_dropout'))
print('peft_target_modules:', adapter_cfg.get('target_modules'))
print('peft_task_type:', adapter_cfg.get('task_type'))
print('first_30_adapter_tensors:')
for file_name, key, shape, count in adapter_examples:
    print(f'- {file_name} :: {key} :: shape={shape} :: params={count}')

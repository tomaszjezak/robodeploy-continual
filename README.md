# RoboDeploy Continual Learning

PI0.5 VLA policy with SOP-style online continual learning in LIBERO simulation.

**Goal**: Overfit a base VLA policy to deployment industrial environments via continuous online updates, achieving 90%+ reliability outside lab/demo settings.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LIBERO Simulation                             │
│  ┌──────────────┐    ┌─────────────┐    ┌──────────────────┐   │
│  │ Franka Panda │────│ PI0.5 Policy│────│ Domain Variations│   │
│  └──────────────┘    └──────┬──────┘    └──────────────────┘   │
│                              │                                   │
└──────────────────────────────┼───────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Collection                               │
│  ┌────────────────┐    ┌──────────────────┐    ┌────────────┐  │
│  │ Episode Buffer │────│ Failure Priority │────│ Corrections │  │
│  └────────────────┘    └──────────────────┘    └────────────┘  │
└──────────────────────────────┼───────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Cloud Learner                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │ LoRA Trainer│────│ Replay Buffer│────│ Metrics Monitor   │  │
│  └─────────────┘    └──────────────┘    └───────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Setup Environment

```bash
# On your cluster
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh

# Or manually
pip install -r requirements.txt
pip install git+https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install git+https://github.com/Physical-Intelligence/openpi.git
```

### 2. Download PI0.5 Checkpoint

```bash
python scripts/download_checkpoint.py --model TensorAuto/tPi0.5-libero
```

### 3. Run Baseline Evaluation

```bash
# Full evaluation
python scripts/eval_baseline.py --task=libero_long --n-episodes=50

# Quick test (mock mode)
python scripts/eval_baseline.py --mock --n-episodes=10
```

### 4. Run Continual Learning Loop

```bash
# Full loop
python scripts/run_continual.py --task=libero_long --cycles=10 --shift-level=medium

# Quick test (mock mode)
python scripts/run_continual.py --mock --cycles=5
```

### 5. Visualize Results

```bash
python scripts/plot_flywheel.py --log-dir=logs/ --show
```

## Project Structure

```
robodeploy_continual/
├── configs/
│   ├── libero_config.yaml       # LIBERO task settings
│   ├── pi05_config.yaml         # PI0.5 model config
│   ├── lora_config.yaml         # LoRA hyperparameters
│   └── continual_config.yaml    # Online learning params
├── src/
│   ├── policy/
│   │   ├── pi05_wrapper.py      # PI0.5 inference wrapper
│   │   └── action_chunking.py   # Efficient action output
│   ├── data/
│   │   ├── episode_buffer.py    # Episode storage
│   │   ├── failure_mining.py    # Entropy-based prioritization
│   │   └── corrections.py       # Oracle/scripted corrections
│   ├── training/
│   │   ├── lora_finetune.py     # LoRA training loop
│   │   └── replay_strategy.py   # Anti-forgetting replay
│   └── continual/
│       ├── online_learner.py    # Main SOP-style orchestrator
│       ├── metrics_monitor.py   # Success/entropy tracking
│       ├── weight_sync.py       # Model update distribution
│       └── dashboard.py         # Metrics visualization
├── scripts/
│   ├── setup_env.sh             # Environment setup
│   ├── download_checkpoint.py   # Get PI0.5 weights
│   ├── eval_baseline.py         # Baseline evaluation
│   ├── run_finetune.py          # LoRA fine-tuning
│   ├── run_continual.py         # Run full loop
│   └── plot_flywheel.py         # Visualize success over cycles
├── checkpoints/                  # Model weights
├── logs/                         # Training logs, metrics
├── data/                         # Episode buffers
└── requirements.txt
```

## Key Features

### SOP-style Online Post-Training

Based on AGIBOT's Scalable Online Post-training (arXiv 2601.03044):
- Robots collect on-policy data during deployment
- Priority queue for failure cases and high-entropy episodes
- HG-DAgger style corrections from oracle policy
- Async weight sync for continuous operation

### Reliability Thresholds

| Metric | Threshold | Action |
|--------|-----------|--------|
| Success rate (50-ep window) | < 90% | Trigger fine-tune batch |
| Action entropy | > 2.0 | Flag as "pre-fail", prioritize |
| Consecutive failures | >= 3 | Immediate correction injection |

### Failure Categories

Automatic categorization for targeted improvement:
- `grasp_slip` - Object dropped after grasp
- `reach_miss` - Failed to reach target position
- `collision` - Unintended contact
- `timeout` - Exceeded max steps
- `occlusion` - Lost visual tracking

### Memory Optimization

For 16GB GPU:
- LoRA rank 8 (not 16)
- bf16 mixed precision
- Gradient checkpointing
- Batch size 1, accumulation 8
- 8-bit loading fallback (bitsandbytes)

## Configuration

### Modify Domain Shift

Edit `configs/libero_config.yaml`:

```yaml
domain_shift:
  current_level: "high"  # none, low, medium, high
```

### Adjust LoRA Settings

Edit `configs/lora_config.yaml`:

```yaml
lora:
  r: 8        # Increase to 16 if memory allows
  lora_alpha: 32
```

### Change Update Triggers

Edit `configs/continual_config.yaml`:

```yaml
thresholds:
  success_rate:
    trigger_threshold: 0.85  # More aggressive updates
```

## Expected Results

| Phase | Success Rate | Notes |
|-------|--------------|-------|
| Baseline (zero-shot) | 60-70% | Before any fine-tuning |
| After LoRA fine-tune | 80-90% | Overfitted to deployment env |
| After continual loop | 90-95% | Adaptive to domain shifts |

## Troubleshooting

### GPU OOM

```bash
# Enable 8-bit loading
python scripts/run_continual.py --8bit

# Or edit configs/pi05_config.yaml:
# model:
#   load_in_8bit: true
```

### LIBERO Import Errors

```bash
# Reinstall with dependencies
pip uninstall libero robosuite mujoco
pip install mujoco>=3.0.0
pip install git+https://github.com/Lifelong-Robot-Learning/LIBERO.git
```

### Rendering Issues (Headless)

```bash
# Set environment variables
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

## References

- [PI0.5 / OpenPI](https://github.com/Physical-Intelligence/openpi)
- [LIBERO Benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [SOP Paper](https://arxiv.org/abs/2601.03044) - Scalable Online Post-training
- [LeRobot](https://github.com/huggingface/lerobot)

## License

MIT

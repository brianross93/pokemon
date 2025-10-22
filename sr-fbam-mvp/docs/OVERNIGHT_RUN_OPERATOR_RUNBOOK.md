# Overnight Capture Operator Runbook

## Pre-Flight Checklist

### 1. System Health Check
- [ ] PyBoy emulator loads ROM successfully
- [ ] RAM addresses calibrated (`0xD13A` for player_x, `0xD122` for player_y)
- [ ] Telemetry pipeline tested with short run
- [ ] Metadata JSON generation working
- [ ] Watchdog thresholds configured

### 2. Overnight Run Configuration
```bash
python scripts/run_overworld_controller.py \
  --rom "Pokemon Blue.gb" \
  --steps 10000 \
  --telemetry-out runs/overnight_$(date +%Y%m%d_%H%M).jsonl \
  --metadata-out runs/overnight_$(date +%Y%m%d_%H%M).meta.json \
  --planner-backend fake \
  --planner-cache-size 64 \
  --planner-cache-ttl 900 \
  --planlet-watchdog-steps 900 \
  --stall-watchdog-steps 600 \
  --watchdog-save-slot 1
```

### 3. Watchdog Tuning Knobs

| Parameter | Default | Purpose | Tuning |
|-----------|---------|---------|--------|
| `--planlet-watchdog-steps` | 900 | Max steps per planlet | Increase if plans need more time |
| `--stall-watchdog-steps` | 600 | Max steps without progress | Decrease if system gets stuck |
| `--planner-cache-ttl` | 900 | Cache expiration (seconds) | Increase for longer runs |
| `--planner-cache-size` | 64 | Max cached plans | Increase for complex scenarios |

## Failure Signatures & Recovery

### 1. Planlet Stall Loop
**Symptoms**: Continuous `PLANLET_STALLED` with `no-path` reason
**Recovery**: 
- Check if system is in menu state
- Verify RAM addresses are correct
- Increase `--planlet-watchdog-steps` if plans need more time
- Decrease `--stall-watchdog-steps` to force faster replanning

### 2. PyBoy Save State Failure
**Symptoms**: `RuntimeError: PyBoy build does not support save_state`
**Recovery**: 
- Remove `--watchdog-save-slot` parameter
- System will continue without save state recovery
- Watchdog will still work for planlet/stall detection

### 3. Memory Address Drift
**Symptoms**: Telemetry shows constant coordinates or menu state
**Recovery**:
- Re-run `scripts/debug_overworld_addresses.py` to recalibrate
- Update `DEFAULT_ADDRS` in `src/middleware/pyboy_adapter.py`
- Restart capture with corrected addresses

### 4. Telemetry Pipeline Issues
**Symptoms**: JSONL files not growing or Parquet conversion fails
**Recovery**:
- Check disk space: `df -h`
- Verify JSONL file permissions
- Test ETL: `python scripts/summarize_telemetry.py --input runs/test.jsonl --output runs/test.parquet`

## Monitoring During Run

### 1. Real-time Health Checks
```bash
# Check telemetry file growth
watch -n 30 'wc -l runs/overnight_*.jsonl'

# Monitor system resources
watch -n 60 'ps aux | grep python'

# Check for error patterns
tail -f runs/overnight_*.jsonl | grep -i error
```

### 2. Parquet Summary Generation
```bash
# Generate periodic summaries
python scripts/summarize_telemetry.py \
  --input runs/overnight_*.jsonl \
  --output runs/overnight_summary.parquet \
  --print-summary
```

### 3. Expected Metrics
- **Gate mix**: Should show ASSOC, FOLLOW, HALT modes
- **Plan source**: Mix of planner and fallback sources
- **Cache hit rate**: Should improve over time
- **Menu state**: Should transition between menu/overworld

## Post-Run Validation

### 1. Data Quality Check
```bash
# Verify telemetry completeness
python scripts/summarize_telemetry.py \
  --input runs/overnight_*.jsonl \
  --output runs/overnight_validation.parquet \
  --print-summary
```

### 2. Feature Store Rebuild
```bash
# Rebuild with new telemetry
python scripts/build_plan_feature_store.py \
  --battle data/planlets/battle_train.jsonl \
  --config configs/train_plan.yaml \
  --output data/planlets/plan_feature_store.pt \
  --weights-out data/planlets/plan_feature_weights.pt
```

### 3. Training Pipeline Test
```bash
# Test mixed-mode trainer
python -m src.training.train_battle_il \
  --train data/planlets/battle_train.jsonl \
  --curriculum-config configs/train_plan.yaml \
  --metrics-out results/train_plan/metrics.json \
  --batch-size 64 --gate-weight 0.5 --epochs 2
```

## Emergency Procedures

### 1. System Hang
- Check process: `ps aux | grep python`
- Kill if necessary: `pkill -f run_overworld_controller`
- Restart with same parameters

### 2. Disk Space Issues
- Check space: `df -h`
- Compress old telemetry: `gzip runs/old_*.jsonl`
- Clean up temp files: `rm -rf /tmp/pyboy_*`

### 3. Memory Leaks
- Monitor memory: `htop` or `top`
- Restart if memory usage > 80%
- Consider reducing `--planner-cache-size`

## Success Criteria

- [ ] Telemetry file grows consistently
- [ ] No critical errors in logs
- [ ] Gate diversity in summaries
- [ ] Menu state transitions captured
- [ ] Feature store rebuilds successfully
- [ ] Training pipeline shows improved metrics

#!/usr/bin/env bash
# Orchestrate the self-distillation loop (LLM-on -> relabel -> retrain -> LLM-off).
# Requires a config file containing stage commands (see configs/self_distill).

set -euo pipefail

RUN_ID="$(date -u +"%Y%m%dT%H%M%SZ")"
OUTPUT_ROOT="runs/self_distill"
CONFIG_PATH="configs/self_distill/template.env"
TEACHER_CHECKPOINT=""
TEACHER_SCORECARD=""
RELABEL_TAG_FILE=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/self_distill.sh [options]

Options:
  --config PATH               Path to stage command config (.env style). Default: configs/self_distill/template.env
  --run-id ID                 Override run identifier (default = UTC timestamp).
  --output-root DIR           Root directory for run artifacts (default = runs/self_distill).
  --teacher-checkpoint PATH   Existing checkpoint used for LLM-on data generation.
  --teacher-scorecard PATH    JSON/CSV scorecard emitted by teacher evaluation.
  --relabel-tag-file PATH     Expected planlet metadata file produced during relabel stage.
  --dry-run                   Print resolved commands without executing them.
  --help                      Show this message.

Stage commands can also be overridden via environment variables:
  LLM_ON_CMD, RELABEL_CMD, RETRAIN_CMD, LLM_OFF_CMD
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --teacher-checkpoint)
      TEACHER_CHECKPOINT="$2"
      shift 2
      ;;
    --teacher-scorecard)
      TEACHER_SCORECARD="$2"
      shift 2
      ;;
    --relabel-tag-file)
      RELABEL_TAG_FILE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift 1
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

RUN_DIR="${OUTPUT_ROOT%/}/${RUN_ID}"
LOG_DIR="$RUN_DIR/logs"
META_DIR="$RUN_DIR/metadata"
ARTIFACT_DIR="$RUN_DIR/artifacts"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
METADATA_FILE="$META_DIR/metadata.json"

mkdir -p "$LOG_DIR" "$META_DIR" "$ARTIFACT_DIR" "$CHECKPOINT_DIR"

if [[ -f "$CONFIG_PATH" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG_PATH"
else
  echo "[warn] Config file $CONFIG_PATH not found; relying on environment overrides only." >&2
fi

LLM_ON_CMD="${LLM_ON_CMD:-}"
RELABEL_CMD="${RELABEL_CMD:-}"
RETRAIN_CMD="${RETRAIN_CMD:-}"
LLM_OFF_CMD="${LLM_OFF_CMD:-}"
expand_tokens() {
  local value="$1"
  value="${value//\{RUN_ID\}/$RUN_ID}"
  echo "$value"
}

LLM_ON_CMD="$(expand_tokens "$LLM_ON_CMD")"
RELABEL_CMD="$(expand_tokens "$RELABEL_CMD")"
RETRAIN_CMD="$(expand_tokens "$RETRAIN_CMD")"
LLM_OFF_CMD="$(expand_tokens "$LLM_OFF_CMD")"
RELABEL_TAG_FILE="$(expand_tokens "${RELABEL_TAG_FILE:-}")"
TEACHER_SCORECARD="$(expand_tokens "${TEACHER_SCORECARD:-}")"

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[dry-run] run_id=$RUN_ID"
  echo "[dry-run] llm-on:     ${LLM_ON_CMD:-<empty>}"
  echo "[dry-run] relabel:    ${RELABEL_CMD:-<empty>}"
  echo "[dry-run] retrain:    ${RETRAIN_CMD:-<empty>}"
  echo "[dry-run] llm-off:    ${LLM_OFF_CMD:-<empty>}"
  exit 0
fi

python - <<'PY'
import json, sys, time
run_dir, metadata_path, config_path, teacher_ckpt, scorecard = sys.argv[1:6]
payload = {
    "run_id": sys.argv[6],
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "config": config_path,
    "teacher_checkpoint": teacher_ckpt or None,
    "teacher_scorecard": scorecard or None,
    "stages": [],
    "run_dir": run_dir,
}
path = metadata_path
with open(path, "w", encoding="utf-8") as fout:
    json.dump(payload, fout, indent=2)
PY "$RUN_DIR" "$METADATA_FILE" "$CONFIG_PATH" "$TEACHER_CHECKPOINT" "$TEACHER_SCORECARD" "$RUN_ID"

update_metadata() {
  local stage="$1"
  local status="$2"
  local command="$3"
  local log_file="$4"
  python - <<'PY'
import json, sys, time
metadata_path, stage, status, command, log_file = sys.argv[1:6]
with open(metadata_path, "r", encoding="utf-8") as fin:
    data = json.load(fin)
entry = {
    "stage": stage,
    "status": status,
    "command": command or None,
    "log": log_file or None,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
data.setdefault("stages", []).append(entry)
with open(metadata_path, "w", encoding="utf-8") as fout:
    json.dump(data, fout, indent=2)
PY "$METADATA_FILE" "$stage" "$status" "$command" "$log_file"
}

record_relabel_tags() {
  local tag_path="$1"
  local scorecard="$2"
  local output_path="$3"
  python - <<'PY'
import json, sys
tag_path, scorecard, output_path, teacher_ckpt = sys.argv[1:5]
payload = {
    "status": "pending",
    "planlet_tag_file": tag_path or None,
    "teacher_scorecard": scorecard or None,
    "teacher_checkpoint": teacher_ckpt or None,
}
with open(output_path, "w", encoding="utf-8") as fout:
    json.dump(payload, fout, indent=2)
PY "$tag_path" "$scorecard" "$output_path" "$TEACHER_CHECKPOINT"
}

run_stage() {
  local stage="$1"
  local command="$2"
  if [[ -z "$command" ]]; then
    echo "[$stage] no command configured; skipping."
    update_metadata "$stage" "skipped" "" ""
    return
  fi
  local log_file="$LOG_DIR/${stage}.log"
  echo "[$stage] executing: $command"
  set +e
  eval "$command" > >(tee "$log_file") 2>&1
  local exit_code=$?
  set -e
  if [[ $exit_code -ne 0 ]]; then
    echo "[$stage] failed with exit code $exit_code"
    update_metadata "$stage" "failed" "$command" "$log_file"
    echo "[fatal] aborting self-distillation run."
    exit $exit_code
  fi
  update_metadata "$stage" "completed" "$command" "$log_file"
  if [[ "$stage" == "relabel" ]]; then
    record_relabel_tags "$RELABEL_TAG_FILE" "$TEACHER_SCORECARD" "$META_DIR/relabel_tags.json"
  fi
  if [[ "$stage" == "retrain" ]]; then
    python - <<'PY'
import json, os, sys
metadata_path, checkpoint_dir = sys.argv[1:3]
latest = None
for name in sorted(os.listdir(checkpoint_dir), reverse=True):
    if name.endswith(".pt") or name.endswith(".pth"):
        latest = os.path.join(checkpoint_dir, name)
        break
if latest is None:
    sys.exit(0)
with open(metadata_path, "r", encoding="utf-8") as fin:
    data = json.load(fin)
data["student_checkpoint"] = latest
with open(metadata_path, "w", encoding="utf-8") as fout:
    json.dump(data, fout, indent=2)
PY "$METADATA_FILE" "$CHECKPOINT_DIR"
  fi
}

run_stage "llm_on" "$LLM_ON_CMD"
run_stage "relabel" "$RELABEL_CMD"
run_stage "retrain" "$RETRAIN_CMD"
run_stage "llm_off" "$LLM_OFF_CMD"

python - <<'PY'
import json, sys, time
metadata_path = sys.argv[1]
with open(metadata_path, "r", encoding="utf-8") as fin:
    data = json.load(fin)
data["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
with open(metadata_path, "w", encoding="utf-8") as fout:
    json.dump(data, fout, indent=2)
PY "$METADATA_FILE"

echo "[done] self-distillation run stored in $RUN_DIR"

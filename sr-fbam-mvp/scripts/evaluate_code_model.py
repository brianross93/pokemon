import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data.frame_dataset import load_datasets
from src.models.fbam_code_baseline import PureFBAMCodeAgent
from src.models.fbam_sparse_memory import FBAMSparseMemoryAgent, FBAMSparseMemoryConfig
from src.models.sr_fbam_code import SRFBAMCodeAgent
from src.training.train_fbam_code import evaluate as evaluate_fbam
from src.training.train_fbam_sparse import evaluate as evaluate_sparse
from src.training.train_srfbam_code import evaluate as evaluate_srfbam


def load_checkpoint(model, checkpoint_path: Path | None, device: torch.device) -> None:
    if not checkpoint_path:
        return
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)


def _episode_to_actions(episode):
    actions = [step.action_index for step in episode.steps]
    return torch.tensor(actions, dtype=torch.long)


def _collect_episode_metrics(model, dataset, device, profile_latency: bool) -> tuple[list[dict], dict]:
    model.eval()
    per_episode: list[dict] = []
    wall_times: list[float] = []
    accuracies: list[float] = []
    total_steps = 0
    total_correct = 0

    profile_totals: dict[str, float] = defaultdict(float)
    profile_episodes = 0

    with torch.no_grad():
        for episode in dataset.episodes:
            frames = [step.frame for step in episode.steps]
            targets = _episode_to_actions(episode)
            num_steps = len(episode.steps)

            profiling_supported = bool(
                profile_latency
                and hasattr(model, "enable_profiling")
                and hasattr(model, "profile_stats")
            )
            if profiling_supported:
                model.enable_profiling()  # type: ignore[attr-defined]
            elif profile_latency and hasattr(model, "reset_profile_stats"):
                model.reset_profile_stats()  # type: ignore[attr-defined]

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            call_input = episode if isinstance(model, SRFBAMCodeAgent) else frames

            start = perf_counter()
            preds = model.predict_actions(call_input)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            wall_time_ms = (perf_counter() - start) * 1000.0

            preds_cpu = preds.detach().cpu()
            correct = int((preds_cpu == targets).sum().item())
            accuracy = float(correct / num_steps) if num_steps else 0.0

            entry: dict = {
                "episode_id": episode.episode_id,
                "num_steps": num_steps,
                "wall_time_ms": wall_time_ms,
                "accuracy": accuracy,
                "correct": correct,
            }

            if profiling_supported:
                stats = model.profile_stats()  # type: ignore[attr-defined]
                model.disable_profiling()  # type: ignore[attr-defined]
                entry["profile"] = stats
                profile_episodes += 1
                for key, value in stats.items():
                    profile_totals[key] += float(value)

            per_episode.append(entry)
            wall_times.append(wall_time_ms)
            accuracies.append(accuracy)
            total_steps += num_steps
            total_correct += correct

    summary: dict[str, float | dict] = {
        "episodes": len(per_episode),
        "total_steps": total_steps,
        "overall_accuracy": float(total_correct / total_steps) if total_steps else 0.0,
    }
    if per_episode:
        summary["wall_time_ms_mean"] = mean(wall_times)
        summary["wall_time_ms_std"] = stdev(wall_times) if len(wall_times) > 1 else 0.0
        summary["accuracy_mean"] = mean(accuracies)
        summary["accuracy_std"] = stdev(accuracies) if len(accuracies) > 1 else 0.0

    if profile_totals and profile_episodes > 0:
        totals = dict(profile_totals)
        summary["profile_totals"] = totals
        summary["profile_episodes"] = profile_episodes
        profile_averages: dict[str, float] = {}
        read_calls = totals.get("read_calls", 0.0)
        write_calls = totals.get("write_calls", 0.0)
        total_gate_events = totals.get("write_calls", 0.0) + totals.get("writes_skipped", 0.0)
        if read_calls > 0:
            profile_averages["neighbors_requested_per_read"] = totals.get("neighbors_requested", 0.0) / read_calls
            profile_averages["neighbors_returned_per_read"] = totals.get("neighbors_returned", 0.0) / read_calls
            profile_averages["faiss_search_ms_per_read"] = totals.get("faiss_search_ms", 0.0) / read_calls
            profile_averages["query_proj_ms_per_read"] = totals.get("query_proj_ms", 0.0) / read_calls
            profile_averages["read_time_ms_per_read"] = totals.get("read_time_ms", 0.0) / read_calls
        if write_calls > 0:
            profile_averages["write_time_ms_per_write"] = totals.get("write_time_ms", 0.0) / write_calls
            profile_averages["write_proj_ms_per_write"] = totals.get("write_proj_ms", 0.0) / write_calls
        if total_gate_events > 0:
            profile_averages["avg_gate_value"] = totals.get("gate_sum", 0.0) / total_gate_events
        summary["profile_averages"] = profile_averages

    return per_episode, summary


def compute_metrics(
    model_name: str,
    data_dir: Path,
    checkpoint: Path | None,
    device: torch.device,
    memory_slots: int = 500,
    memory_dim: int = 128,
    k_neighbors: int = 10,
    index_desc: str = "Flat",
    min_write_gate: float = 1e-3,
    eval_split: str = "eval",
    profile_latency: bool = False,
    collect_episode_metrics: bool = False,
) -> tuple[dict, list[dict]]:
    train_dataset, eval_dataset = load_datasets(data_dir)
    num_actions = train_dataset.num_actions or eval_dataset.num_actions
    if model_name == 'fbam':
        model = PureFBAMCodeAgent(num_actions=num_actions)
        evaluate_fn = evaluate_fbam
    elif model_name == 'fbam_sparse':
        config = FBAMSparseMemoryConfig(
            memory_slots=memory_slots,
            memory_dim=memory_dim,
            k_neighbors=k_neighbors,
            index_description=index_desc,
            min_write_gate=min_write_gate,
        )
        model = FBAMSparseMemoryAgent(num_actions=num_actions, config=config)
        evaluate_fn = evaluate_sparse
    elif model_name == 'srfbam':
        model = SRFBAMCodeAgent(num_actions=num_actions)
        evaluate_fn = evaluate_srfbam
    else:
        raise ValueError(f'Unsupported model: {model_name}')

    load_checkpoint(model, checkpoint, device)
    model.to(device)

    dataset = eval_dataset if eval_split == "eval" else train_dataset
    metrics = evaluate_fn(model, dataset, device)
    metrics['num_episodes'] = len(dataset.episodes)
    metrics['total_steps'] = sum(len(ep.steps) for ep in dataset.episodes)
    metrics['data_dir'] = str(data_dir)
    metrics['model'] = model_name
    metrics['checkpoint'] = str(checkpoint) if checkpoint else None
    metrics['eval_split'] = eval_split

    per_episode: list[dict] = []
    if collect_episode_metrics:
        per_episode, summary = _collect_episode_metrics(model, dataset, device, profile_latency)
        metrics['latency_wall_time_ms_mean'] = summary.get('wall_time_ms_mean', 0.0)
        metrics['latency_wall_time_ms_std'] = summary.get('wall_time_ms_std', 0.0)
        metrics['latency_accuracy_mean'] = summary.get('accuracy_mean', 0.0)
        metrics['latency_accuracy_std'] = summary.get('accuracy_std', 0.0)
        metrics['latency_overall_accuracy'] = summary.get('overall_accuracy', 0.0)
        if 'profile_totals' in summary:
            metrics['profile_totals'] = summary['profile_totals']
            metrics['profile_averages'] = summary.get('profile_averages', {})
            metrics['profile_episodes'] = summary.get('profile_episodes', 0)

    return metrics, per_episode


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate FBAM/SR-FBAM on frame/action datasets without retraining.')
    parser.add_argument('--model', choices=['fbam', 'fbam_sparse', 'srfbam'], required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='Device for evaluation (default: cuda).')
    parser.add_argument('--metrics-out', type=Path)
    parser.add_argument('--memory-slots', type=int, default=500)
    parser.add_argument('--memory-dim', type=int, default=128)
    parser.add_argument('--k-neighbors', type=int, default=10)
    parser.add_argument('--index-desc', type=str, default='Flat')
    parser.add_argument('--min-write-gate', type=float, default=1e-3)
    parser.add_argument('--eval-split', choices=['train', 'eval'], default='eval')
    parser.add_argument('--profile-latency', action='store_true')
    parser.add_argument('--per-episode-out', type=Path)
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    device = torch.device(args.device)
    collect_episode_metrics = args.profile_latency or args.per_episode_out is not None
    metrics, per_episode = compute_metrics(
        args.model,
        args.data_dir,
        args.checkpoint,
        device,
        memory_slots=args.memory_slots,
        memory_dim=args.memory_dim,
        k_neighbors=args.k_neighbors,
        index_desc=args.index_desc,
        min_write_gate=args.min_write_gate,
        eval_split=args.eval_split,
        profile_latency=args.profile_latency,
        collect_episode_metrics=collect_episode_metrics,
    )

    payload = json.dumps(metrics, indent=2)
    print(payload)
    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(payload, encoding='utf-8')
    if args.per_episode_out:
        args.per_episode_out.parent.mkdir(parents=True, exist_ok=True)
        episode_payload = json.dumps(per_episode, indent=2)
        args.per_episode_out.write_text(episode_payload, encoding='utf-8')


if __name__ == '__main__':
    main()

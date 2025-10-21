import argparse
import json
from pathlib import Path
from copy import deepcopy

TARGET_STEPS = 1000


def load_steps(path: Path):
    episodes = {}
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            episodes.setdefault(record['episode_id'], []).append(record)
    return episodes


def load_summaries(path: Path):
    summaries = {}
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            summaries[record['episode_id']] = record
    return summaries


def repeat_episode(base_steps, base_summary, new_id, target_steps=TARGET_STEPS):
    steps_out = []
    cumulative_ms = 0.0
    rep = 0
    while len(steps_out) < target_steps:
        for record in base_steps:
            if len(steps_out) >= target_steps:
                break
            step = deepcopy(record)
            step_number = len(steps_out) + 1
            cumulative_ms += step['delta_wall_time_ms']
            step['episode_id'] = new_id
            step['step'] = step_number
            step['frame_id'] = f"{new_id}_step{step_number:04d}"
            step['cumulative_wall_time_ms'] = cumulative_ms
            steps_out.append(step)
        rep += 1
        if rep > 10:
            break
    summary = deepcopy(base_summary)
    summary['episode_id'] = new_id
    summary['num_steps'] = len(steps_out)
    summary['total_wall_time_ms'] = cumulative_ms
    return summary, steps_out


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')



def write_step_jsonl(path: Path, episodes_steps):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for step_list in episodes_steps:
            for record in step_list:
                f.write(json.dumps(record) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=Path, default=Path('data/episodes'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/episodes_1000_long'))
    parser.add_argument('--episodes', type=int, default=5)
    args = parser.parse_args()

    base_steps = load_steps(args.base_dir / 'eval.jsonl')
    base_summaries = load_summaries(args.base_dir / 'episodes.jsonl')

    sorted_episode_ids = sorted(base_steps)
    if not sorted_episode_ids:
        raise SystemExit('No base episodes found')

    summaries_out = []
    steps_out = []

    for idx in range(args.episodes):
        base_id = sorted_episode_ids[idx % len(sorted_episode_ids)]
        summary, steps = repeat_episode(base_steps[base_id], base_summaries[base_id], f'episode_{idx:04d}')
        summaries_out.append(summary)
        steps_out.append(steps)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / 'train.jsonl').write_text('', encoding='utf-8')
    write_step_jsonl(output / 'train.jsonl', steps_out)
    write_step_jsonl(output / 'eval.jsonl', steps_out)
    write_jsonl(output / 'episodes.jsonl', summaries_out)

    metadata = {
        'train_file': str((output / 'train.jsonl').as_posix()),
        'eval_file': str((output / 'eval.jsonl').as_posix()),
        'train_episodes': 0,
        'eval_episodes': len(summaries_out),
        'train_total_steps': 0,
        'eval_total_steps': sum(s['num_steps'] for s in summaries_out),
        'train_average_steps': 0.0,
        'eval_average_steps': sum(s['num_steps'] for s in summaries_out) / max(1, len(summaries_out)),
    }
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f"[ok] wrote {len(summaries_out)} episodes with {metadata['eval_average_steps']:.1f} average steps to {output}")


if __name__ == '__main__':
    main()

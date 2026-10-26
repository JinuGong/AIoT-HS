#!/usr/bin/env python3
"""교수자용: teamXX_run1~3.json을 팀별로 묶어 비교하고 CSV 저장.

사용: python compare_teams.py results/*.json
전제: 팀마다 각 3회 실행. 반복 측정의 중앙값을 팀 대표값으로 사용.
"""
import csv
import glob
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

EXPECTED_RUNS = {1, 2, 3}
EXPECTED = {
    'warmup': 100,
    'repeats_per_trial': 200,
    'input_shape': '(batch_size, 3, 224, 224)',
    'model': 'fixed-four-conv-CNN (untrained, benchmark only)',
    'cudnn_benchmark': True,
}
COMPARE_SETTINGS = (
    'warmup', 'repeats_per_trial', 'input_shape', 'model',
    'precision', 'cudnn_benchmark', 'cudnn_version', 'device',
    'torch_version', 'cuda_version', 'power_mode_report',
)


def verdict(deviation_pct):
    dev = abs(deviation_pct)
    return 'OK' if dev <= 5 else ('재측정' if dev <= 10 else '점검')


def team_run_from_name(p):
    m = re.fullmatch(r'team(\d{2})_run([123])\.json', p.name, flags=re.I)
    if not m:
        raise ValueError(f'파일명이 규칙과 다릅니다: {p.name} (예: team01_run1.json)')
    return f'team{m.group(1)}', int(m.group(2))


def main(paths):
    expanded = []
    for pattern in paths:
        expanded.extend(glob.glob(pattern) or [pattern])
    files = sorted({Path(p) for p in expanded})
    if not files:
        sys.exit('JSON 파일을 지정하세요: python compare_teams.py results/*.json')
    groups = defaultdict(dict)
    for p in files:
        if not p.exists():
            raise FileNotFoundError(p)
        team, run_number = team_run_from_name(p)
        if run_number in groups[team]:
            raise ValueError(f'{team} run{run_number} 파일이 중복되어 있습니다')
        d = json.loads(p.read_text(encoding='utf-8'))
        results = {int(r['batch_size']): r for r in d['results']}
        if not {1, 16}.issubset(results):
            raise ValueError(f'{p}: B=1과 B=16 결과가 모두 필요합니다')
        for key, expected in EXPECTED.items():
            if d.get(key) != expected:
                print(f'[주의] {p.name}: {key}={d.get(key)!r} (권장: {expected!r})')
        d['_parsed_results'] = results
        groups[team][run_number] = (p, d)

    rows = []
    for team, runs in sorted(groups.items()):
        if set(runs) != EXPECTED_RUNS:
            print(f'[주의] {team}: 제출 {len(runs)}회, 필요한 실행 번호 {sorted(EXPECTED_RUNS)}')
        ds = [d for _, d in (runs[k] for k in sorted(runs))]
        for key in COMPARE_SETTINGS:
            vals = {str(d.get(key)) for d in ds}
            if len(vals) != 1:
                print(f'[주의] {team}: 실행 간 {key} 값이 다릅니다: {sorted(vals)}')
        b1 = [float(d['_parsed_results'][1]['median_ms']) for d in ds]
        b16 = [float(d['_parsed_results'][16]['images_per_second']) for d in ds]
        wall = [float(d['_parsed_results'][1]['wall_median_ms']) for d in ds
                if d['_parsed_results'][1].get('wall_median_ms') is not None]
        after_temps = [float(t) for d in ds for t in
                       d.get('sysfs_after', {}).get('temps_c', {}).values()]
        med_b1 = statistics.median(b1)
        med_b16 = statistics.median(b16)
        b1_range_pct = (max(b1) - min(b1)) / med_b1 * 100
        b16_range_pct = (max(b16) - min(b16)) / med_b16 * 100
        if max(b1_range_pct, b16_range_pct) > 5:
            print(f'[주의] {team}: 3회 재실행 간 편차가 큽니다 '
                  f'(B1 범위 {b1_range_pct:.1f}%, B16 범위 {b16_range_pct:.1f}%)')
        sample = ds[0]
        rows.append({
            'team': team,
            'runs': len(ds),
            'b1_median_ms': round(med_b1, 4),
            'b1_run_range_%': round(b1_range_pct, 1),
            'b1_wall_ms': round(statistics.median(wall), 4) if wall else '',
            'b16_img_s': round(med_b16, 2),
            'b16_run_range_%': round(b16_range_pct, 1),
            'end_snapshot_max_temp_c': max(after_temps) if after_temps else '',
            'hostname': sample.get('hostname', ''),
            'torch': sample.get('torch_version', ''),
            'cuda': sample.get('cuda_version', ''),
            'cudnn': sample.get('cudnn_version', ''),
            'power_mode': ' '.join(str(sample.get('power_mode_report', '')).split())[:60],
        })

    settings = ((r['torch'], r['cuda'], r['cudnn']) for r in rows)
    if len(set(settings)) > 1:
        print('[주의] 팀 간 PyTorch/CUDA/cuDNN 버전이 동일하지 않습니다.')
    m1 = statistics.median(r['b1_median_ms'] for r in rows)
    m16 = statistics.median(r['b16_img_s'] for r in rows)
    for r in rows:
        r['b1_dev_%'] = round((r['b1_median_ms'] / m1 - 1) * 100, 1)  # +: 느림
        r['b16_dev_%'] = round((r['b16_img_s'] / m16 - 1) * 100, 1)  # -: 느림
        r['판정'] = max((verdict(r['b1_dev_%']), verdict(r['b16_dev_%'])),
                      key=['OK', '재측정', '점검'].index)

    keys = [
        'team', 'runs', 'b1_median_ms', 'b1_dev_%', 'b1_run_range_%',
        'b16_img_s', 'b16_dev_%', 'b16_run_range_%', '판정',
        'end_snapshot_max_temp_c', 'torch', 'cuda', 'cudnn', 'power_mode',
        'hostname', 'b1_wall_ms',
    ]
    print(f'전체 중앙값: B=1 {m1:.4f} ms, B=16 {m16:.2f} images/s (팀 {len(rows)}개)')
    print(' | '.join(keys[:14]))
    for r in rows:
        print(' | '.join(str(r[k]) for k in keys[:14]))
    with open('compare_teams.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print('저장: compare_teams.csv')
    print('※ 온도 값은 실행 종료 시점의 스냅샷이며, 측정 전 기간의 최고 온도가 아닙니다.')


if __name__ == '__main__':
    main(sys.argv[1:])

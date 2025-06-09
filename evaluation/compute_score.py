# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
import os
import numpy as np
import editdistance
from collections import defaultdict



def compute_EM(target, predictions, passk):
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    EM_scores = []
    for prediction in predictions[:passk]:
        prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()][:len(target_lines)]
        if len(target_lines) != len(prediction_lines):
            EM_scores.append(0)
            continue
        if target_lines == prediction_lines:
            EM_scores.append(1)
            continue
        EM_scores.append(0)
    return any(EM_scores)


def compute_ES(target, predictions, passk):
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_str = '\n'.join(target_lines)
    ES_scores = []
    for prediction in predictions[:passk]:
        prediction_lines = [line.strip() for line in prediction.splitlines() if line.strip()][:len(target_lines)]
        prediction_str = '\n'.join(prediction_lines)
        if max(len(target_str), len(prediction_str)) == 0:
            print(1)
        ES_scores.append(
            1 - (editdistance.eval(target_str, prediction_str) / max(len(target_str), len(prediction_str)))
        )
    return max(ES_scores)


def compute_score_by_repo_with_metadata(repos, lines, stype, passk=1):
    scores = defaultdict(list)
    for line in lines:
        repo = line['metadata']['task_id'].split('/')[0]
        if repo not in repos:
            continue
        samples = [line['choices'][i]['text'] for i in range(len(line['choices']))]
        samples = [line['pred_res']]
        if stype == 'EM':
            score = compute_EM(line['metadata']['ground_truth'], samples, passk)
            scores[repo].append(score)
            line['em'] = score
        elif stype == 'ES':
            score = compute_ES(line['metadata']['ground_truth'], samples, passk)
            scores[repo].append(score)
    avg_scores = {repo: round(sum(scores[repo]) / len(scores[repo]), 4) for repo in scores}
    repo_count = {repo: len(scores[repo]) for repo in scores}
    # print(stype)
    sum_scores = 0
    for repo in avg_scores.keys():
        # print(f'{avg_scores[repo]}\t{repo_count[repo]}\t{repo}')
        sum_scores += avg_scores[repo]
    res_score = format(sum_scores / len(avg_scores) * 100, '.2f')
    # print(res_score)
    ans = '{:.2f}'.format(sum([sum(scores[repo]) for repo in scores])/sum([len(scores[repo]) for repo in scores])*100)
    print(stype, ans)
    return ans
    # sum([sum(scores[repo]) for repo in scores])
    # sum([len(scores[repo]) for repo in scores])


def load_jsonl(fname):
    with open(fname, 'r', encoding='utf8') as f:
        lines = []
        for line in f:
            lines.append(json.loads(line))
        return lines

if __name__ == '__main__':
    lang = 'python'
    benchmark = 'cceval'
    model = 'ds'

    if benchmark == 'cceval':
        repos = json.loads(open('cceval/data.json', 'r', encoding='utf-8').read())[lang]
    else:
        repo_dir = f'repos/{lang}'
        entries = os.listdir(repo_dir)
        repos = [entry for entry in entries if os.path.isdir(os.path.join(repo_dir, entry))]
    
    # file_paths = ['predictions/cceval_all.jsonl', 'predictions/cceval_all_tag.jsonl', 'predictions/cceval_2k_rgrg.jsonl']
    file_paths = [
                #   f'/data/dengle/projcoder_v1/predictions/{benchmark}{"/" + model if model != "ds" else ""}/rg_{lang}.jsonl',
                  f'/data/dengle/projcoder_v1/predictions/{benchmark}{"/" + model if model != "ds" else ""}/rc_{lang}.jsonl',
                  f'/data/dengle/projcoder_v1/predictions/{benchmark}{"/" + model if model != "ds" else ""}/ours_{lang}.jsonl',
                #   f'/data/dengle/projcoder_v1/predictions/{benchmark}/ours_{lang}_none.jsonl',
                #   f'/data/dengle/projcoder_v1/predictions/{benchmark}/ours_{lang}_uer.jsonl',
                #   f'/data/dengle/projcoder_v1/predictions/{benchmark}/ours_{lang}_fsr.jsonl',
                  f'/data/dengle/projcoder_v1/predictions/{benchmark}{"/" + model if model != "ds" else ""}/repofuse_{lang}_rg.jsonl'
                ]
    # file_paths = []
    # file_paths = ['/data/dengle/projcoder_v1/predictions/cceval/rg_python.jsonl',
    #               '/data/dengle/projcoder_v1/predictions/cceval/rc_python.jsonl',
    #               '/data/dengle/projcoder_v1/predictions/cceval/ours_python_4k_uer.jsonl',
    #               '/data/dengle/projcoder_v1/predictions/cceval/ours_python_4k_fsr.jsonl',
    #             #   '/data/dengle/projcoder_v1/predictions/cceval/ours_python.jsonl',
    #               '/data/dengle/projcoder_v1/predictions/cceval/ours_python_4k.jsonl']
    # file_paths = ['/data/dengle/projcoder_v1/predictions/cceval/ours_java_uer.jsonl', '/data/dengle/projcoder_v1/predictions/cceval/ours_java_fsr.jsonl', '/data/dengle/projcoder_v1/predictions/cceval/ours_java.jsonl', '/data/dengle/projcoder_v1/predictions/cceval/rc_java.jsonl']
    file_paths = ['/data/dengle/projcoder_v1/predictions/sota_test/examples.jsonl']
    for file_path in file_paths:
        print(file_path)
        if not os.path.exists(file_path):
            print('文件不存在')
            continue
        em = compute_score_by_repo_with_metadata(repos, load_jsonl(file_path), 'EM', passk=1)
        es = compute_score_by_repo_with_metadata(repos, load_jsonl(file_path), 'ES', passk=1)
        print(f'{em} & {es}')
        print('----------')

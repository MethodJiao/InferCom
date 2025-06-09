from collections import defaultdict
import sys
sys.path.append('/data/dengle/projcoder_v1')
from evaluation.compute_score import compute_EM, compute_ES
from evaluation.f1 import compute_identifier_match
from utils.utils import Utils
import numpy as np

def compute_id_match(fpath, lang):
    examples = Utils.load_jsonl(fpath)
    em_sum = 0
    f1_sum = 0
    for example in examples:
        target = example['metadata']['ground_truth']
        pred = example['pred_res']
        em, f1 = compute_identifier_match(pred, target, lang)
        f1_sum += f1
        em_sum += em
    
    return em_sum/len(examples) * 100, f1_sum/len(examples) * 100

def compute_score_by_repo_with_metadata(fpath, stype, passk=1, repos=[]):
    lines = Utils.load_jsonl(fpath)
    if repos:
        lines = [line for line in lines if line['metadata']['task_id'].split('/')[0] in repos]
    scores = list()
    for line in lines:
        # samples = [line['choices'][i]['text'] for i in range(len(line['choices']))]
        samples = [line['pred_res']]
        if stype == 'EM':
            score = compute_EM(line['metadata']['ground_truth'], samples, passk)
            scores.append(score)
            line['em'] = score
        elif stype == 'ES':
            score = compute_ES(line['metadata']['ground_truth'], samples, passk)
            scores.append(score)

    # print(res_score)
    Utils.dump_jsonl(lines, fpath+'_em')
    ans = sum(scores) / len(scores) * 100
    return ans

def get_rq1_table():
    for benchmark in ['projbench', 'cceval'][1:]:
        for lang in ['python', 'java']:
            for baseline in ['infile', 'repofuse', 'rc', 'ours'][2:]:
                temp = []
                for m_type in ['n3_ds', 'n3_cl', 'n3_sc']:
                    model = m_type.split('_')[-1]
                    fpath = f'/data/dengle/projcoder_v1/predictions/{benchmark}/{model + "/" if model != "ds" else ""}{baseline}_{lang}.jsonl'
                    # print(fpath)
                    em = compute_score_by_repo_with_metadata(fpath, 'EM')
                    es = compute_score_by_repo_with_metadata(fpath, 'ES')
                    id_em, id_f1 = compute_id_match(fpath=fpath, lang=lang)
                    temp.extend([em, es, id_em, id_f1])
                # print(f'{benchmark} {lang} {baseline}')
                print('& & ' + baseline + ' & ' + ' & '.join(temp) + ' \\\\')
    
def get_rq1_impovement():
    em_improvements = []
    es_improvements = []
    id_em_improvements = []
    id_f1_improvements = []
    for benchmark in ['projbench', 'cceval']:
        for lang in ['python', 'java']:
            for m_type in ['n3_ds', 'n3_cl', 'n3_sc']:
                temp_em = []
                temp_es = []
                temp_idem = []
                temp_idf1= []
                for baseline in ['infile', 'repofuse', 'rc', 'ours']:
                    model = m_type.split('_')[-1]
                    fpath = f'/data/dengle/projcoder_v1/predictions/{benchmark}/{model + "/" if model != "ds" else ""}{baseline}_{lang}.jsonl'
                    # print(fpath)
                    em = compute_score_by_repo_with_metadata(fpath, 'EM')
                    es = compute_score_by_repo_with_metadata(fpath, 'ES')
                    id_em, id_f1 = compute_id_match(fpath=fpath, lang=lang)
                    temp_em.append(float(em))
                    temp_es.append(float(es))
                    temp_idem.append(float(id_em))
                    temp_idf1.append(float(id_f1))
                # print(f'{benchmark} {lang} {baseline}')
                em_improvement = (temp_em[-1] - max(temp_em[:-1])) / max(temp_em[:-1])
                es_improvement = temp_es[-1] - max(temp_es[:-1])
                id_em_improvement = (temp_idem[-1] - max(temp_idem[:-1])) / max(temp_idem[:-1])
                id_f1_improvement = temp_idf1[-1] - max(temp_idf1[:-1])
                em_improvements.append(em_improvement)
                es_improvements.append(es_improvement)
                id_em_improvements.append(id_em_improvement)
                id_f1_improvements.append(id_f1_improvement)
                print(f'{benchmark} {lang} {em_improvement} {es_improvement} {id_em_improvement} {id_f1_improvement}')
    print('em', np.mean(em_improvements))
    print('es', np.mean(es_improvements))
    print('id_em', np.mean(id_em_improvements))
    print('id_f1', np.mean(id_f1_improvements))


def get_rq2_impovement():
    table = {
        'projbench': {
            'python': [],
            'java': []
        },
        'cceval': {
            'python': [],
            'java': []
        }
    }

    for benchmark in ['projbench', 'cceval']:
        for lang in ['python', 'java']:
            for baseline in ['none', 'uer', 'fsr']:
                fpath = f'/data/dengle/projcoder_v1/predictions/{benchmark}/ours_{lang}_{baseline}.jsonl'
                # print(fpath)
                em = compute_score_by_repo_with_metadata(fpath, 'EM')
                es = compute_score_by_repo_with_metadata(fpath, 'ES')
                id_em, id_f1 = compute_id_match(fpath=fpath, lang=lang)
                table[benchmark][lang].append([float(em), float(es), float(id_em), float(id_f1)])
    print(1)
    uer_impovements = []
    fsr_improvements = []
    id_uer_improvements = []
    id_fsr_improvements = []
    for benchmark in ['projbench', 'cceval']:
        for lang in ['python', 'java']:
            uer_improvement = table[benchmark][lang][1][0] - table[benchmark][lang][0][0]
            fsr_improvement = table[benchmark][lang][2][0] - table[benchmark][lang][0][0]
            id_uer_improvement = table[benchmark][lang][1][2] - table[benchmark][lang][0][2]
            id_fsr_improvement = table[benchmark][lang][2][2] - table[benchmark][lang][0][2]
            uer_impovements.append(uer_improvement)
            fsr_improvements.append(fsr_improvement)
            id_uer_improvements.append(id_uer_improvement)
            id_fsr_improvements.append(id_fsr_improvement)
    print(np.mean(uer_impovements), np.mean(fsr_improvements), np.mean(id_uer_improvements), np.mean(id_fsr_improvements))


def to_str(num):
    return '{:.2f}'.format(num)

def get_rq3_table():
    ans = [[],[],[],[]]
    for benchmark in ['projbench', 'cceval']:
        for lang in ['python', 'java']:
            for baseline in ['infile', 'repofuse', 'rc']:
                res = []
                files = [f'/data/dengle/projcoder_v1/predictions/{benchmark}/{baseline}_{lang}.jsonl', f'/data/dengle/projcoder_v1/predictions/{benchmark}/api_{baseline}_{lang}.jsonl']
                for idx in range(len(files)):
                    temp = []
                    fpath = files[idx]
                    em = compute_score_by_repo_with_metadata(fpath, 'EM')
                    es = compute_score_by_repo_with_metadata(fpath, 'ES')
                    id_em, id_f1 = compute_id_match(fpath=fpath, lang=lang)
                    temp = [em, es, id_em, id_f1]
                    # temp.extend([to_str(em), to_str(es), to_str(id_em), to_str(id_f1)])
                    # if idx == 1:
                    #     temp = [f'\\textbf{"{" + i + "}"}' for i in temp]
                    #     temp.insert(0, '+AIM')
                    # else:
                    #     temp.insert(0, baseline)
                    res.append(temp)
                for i in range(len(res[0])):
                    ans[i].append(res[1][i] - res[0][i])
                # for i in res:
                #     print('& & ' + ' & '.join(i) + ' \\\\')
                # print('---------------')
    for i in res:
        print(np.mean(i))




if __name__ == '__main__':
    # get_rq3_table()
    # get_rq1_impovement()
    repos = ['nanoVLM', 'FramePack', 'ZeroSearch', 'Bagel', 'dia']
    for file_name in [
        '/data/dengle/projcoder_v1/predictions/sota_test/infile.jsonl',
        '/data/dengle/projcoder_v1/predictions/sota_test/rc_python.jsonl',
        '/data/dengle/projcoder_v1/predictions/sota_test/ours_python.jsonl'
    ]:
        print(compute_score_by_repo_with_metadata(file_name, 'EM', repos=repos))
        # print(compute_score_by_repo_with_metadata(file_name, 'ES'))
        
# coding: utf-8

# ipython notebook requires this
# %matplotlib inline

# python console requires this
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('/data/dengle/projcoder_v1')
from evaluation import venn
from evaluation.compute_score import compute_EM
from utils.utils import Utils


def get_true_id(filename):
    examples = Utils.load_jsonl(filename)
    
    res = []
    for i in examples:
        em = compute_EM(i['metadata']['ground_truth'], [i['pred_res']], 1)
        if em:
            res.append(i['metadata']['task_id'])

    return res

vn_list = [
    {
        'name': 'InFile',
        'baseline': 'infile',
        'res': []
    }, 
    {
        'name': 'RepoFuse',
        'baseline': 'repofuse',
        'res': []
    },     
    {
        'name': 'RepoCoder',
        'baseline': 'rc',
        'res': []
    }, 
    {
        'name': 'ours',
        'baseline': 'ours',
        'res': []
    }, 
]

# benchmark = 'projbench'
# lang = 'python'
for benchmark in ['projbench', 'cceval']:
    for lang in ['python', 'java']:
        for i in vn_list:
            fpath = f'/data/dengle/projcoder_v1/predictions/{benchmark}/{i["baseline"]}_{lang}.jsonl'
            i['res'] = get_true_id(fpath)

        labels = venn.get_labels([i['res'] for i in vn_list], fill=['number'])
        fig, ax = venn.venn4(labels, names=[i['name'] for i in vn_list], figsize=(24, 24), dpi=384, fontsize=55)
        fig.savefig(f'venn/{benchmark}_{lang}.png', bbox_inches='tight')
        plt.close()
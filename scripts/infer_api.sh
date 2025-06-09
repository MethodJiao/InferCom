benchmark="sota_test"
lang="python"
repo_dir="repos/sota_test"
rg_file="/data/dengle/projcoder_v1/predictions/sota_test/rg_python.jsonl"

python run.py \
    --process infer_api \
    --summary_cuda -1 \
    --lang "$lang" \
    --repo_dir "$repo_dir" \
    --encode_cuda 7 \
    --benchmark "$benchmark" \
    --rg_file "$rg_file" \
    --k 4
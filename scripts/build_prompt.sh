benchmark="sota_test"
lang="python"
output="/data/dengle/projcoder_v1/prompts/$benchmark/ours_${lang}.jsonl"

if [ "$benchmark" = "cceval" ]; then
    repo_dir="/data/dengle/repofuse/crosscodeeval_rawdata"
elif [ "$benchmark" = "projbench" ]; then
    repo_dir="repos/${lang}"
elif [ "$benchmark" = "sota_test" ]; then
    repo_dir="repos/sota_test"
elif [ "$benchmark" = "repoeval_api" ]; then
    repo_dir="repos/repoeval_api"
fi

python run.py \
    --process build_prompt \
    --rc_file "/data/dengle/projcoder_v1/prompts/$benchmark/rc_$lang.jsonl" \
    --prompt_output $output \
    --uer 1 \
    --fsr 1 \
    --benchmark "$benchmark" \
    --repo_dir "$repo_dir" \
    --lang "$lang"
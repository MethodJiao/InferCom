# !/bin/sh

language_type="python"
FILE="/data/dengle/projcoder_v1/baselines/RepoCoder0/utils.py"

# 如果跑cceval注释下面：
# if [ "$language_type" = "python" ]; then
#   sed -i 's/\"\*\.java"/"\*\.py"/g' "$FILE"
#   sed -i 's/java/python/g' "$FILE"
# elif [ $language_type = "java" ]; then
#   sed -i 's/"\*\.py"/"\*\.java"/g' "$FILE"
#   sed -i 's/python/java/g' "$FILE"
# else
#   echo "未知的语言类型: $language_type"
#   exit 1
# fi

python baselines/RepoCoder0/run_pipeline.py \
    --process build_rg \
    --lang "$language_type"
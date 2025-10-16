from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
model_name = "microsoft/unixcoder-base"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# # 保存到本地目录（可选）
# model.save_pretrained("./unixcoder-base")
# tokenizer.save_pretrained("./unixcoder-base")
tokenizer = RobertaTokenizer.from_pretrained(model_name)
config = RobertaConfig.from_pretrained(model_name)
config.is_decoder = True
model = RobertaModel.from_pretrained(model_name, config=config)
tokenizer.save_pretrained("./unixcoder-base")
model.save_pretrained("./unixcoder-base")
config.save_pretrained("./unixcoder-base")
pass
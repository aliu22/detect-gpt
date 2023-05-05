import transformers



cache_dir = "/scratch/network/al34/.cache"
model = 'roberta-base-openai-detector'
detector = transformers.AutoModelForSequenceClassification.from_pretrained(model, cache_dir = cache_dir)
tokenizer = transformers.AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)


cache_dir = "/scratch/network/al34/.cache"
model = 'roberta-large-openai-detector'
detector = transformers.AutoModelForSequenceClassification.from_pretrained(model, cache_dir = cache_dir)
tokenizer = transformers.AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)

import random
import datasets

SEPARATOR = '<<<SEP>>>'


DATASETS = ['writing', 'english', 'german', 'pubmed', 'xnli_french', 'xnli_spanish', 'xnli_english', 'xnli_german', 'amazon_home', 'cnn_dailymail', 'allocine']


def load_pubmed(cache_dir):
    data = datasets.load_dataset('pubmed_qa', 'pqa_labeled', split='train', cache_dir=cache_dir)
    
    # combine question and long_answer
    data = [f'Question: {q} Answer:{SEPARATOR}{a}' for q, a in zip(data['question'], data['long_answer'])]

    return data


def process_prompt(prompt):
    return prompt.replace('[ WP ]', '').replace('[ OT ]', '')


def process_spaces(story):
    return story.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()


def load_writing(cache_dir=None):
    writing_path = 'data/writingPrompts'
    
    with open(f'{writing_path}/valid.wp_source', 'r') as f:
        prompts = f.readlines()
    with open(f'{writing_path}/valid.wp_target', 'r') as f:
        stories = f.readlines()
    
    prompts = [process_prompt(prompt) for prompt in prompts]
    joined = [process_spaces(prompt + " " + story) for prompt, story in zip(prompts, stories)]
    filtered = [story for story in joined if 'nsfw' not in story and 'NSFW' not in story]

    random.seed(0)
    random.shuffle(filtered)

    return filtered


# .load_dataset('imdb'.. )

def load_language(language, cache_dir):
    # load either the english or german portion of the wmt16 dataset
    assert language in ['en', 'de']
    d = datasets.load_dataset('wmt16', 'de-en', split='train', cache_dir=cache_dir)
    docs = d['translation']
    desired_language_docs = [d[language] for d in docs]
    lens = [len(d.split()) for d in desired_language_docs]
    sub = [d for d, l in zip(desired_language_docs, lens) if l > 100 and l < 150]
    return sub

def load_allocine(cache_dir):
    data = datasets.load_dataset('allocine', split='train', cache_dir=cache_dir)['review']
    lens = [len(d.split()) for d in data]
    sub = [d for d, l in zip(data, lens) if l > 100 and l < 150]
    return sub


def load_german(cache_dir):
    return load_language('de', cache_dir)


def load_english(cache_dir):
    return load_language('en', cache_dir)


def load_xnli_french(cache_dir):
    data = datasets.load_dataset('xnli', 'fr', split='train', cache_dir=cache_dir)
    data = [f'Premise: {q} Hypothesis:{SEPARATOR}{a}' for q, a in zip(data['premise'], data['hypothesis'])]
    lens = [len(d.split()) for d in data]
    sub = [d for d, l in zip(data, lens) if l > 100 and l < 150]
    return sub

def load_xnli_spanish(cache_dir):
    data = datasets.load_dataset('xnli', 'es', split='train', cache_dir=cache_dir)
    data = [f'Premise: {q} Hypothesis:{SEPARATOR}{a}' for q, a in zip(data['premise'], data['hypothesis'])]
    lens = [len(d.split()) for d in data]
    sub = [d for d, l in zip(data, lens) if l > 100 and l < 150]
    return sub

def load_xnli_english(cache_dir):
    data = datasets.load_dataset('xnli', 'en', split='train', cache_dir=cache_dir)
    data = [f'Premise: {q} Hypothesis:{SEPARATOR}{a}' for q, a in zip(data['premise'], data['hypothesis'])]
    lens = [len(d.split()) for d in data]
    sub = [d for d, l in zip(data, lens) if l > 100 and l < 150]
    return sub

def load_xnli_german(cache_dir):
    data = datasets.load_dataset('xnli', 'de', split='train', cache_dir=cache_dir)
    data = [f'Premise: {q} Hypothesis:{SEPARATOR}{a}' for q, a in zip(data['premise'], data['hypothesis'])]
    lens = [len(d.split()) for d in data]
    sub = [d for d, l in zip(data, lens) if l > 100 and l < 150]
    return sub

def load_amazon_home(cache_dir):
    data = datasets.load_dataset('amazon_us_reviews', 'Home_Entertainment_v1_00', split='train', cache_dir=cache_dir)['review_body']
    lens = [len(d.split()) for d in data]
    sub = [d for d, l in zip(data, lens) if l > 100 and l < 150]
    return sub

def load_cnn_dailymail(cache_dir):
    data = datasets.load_dataset('cnn_dailymail', '1.0.0', split='train', cache_dir=cache_dir)['article']
    return data

def load(name, cache_dir, **kwargs):
    if name in DATASETS:
        load_fn = globals()[f'load_{name}']
        return load_fn(cache_dir=cache_dir, **kwargs)
    else:
        raise ValueError(f'Unknown dataset {name}')

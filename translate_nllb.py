import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from tqdm import tqdm
from argparse import ArgumentParser
import re

def parse_args():
    args = ArgumentParser()
    args.add_argument("--file", type=str, default = None)
    args.add_argument("--output", type=str, default = None)
    args.add_argument("--translate-column", type=str, default = None)
    args.add_argument("--replace", default = True)
    args.add_argument("--split-sentences", default = False, choices=[True, False])
    args.add_argument("--src-lang", type=str, default = 'eng_Latn')
    args.add_argument("--tgt-lang", type=str, default = 'arb_Arab')
    args.add_argument("--data-format-in", type=str, choices=["csv", "tsv", "json", "jsonl", "txt", "parquet"])
    args.add_argument("--data-format-out", type=str, choices=["csv", "tsv", "json", "jsonl", "txt", "parquet"])
    args.add_argument("--batch-size", default=48, type = int)
    args.add_argument("--with-cuda", default=False)
    
    return args.parse_args()

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"

def split_into_sentences(text):

    if '.' not in text and len(text) > 2:
        return [text]

    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    
    return sentences

def read_data(path, data_format):
    if data_format == 'csv':
        df = pd.read_csv(path, index = False)
    elif data_format == 'tsv':
        df = pd.read_csv(path, sep = '\t', index = False)
    elif data_format == 'json':
        df = pd.read_json(path)
    elif data_format == 'jsonl':
        df = pd.read_json(path, lines = True, orient = 'records')        
    elif data_format == 'parquet':
        df = pd.read_parquet(path, engine='pyarrow')
    elif data_format == 'txt':
        with open(path, 'r') as f:
            df = pd.DataFrame(f.readlines())
    return df

def save_data(df, path, data_format):
    if data_format == 'csv':
        df.to_csv(path)
    elif data_format == 'tsv':
        df.to_csv(path, sep = '\t')
    elif data_format == 'json':
        df.to_json(path)
    elif data_format == 'jsonl':
        df.to_json(path, lines = True, orient = 'records')        
    elif data_format == 'parquet':
        df.to_parquet(path, engine='pyarrow')
    elif data_format == 'txt':
        df.to_csv(path, sep=' ', index=False)
        
def main(args):

    checkpoint = 'nllb-200-distilled-1.3B'
    model_fb = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    tokenizer_fb = AutoTokenizer.from_pretrained(checkpoint)

    device = "cpu" if args.with_cuda == False else 0

    input_lang = args.src_lang
    target_lang = args.tgt_lang
    
    translation_pipeline_fb = pipeline('translation', 
                                    model=model_fb, 
                                    tokenizer=tokenizer_fb, 
                                    src_lang=input_lang, 
                                    tgt_lang=target_lang, 
                                    max_length = 400,
                                    device = device)

    df = read_data(args.file, args.data_format_in)

    input_col = args.translate_column if args.translate_column != None else 0
    if args.replace != True:
        translated_column = args.replace
        df[translated_column] = ''
    else:
        translated_column = input_col


    if args.split_sentences == True:
        print('Translating by splitting sentences')
        for i in tqdm(df.index):

            input_text = df.at[i, input_col]
            translated_text = translation_pipeline_fb(split_into_sentences(input_text))
            translated_text  = [t['translation_text'] for t in translated_text]
    
            df.at[i, translated_column] = ' '.join(translated_text).strip()
        save_data(df, args.output, args.data_format_out)
    
    else:
        print('Translating without splitting sentences')
        num_idx = df.shape[0]
        batch_size = args.batch_size
        batches = [range(i, min(i + batch_size, num_idx)) for i in range(0, len(df), batch_size)]

        for batch_indices in tqdm(batches):

            batch_input_text = list(df.loc[batch_indices, input_col])
            translated_text = translation_pipeline_fb(batch_input_text)
            translated_text  = [t['translation_text'] for t in translated_text]

            df.loc[batch_indices, translated_column] = translated_text

        save_data(df, args.output, args.data_format_out)
    
if __name__=="__main__":
    fail = False
    args = parse_args()
    
    if args.file == None:
        print("Need a --file to translate")
        fail = True
    if args.output == None:
        print("Missing an --ouput location")
        fail = True
        
    if not fail:
        main(args)

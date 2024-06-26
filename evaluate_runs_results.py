import csv
import glob
import pickle
import re

from dotenv import load_dotenv

load_dotenv()
import numpy
from bert_score import BERTScorer
from evaluation import f1_score

import evaluate

rouge = evaluate.load('rouge')
# bertscore = BERTScorer(lang='en', device='cuda')
bertscore = BERTScorer(model_type='microsoft/deberta-xlarge-mnli', device='cuda')
_main_path = 'public_ckpt'

ADD_METEOR = True
if ADD_METEOR:
    meteor_scorer = evaluate.load('meteor')

DO_PRED_CLEAN = True



def evaluate_folder(main_path, skip_exists=True):
    results_path = f'{main_path}/results.txt'
    results_csv_path = f'{main_path}/results.csv'
    paths = glob.glob(f'{main_path}/*/evaluation_result*.pkl')
    all_results = []
    csv_results = []
    csv_results.append(['path',
                        'ppl',
                        'F1',
                        'bleu',
                        'bleu-1',
                        'bleu-2',
                        'bleu-3',
                        'bleu-4',
                        'rouge1',
                        'rouge2',
                        'rougel',
                        'BERT f1',
                        'BERT precision',
                        'BERT recall',
                        'dist-1',
                        'dist-2',
                        'meteor',
                        'valid_num'])
    for path in paths:
        with open(path, 'rb') as file:
            results = pickle.load(file)
        if results.get('result_str') is not None and skip_exists:
            all_results.append(results['result_str'])
            csv_results.append(results['csv'])
            continue
        preds = results['pred_text']
        clean_preds = []
        if DO_PRED_CLEAN:
            for pred in preds:
                search_result = re.search('R:|Q:|Summary:|\n|\:', pred)
                if search_result is not None:
                    clean_preds.append(pred[:search_result.span()[0]])
                else:
                    clean_preds.append(pred)
            preds = clean_preds
        tgt = results['gt_text']

        def bleu_score(prediction, ground_truths):
            from sacrebleu import BLEU
            bleu = BLEU()
            score = bleu.corpus_score(prediction, ground_truths)
            return score

        bleu = bleu_score(preds, [tgt])

        precision, recall, f1 = bertscore.score(preds, tgt, verbose=False, batch_size=64)
        mean_precision = precision.mean().item()
        mean_recall = recall.mean().item()
        mean_f1 = f1.mean().item()

        def eval_distinct(corpus):
            unigrams = []
            bigrams = []
            for n, rep in enumerate(corpus):
                rep = rep.strip()
                temp = rep.split(' ')
                unigrams += temp
                for i in range(len(temp) - 1):
                    bigrams.append(temp[i] + ' ' + temp[i + 1])
            distink_1 = len(set(unigrams)) * 1.0 / len(unigrams)
            distink_2 = len(set(bigrams)) * 1.0 / len(bigrams)
            return distink_1, distink_2

        rouge_results = rouge.compute(predictions=preds, references=tgt)
        rouge1, rouge2, rougel = rouge_results['rouge1'], rouge_results['rouge2'], rouge_results['rougeL']
        me_score = 0
        if ADD_METEOR:
            me_score = meteor_scorer.compute(predictions=preds, references=tgt)['meteor']
        from evaluation import rouge_score
        _rouge = rouge_score(preds, [tgt])
        f1 = [f1_score(p, [t]) for p, t in zip(preds, tgt)]
        f1 = numpy.asfarray(f1).mean()
        ppl=''

        result_str = f"""
        path: {path}
        F1: {f1}
        bleu: {bleu.score}
        bleu detail: {bleu.precisions}
        rouge1, rouge2, rougel: {rouge1, rouge2, rougel}
        BERT f1: {mean_f1}
        BERT precision: {mean_precision}
        BERT recall: {mean_recall}
        dist: {eval_distinct(preds)}
        METEOR: {me_score}
        valid_num: {len(preds)}
        """
        csv_data = [path,
                    f1 * 100.0,
                    bleu.score,
                    *bleu.precisions,
                    rouge1 * 100.0,
                    rouge2 * 100.0,
                    rougel * 100.0,
                    mean_f1 * 100.0,
                    mean_precision * 100.0,
                    mean_recall * 100.0,
                    *eval_distinct(preds),
                    me_score,
                    len(preds)]
        csv_results.append(csv_data)
        print(result_str)
        all_results.append(result_str)
        with open(path, 'wb') as file:
            results['result_str'] = result_str
            results['csv'] = csv_data
            pickle.dump(results, file)

    with open(results_path, 'w') as file:
        file.write("\n=====\n".join(all_results))
    with open(results_csv_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(csv_results)


if __name__ == '__main__':
    evaluate_folder(_main_path, skip_exists=False)

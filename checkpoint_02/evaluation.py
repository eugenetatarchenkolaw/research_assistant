"""Frozen source spans, development-only tuning and held-out reporting."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import product
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from .corpus import ROOT, Document, TOKEN_RE, corpus_fingerprint, write_json
from .metrics import per_query_metrics
from .retrieval import BM25Index
from .splitters import ChunkConfig, Chunk, split_corpus


@dataclass(frozen=True)
class ExperimentConfig:
    strategy: str = "token"
    size: int = 256
    overlap: int = 32
    mode: str = "lower"
    k1: float = 1.2
    b: float = 0.75

    @property
    def key(self) -> str:
        return f"{self.strategy}-{self.size}-{self.overlap}-{self.mode}-{self.k1:g}-{self.b:g}"

    @property
    def split(self) -> ChunkConfig:
        return ChunkConfig(self.strategy, self.size, self.overlap)


def load_questions(documents: list[Document], path: Path = ROOT / "data/questions.jsonl") -> list[dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records or len({q['question_id'] for q in records}) != len(records):
        raise ValueError("Missing questions or duplicate question IDs")
    docs = {d.document_id: d for d in documents}
    groups = {}
    for q in records:
        if not q['question'].strip() or not q['evidence'] or q['split'] not in {'dev', 'test'}:
            raise ValueError(f"Invalid question: {q['question_id']}")
        if q['group'] in groups and groups[q['group']] != q['split']:
            raise ValueError("Evidence group leaks between development and test")
        groups[q['group']] = q['split']
        for e in q['evidence']:
            d = docs[e['document_id']]
            if d.text_sha256 != q['text_sha256']:
                raise ValueError(f"Parser/corpus mismatch: {q['question_id']}; regenerate reviewed spans")
            if not 0 <= e['start'] < e['end'] <= len(d.text):
                raise ValueError("Evidence span outside source")
            if sha256(d.text[e['start']:e['end']].encode()).hexdigest() != e['sha256']:
                raise ValueError("Evidence quote hash changed")
    return records


def map_qrels(questions: list[dict], chunks: list[Chunk]) -> dict[str, list[str]]:
    """A relevant chunk contains a complete annotated source span."""
    return {q['question_id']: [c.chunk_id for c in chunks if any(
        c.document_id == e['document_id'] and c.start <= e['start'] and c.end >= e['end']
        for e in q['evidence'])] for q in questions}


def typo_query(text: str) -> str:
    candidates = [m for m in TOKEN_RE.finditer(text) if len(m.group()) >= 8 and m.group().isalpha()]
    if not candidates:
        return text
    word = max(candidates, key=lambda m: len(m.group()))
    position = word.start() + len(word.group()) // 2
    return text[:position] + text[position + 1:]


def run_config(documents: list[Document], questions: list[dict], config: ExperimentConfig,
               variant: str = 'original') -> tuple[pd.DataFrame, list[dict]]:
    if not questions:
        raise ValueError("No questions for this experiment")
    chunks = split_corpus(documents, config.split)
    t0 = time.perf_counter()
    index = BM25Index(chunks, config.k1, config.b, config.mode)
    index_seconds = time.perf_counter() - t0
    qrels = map_qrels(questions, chunks)
    rows, predictions = [], []
    for q in questions:
        query = q['question']
        if variant == 'paraphrase': query = q['paraphrase']
        elif variant == 'typo': query = typo_query(query)
        elif variant == 'case': query = query.upper() + '?!'
        elif variant != 'original': raise ValueError("Unknown query variant")
        t0 = time.perf_counter()
        found = index.search(query, 10)
        milliseconds = 1000 * (time.perf_counter() - t0)
        ids = [c.chunk_id for c, _ in found]
        gold = qrels[q['question_id']]
        prediction = dict(question_id=q['question_id'],query=query,variant=variant,
            config=config.key, gold=gold, results=[dict(chunk_id=c.chunk_id,document_id=c.document_id,
                pages=c.pages,start=c.start,end=c.end,score=score,tokens=c.tokens) for c,score in found])
        predictions.append(prediction)
        for k in (1, 5, 10):
            rows.append(dict(question_id=q['question_id'],document_id=q['document_id'],group=q['group'],
                split=q['split'],variant=variant,config=config.key,**asdict(config),k=k,
                **per_query_metrics(ids,gold,k),chunks=len(chunks),index_seconds=index_seconds,
                query_ms=milliseconds,context_tokens=sum(c.tokens for c,_ in found[:k])))
    return pd.DataFrame(rows), predictions


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    keys=['config','strategy','size','overlap','mode','k1','b','split','variant','k']
    values=['hit','precision','recall','mrr','ndcg','addressable','context_tokens','query_ms','chunks','index_seconds']
    out=rows.groupby(keys,dropna=False)[values].mean().reset_index()
    counts=rows.groupby(keys,dropna=False).size().rename('questions').reset_index()
    return out.merge(counts,on=keys,validate='one_to_one')


def run_study(documents: list[Document], questions: list[dict], output: Path = ROOT / 'results') -> dict:
    output.mkdir(parents=True,exist_ok=True)
    dev=[q for q in questions if q['split']=='dev']
    test=[q for q in questions if q['split']=='test']
    configs=[ExperimentConfig(*x) for x in product(('token','sentence'),(128,256,512),(0,32),('lower','stem'),(1.2,1.8),(.35,.75))]
    frames=[]
    for config in configs:
        rows,_=run_config(documents,dev,config)
        frames.append(aggregate(rows))
    development=pd.concat(frames,ignore_index=True)
    # Freeze the winner before the test set is queried.
    order=development[development.k==5].sort_values(['hit','mrr','context_tokens','config'],ascending=[False,False,True,True])
    winner=order.iloc[0]['config']
    selected=next(c for c in configs if c.key==winner)
    baseline=ExperimentConfig()
    write_json(output/'selection.json',dict(selected=asdict(selected),baseline=asdict(baseline),
        selection_rule='Development Hit@5, then MRR@5, then fewer context tokens, then config key',
        corpus_sha256=corpus_fingerprint(documents),questions_sha256=sha256((ROOT/'data/questions.jsonl').read_bytes()).hexdigest(),
        configurations=len(configs),development_questions=len(dev),test_questions=len(test),
        test_status='held-out source pages within the same three documents; not independent human questions'))
    test_rows,all_predictions=[],[]
    for label,config in [('baseline',baseline),('selected',selected)]:
        for variant in ['original','paraphrase','typo','case']:
            rows,preds=run_config(documents,test,config,variant)
            rows['system']=label
            test_rows.append(rows)
            all_predictions += [dict(p,system=label) for p in preds]
    details=pd.concat(test_rows,ignore_index=True)
    summary=details.groupby(['system','variant','k'])[['hit','precision','recall','mrr','ndcg','addressable','context_tokens','query_ms']].mean().reset_index()
    summary['questions']=len(test)
    development.to_csv(output/'development.csv',index=False)
    summary.to_csv(output/'test_summary.csv',index=False)
    details.to_csv(output/'test_per_query.csv',index=False)
    (output/'predictions.jsonl').write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in all_predictions),encoding='utf-8')
    return dict(development=development,test=summary,details=details,selected=selected,baseline=baseline)

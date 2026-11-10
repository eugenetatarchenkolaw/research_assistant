from dataclasses import replace
from math import log, log2
from pathlib import Path
import json
import math
import pytest

from checkpoint_02.corpus import Document, Page, clean_page, TOKEN_RE, normalize, load_corpus, ROOT
from checkpoint_02.splitters import Chunk, ChunkConfig, split_document, split_corpus
from checkpoint_02.retrieval import BM25Index, tokenize
from checkpoint_02.metrics import evaluate_retrieval, per_query_metrics, paired_group_interval
from checkpoint_02.audit import duplicate_candidates, word_frequencies
from checkpoint_02.evaluation import load_questions, map_qrels, ExperimentConfig, run_config, typo_query
from checkpoint_02.questions import validate_response, generate_questions


def doc(text='One two. Three four five. Six seven eight nine ten.'):
    return Document('d','test',text,(Page(1,0,len(text),text),),'sha','https://example.org/d.pdf')


@pytest.mark.parametrize('size,overlap',[(0,0),(-1,0),(3,3),(3,-1),(2,4),(2.5,0)])
def test_bad_chunk_parameters(size,overlap):
    with pytest.raises(ValueError): ChunkConfig(size=size,overlap=overlap)


@pytest.mark.parametrize('strategy',['token','sentence'])
@pytest.mark.parametrize('size,overlap',[(1,0),(4,0),(4,1),(4,3),(100,0)])
def test_chunk_coverage_offsets_and_bound(strategy,size,overlap):
    d=doc(); config=ChunkConfig(strategy,size,overlap)
    chunks=split_document(d,config)
    assert chunks==split_document(d,config)
    assert all(0<c.tokens<=size for c in chunks)
    assert all(d.text[c.start:c.end]==c.text for c in chunks)
    assert all(any(c.start<=m.start() and c.end>=m.end() for c in chunks) for m in TOKEN_RE.finditer(d.text))
    assert len({c.chunk_id for c in chunks})==len(chunks)


def test_split_empty_and_duplicate_ids():
    assert split_document(doc('...'),ChunkConfig())==[]
    with pytest.raises(ValueError): split_corpus([doc(),doc()],ChunkConfig())


def test_ids_change_with_text_and_configuration():
    a=split_document(doc(),ChunkConfig(size=4,overlap=1))
    b=split_document(doc(),ChunkConfig(size=5,overlap=1))
    c=split_document(doc('Other text here.'),ChunkConfig(size=4,overlap=1))
    assert not {x.chunk_id for x in a}&{x.chunk_id for x in b+c}


def test_cleaning_keeps_offsets_as_new_canonical_text():
    assert clean_page('qual-\nity and soft\u00ad\nware\n12')=='quality and software'
    assert normalize('ABC \n def')=='abc def'


def test_metrics_hand_calculation():
    r=per_query_metrics(['a','b','c'],['b','c'],3)
    assert r['precision']==pytest.approx(2/3)
    assert r['recall']==1 and r['hit']==1 and r['mrr']==.5
    assert r['ndcg']==pytest.approx((1/log2(3)+1/log2(4))/(1+1/log2(3)))


def test_empty_gold_stays_in_denominator_and_missing_predictions_zero():
    r=evaluate_retrieval({'a':['yes']},{'a':['yes'],'b':[]},1)
    assert r['hit']==.5 and r['questions']==2 and r['addressable']==.5
    assert evaluate_retrieval({}, {'a':['yes']},5)['hit']==0
    assert per_query_metrics(['yes'],['yes'],5)['precision']==.2


def test_duplicate_predictions_rejected():
    with pytest.raises(ValueError): per_query_metrics(['a','a'],['a'],5)
    with pytest.raises(ValueError): evaluate_retrieval({}, {},1)
    with pytest.raises(ValueError): evaluate_retrieval({'bad':[]},{'ok':['a']},1)
    with pytest.raises(ValueError): per_query_metrics([],[],0)


def test_bm25_exact_formula_and_no_fake_hits():
    chunks=split_document(doc('apple apple'),ChunkConfig())
    index=BM25Index(chunks)
    expected=log(1+.5/1.5)*2*2.2/(2+1.2)
    assert index.scores('apple')[0]==pytest.approx(expected)
    assert index.search('unknown')==[] and index.search('')==[]
    assert index.scores('apple apple')[0]==index.scores('apple')[0]


@pytest.mark.parametrize('k1,b',[(0,.5),(-1,.5),(math.nan,.5),(1,math.nan),(1,1.1),(1,-.1)])
def test_invalid_bm25(k1,b):
    with pytest.raises(ValueError): BM25Index(split_document(doc(),ChunkConfig()),k1,b)


def test_bm25_ties_are_stable():
    a=Chunk('b','d',0,5,'apple',(1,),1); b=replace(a,chunk_id='a')
    assert [c.chunk_id for c,_ in BM25Index([a,b]).search('apple')]==['a','b']


def test_duplicate_candidates():
    pairs=duplicate_candidates([('a','ABC def'),('b','abc  def'),('c','abc deg')],.8)
    assert len(pairs)==3 and pairs[0]['method']=='sha256'
    assert duplicate_candidates([('a',''),('b','')])==[]


def test_question_response_validation():
    text='A verified source phrase.'
    valid=json.dumps([dict(question='Which phrase?',evidence_quote='source phrase')])
    records=validate_response(valid,text,1)
    assert records[0]['start']==text.index('source phrase')
    for bad in ['{}','[]',json.dumps([dict(question='Q?',evidence_quote='invented')])]:
        with pytest.raises(ValueError): validate_response(bad,text,1)
    with pytest.raises(ValueError): validate_response(valid,text+text,1)


def test_api_mock_retry_and_provenance():
    class Response:
        def __init__(self,status): self.status_code=status
        def raise_for_status(self): assert self.status_code==200
        def json(self): return dict(model='test-model',id='id1',choices=[dict(message=dict(content='[{"question":"Q?","evidence_quote":"source"}]'))])
    class Session:
        def __init__(self): self.calls=0
        def post(self,*args,**kwargs):
            self.calls+=1
            return Response(429 if self.calls==1 else 200)
    session=Session()
    records,meta=generate_questions('source text',1,model='test',base_url='https://example.org/v1',api_key='private',session=session,sleep=lambda _:None)
    assert session.calls==2 and meta['response_id']=='id1'
    assert 'private' not in json.dumps(meta)


def test_api_requires_explicit_credentials():
    with pytest.raises(ValueError): generate_questions('source',1,model='m',base_url='https://example.org',api_key='')
    with pytest.raises(ValueError): generate_questions('source',1,model='m',base_url='http://example.org',api_key='k')


def test_paired_bootstrap_zero_difference():
    assert paired_group_interval([1,0],[1,0],['a','b'],draws=100)==(0,0)


def test_real_sources_questions_and_qrels():
    docs,audit=load_corpus(download=False)
    questions=load_questions(docs)
    assert len(audit)==105 and sum(a['included'] for a in audit)==90
    assert len(questions)==60 and not any(q['human_review'] for q in questions)
    dev={q['group'] for q in questions if q['split']=='dev'}
    test={q['group'] for q in questions if q['split']=='test'}
    assert not dev & test
    chunks=split_corpus(docs,ChunkConfig('token',256,32))
    qrels=map_qrels(questions,chunks)
    assert len(qrels)==60
    for q in questions:
        assert q['question_id'] in qrels
    rows,_=run_config(docs,questions[:2],ExperimentConfig())
    assert set(rows.k)=={1,5,10} and len(rows)==6
    assert rows[['hit','precision','recall','mrr','ndcg']].max().max()<=1


def test_tiny_chunks_do_not_drop_unaddressable_questions():
    docs,_=load_corpus(download=False); questions=load_questions(docs)[:2]
    rows,_=run_config(docs,questions,ExperimentConfig(size=1,overlap=0))
    assert len(rows)==6 and rows.hit.sum()==0 and rows.addressable.sum()==0


def test_corrupt_evidence_rejected(tmp_path):
    docs,_=load_corpus(download=False)
    questions=load_questions(docs)
    questions[0]['evidence'][0]['sha256']='broken'
    path=tmp_path/'questions.jsonl'
    path.write_text('\n'.join(json.dumps(q) for q in questions))
    with pytest.raises(ValueError): load_questions(docs,path)

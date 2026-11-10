"""Streamlit exploration; retrieval and metrics live in importable modules."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import json
import pandas as pd
import streamlit as st
from checkpoint_02.corpus import ROOT, load_corpus, corpus_fingerprint
from checkpoint_02.splitters import ChunkConfig, split_corpus
from checkpoint_02.retrieval import BM25Index
from checkpoint_02.audit import word_frequencies, duplicate_candidates
from checkpoint_02.evaluation import ExperimentConfig, load_questions, run_config, aggregate
from checkpoint_02.plots import word_cloud, plot_development
LABELS = {'TITLE': 'Исследование корпуса ГОСТов', 'SUBTITLE': 'Чекпойнт 02 / Research Assistant', 'DESCRIPTION': 'Облака слов, разбиение текста и проверка поиска BM25 на одних исходных документах.', 'WARN': 'Вопросы созданы нейросетью и сверены с фрагментами источников. Независимой проверки человеком пока нет.', 'SETTINGS': 'Параметры поиска', 'SPLITTER': 'Сплиттер', 'SIZE': 'Размер чанка, слов', 'OVERLAP': 'Перекрытие, слов', 'MODE': 'Обработка слов', 'DOCS': 'Документы', 'PAGES': 'Страницы PDF', 'SEARCH_PAGES': 'Страницы в поиске', 'CHUNKS': 'Чанки', 'CLOUD': 'Облако слов', 'SEARCH': 'Поиск BM25', 'EXPERIMENT': 'Эксперимент', 'QUALITY': 'Качество данных', 'QUESTIONS': 'Вопросы', 'SCOPE': 'Уровень анализа', 'CORPUS': 'Весь корпус', 'DOCUMENT': 'Документ', 'CHUNK': 'Чанк', 'CHOOSE_DOC': 'Выберите документ', 'STOP': 'Стоп-слова', 'STOP_NONE': 'Без стоп-слов', 'STOP_BASIC': 'Только служебные слова', 'STOP_DOMAIN': 'Также частые слова домена', 'STOP_NOTE': 'Стоп-список меняет только облако. BM25 продолжает использовать свой словарь.', 'TOPWORDS': 'Частотная таблица', 'FRAGMENT': 'Исходный фрагмент', 'DOWNLOAD': 'Сохранить частоты CSV', 'QUERY': 'Вопрос к корпусу', 'DEFAULT_QUERY': 'Как управлять документированной информацией?', 'EMPTY': 'Совпадений по словам нет. Нулевые оценки не выдаем за найденный ответ.', 'OPEN': 'Открыть страницу источника', 'DEV': 'Текущие параметры: данные для подбора', 'DEV_NOTE': 'Метрики ниже пересчитываются при смене параметров. Тест не используется для подбора.', 'TEST': 'Зафиксированный результат на отложенных вопросах', 'TEST_NOTE': 'Таблица считана из сохраненного эксперимента. Настройки слева ее не меняают.', 'RUN_NOTE': 'Полный перебор 96 конфигураций запускается в EDA.ipynb. Здесь проверяется один вариант.', 'AUDIT': 'Аудит всех страниц', 'DUP': 'Найти близкие чанки', 'NO_DUP': 'Пар выше порога 0.85 не найдено.', 'DUP_NOTE': 'Совпадение не означает ошибку. Определения повторяются в приложениях, а соседние чанки перекрываются.', 'NO_API': 'Генерация новых вопросов запускается отдельной командой. Ключ API не нужен для этой панели.', 'BANK': 'Фиксированные вопросы и связь с источником', 'ERROR': 'Не удалось открыть корпус. Проверьте PDF в checkpoint_02/data/raw и их SHA-256.'}
L=LABELS
st.set_page_config(page_title='Research Assistant | Checkpoint 02',page_icon=None,layout='wide')
st.markdown("""<style>
.block-container {max-width:1380px;padding-top:2.2rem;}
[data-testid="stSidebar"] {background:#101d2c;}
[data-testid="stSidebar"] * {color:#e7eef4;}
[data-testid="stMetric"] {border:1px solid #dbe3e9;border-radius:12px;padding:16px 22px;}
h1 {letter-spacing:-.035em;}
</style>""",unsafe_allow_html=True)
st.caption(L['SUBTITLE'])
st.title(L['TITLE']); st.write(L['DESCRIPTION'])
@st.cache_data
def inputs():
    docs,audit=load_corpus()
    return docs,pd.DataFrame(audit),load_questions(docs)
try:
    docs,audit,questions=inputs()
except (OSError,RuntimeError,ValueError) as exc:
    st.error(L['ERROR']); st.code(str(exc)); st.stop()
with st.sidebar:
    st.markdown('## RA / Research Assistant')
    st.caption('BM25 LAB / 2026')
    st.divider(); st.subheader(L['SETTINGS'])
    strategy=st.selectbox(L['SPLITTER'],['token','sentence'],index=0,key='strategy')
    size=st.select_slider(L['SIZE'],options=[128,256,512],value=256,key='size')
    overlap=st.slider(L['OVERLAP'],0,64,32,step=8,key='overlap')
    mode=st.selectbox(L['MODE'],['lower','stem'],index=0,key='mode')
    k1=st.slider('BM25 k1',0.2,2.5,1.2,step=.1,key='k1')
    b=st.slider('BM25 b',0.0,1.0,.75,step=.05,key='b')
    st.divider(); st.caption('Corpus SHA-256'); st.code(corpus_fingerprint(docs)[:16])
config=ExperimentConfig(strategy,size,overlap,mode,k1,b)
chunks=split_corpus(docs,config.split)
cols=st.columns(4)
for col,label,value in zip(cols,[L['DOCS'],L['PAGES'],L['SEARCH_PAGES'],L['CHUNKS']],[len(docs),len(audit),int(audit.included.sum()),len(chunks)]):
    col.metric(label,value)
st.caption(L['WARN'])
tabs=st.tabs([L[k] for k in ['CLOUD','SEARCH','EXPERIMENT','QUALITY','QUESTIONS']])
with tabs[0]:
    controls=st.columns([1,1,1])
    scope=controls[0].selectbox(L['SCOPE'],[L['CORPUS'],L['DOCUMENT'],L['CHUNK']],key='scope')
    stop_label=controls[1].selectbox(L['STOP'],[L['STOP_BASIC'],L['STOP_NONE'],L['STOP_DOMAIN']],key='stop')
    stop_mode={L['STOP_BASIC']:'basic',L['STOP_NONE']:'none',L['STOP_DOMAIN']:'domain'}[stop_label]
    text='\n'.join(d.text for d in docs)
    if scope!=L['CORPUS']:
        docid=controls[2].selectbox(L['CHOOSE_DOC'],[d.document_id for d in docs],key='document')
        text=next(d.text for d in docs if d.document_id==docid)
        if scope==L['CHUNK']:
            candidates=[c for c in chunks if c.document_id==docid]
            chosen=st.selectbox(L['CHUNK'],range(len(candidates)),format_func=lambda i:f"{i+1} / {candidates[i].chunk_id[:8]} / PDF {candidates[i].pages}",key='chunk')
            text=candidates[chosen].text
    left,right=st.columns([2.3,1])
    left.image(word_cloud(text,stop_mode).to_array(),use_container_width=True)
    counts=pd.DataFrame(word_frequencies(text,stop_mode).most_common(),columns=['word','count'])
    right.subheader(L['TOPWORDS']); right.dataframe(counts.head(15),hide_index=True,use_container_width=True)
    right.download_button(L['DOWNLOAD'],counts.to_csv(index=False).encode('utf-8-sig'),'word_frequencies.csv','text/csv')
    st.caption(L['STOP_NOTE'])
    if scope==L['CHUNK']:
        with st.expander(L['FRAGMENT'],expanded=True): st.text(text)
with tabs[1]:
    query=st.text_input(L['QUERY'],L['DEFAULT_QUERY'],key='query')
    top_k=st.select_slider('TOP-k',[1,5,10],value=5,key='top_k')
    index=BM25Index(chunks,k1,b,mode)
    found=index.search(query,top_k)
    if not found: st.info(L['EMPTY'])
    for rank,(chunk,score) in enumerate(found,1):
        with st.expander(f'{rank}. {chunk.document_id} / PDF {chunk.pages} / BM25 {score:.3f}',expanded=rank==1):
            st.text(chunk.text)
            doc=next(d for d in docs if d.document_id==chunk.document_id)
            st.link_button(L['OPEN'],doc.source_url+f'#page={chunk.pages[0]}')
with tabs[2]:
    st.subheader(L['DEV']); st.caption(L['DEV_NOTE'])
    @st.cache_data
    def evaluate_current(config,fingerprint):
        rows,_=run_config(docs,[q for q in questions if q['split']=='dev'],config)
        return aggregate(rows)
    metrics=evaluate_current(config,corpus_fingerprint(docs))
    st.dataframe(metrics[['k','hit','precision','recall','mrr','ndcg','addressable','questions','context_tokens']],hide_index=True)
    st.caption(L['RUN_NOTE'])
    st.subheader(L['TEST']); st.caption(L['TEST_NOTE'])
    test_path=ROOT/'results/test_summary.csv'
    if test_path.exists():
        table=pd.read_csv(test_path)
        st.dataframe(table[table.variant=='original'],hide_index=True)
with tabs[3]:
    st.subheader(L['AUDIT']); st.dataframe(audit,hide_index=True)
    st.caption(L['DUP_NOTE'])
    if st.button(L['DUP'],key='duplicates'):
        pairs=duplicate_candidates([(c.chunk_id,c.text) for c in chunks])
        if pairs: st.dataframe(pd.DataFrame(pairs),hide_index=True)
        else: st.info(L['NO_DUP'])
with tabs[4]:
    st.subheader(L['BANK']); st.caption(L['NO_API'])
    st.dataframe(pd.DataFrame(questions)[['question_id','question','document_id','page','section','split','human_review']],hide_index=True)

'''Rebuild the report from executed experiments and pinned source documents.'''
from pathlib import Path
from dataclasses import asdict
import argparse, json, platform
import pandas as pd
from .corpus import ROOT, load_corpus, load_sources, corpus_fingerprint, write_json
from .splitters import ChunkConfig, split_corpus
from .evaluation import load_questions, run_study
from .audit import duplicate_candidates, word_frequencies
from .metrics import paired_group_interval
from .plots import (plot_page_lengths, plot_term_frequencies, plot_development,
                    plot_test, plot_augmentation, plot_word_cloud, save_figure)


def table(frame):
    labels=list(frame.columns)
    lines=['| '+' | '.join(labels)+' |','| '+' | '.join(['---']*len(labels))+' |']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def practical_pdf():
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from matplotlib.font_manager import findfont, FontProperties
    pdfmetrics.registerFont(TTFont('Readable',findfont(FontProperties(family='DejaVu Sans'))))
    styles=getSampleStyleSheet()
    for name in ['Title','Heading2','BodyText']:
        styles[name].fontName='Readable'
    styles['Title'].fontSize=18; styles['Title'].leading=23
    styles['BodyText'].fontSize=10.5; styles['BodyText'].leading=15
    paragraphs=[
        ('Title','Research Assistant: практическая ценность'),
        ('BodyText','Чекпойнт 02 / 10 ноября 2026'),
        ('Heading2','Кому нужен поиск по документам'),
        ('BodyText','Исследователь или специалист по качеству задает вопрос к набору документов и получает фрагменты с номерами страниц. Так проще найти определение, проверить требование и вернутся к полному контексту. Поиск не заменяет чтение стандарта и не подтверждает соответствие организации его требованиям.'),
        ('Heading2','Что проверено на этом этапе'),
        ('BodyText','Прототип обрабатывает три цифровых PDF по менеджменту качества: 105 страниц, из которых 90 включены в поиск. Отдельные модули извлекают текст, режут его на чанки, строят индекс BM25 и считают метрики. Панель Streamlit показывает облака слов, источники, найденные фрагменты и влияние параметров. Длйа нового пользовательского корпуса потребуются новый реестр источников и разметка.'),
        ('Heading2','Как оценивать пользу'),
        ('BodyText','Заказчик может собрать реальные вопросы специалистов и отметить фрагменты с ответами. Основные показатели: доля вопросов с доказательством в TOP-1, TOP-5 и TOP-10, время поиска, достаточность контекста и корректность ссылки на страницу. Синтезированные вопросы этого чекпойнта подходят для отладки, но не заменяют независимой проверки человеком.'),
        ('Heading2','Ограничения и риски'),
        ('BodyText','Текстовый слой PDF может разрывать слова или терять структуру таблиц. Поиск по словам пропускает часть синонимов; длинные чанки увеличивают нагрузку при чтении. Нужно проверять редакции стандартов и права на передачу копий. Конфиденциальные документы нельзя отправлять внешнему генератору без согласия владельца. Сканы, таблицы как изображения и автоматическая проверка актуальности пока не поддержаны.'),
        ('Heading2','Следующая итерация'),
        ('BodyText','Сначала нужны независимые вопросы и полная проверка доказательств. Затем к тому же корпусу и тесту можно добавить векторный поиск и повторное ранжирование. Сбор библиографии, граф публикаций и ответ по нескольким источникам остаются следующими частями архитектуры. Результаты на трэ ГОСТах не переносятся автоматически на научные статьи.')]
    story=[]
    for style,value in paragraphs:
        story.extend([Paragraph(value,styles[style]),Spacer(1,7)])
    SimpleDocTemplate(str(ROOT/'practical_value.pdf'),pagesize=A4,leftMargin=46,rightMargin=46,topMargin=38,bottomMargin=35).build(story)


def build_report():
    docs,audit_rows=load_corpus()
    questions=load_questions(docs)
    audit=pd.DataFrame(audit_rows)
    dev=pd.read_csv(ROOT/'results/development.csv')
    test=pd.read_csv(ROOT/'results/test_summary.csv')
    details=pd.read_csv(ROOT/'results/test_per_query.csv')
    selection=json.loads((ROOT/'results/selection.json').read_text())
    chunks=split_corpus(docs,ChunkConfig())
    page_pairs=duplicate_candidates([(f'{d.document_id}:p{p.number}',p.text) for d in docs for p in d.pages])
    chunk_pairs=duplicate_candidates([(c.chunk_id,c.text) for c in chunks])
    audit.to_csv(ROOT/'results/page_audit.csv',index=False)
    write_json(ROOT/'results/duplicate_candidates.json',dict(pages=page_pairs,chunks=chunk_pairs,threshold=.85))
    full='\n'.join(d.text for d in docs)
    pd.DataFrame(word_frequencies(full).most_common(),columns=['word','count']).to_csv(ROOT/'results/word_frequencies.csv',index=False)
    figure_functions={
        'page_lengths':lambda:plot_page_lengths(audit),
        'word_frequencies':lambda:plot_term_frequencies(full),
        'corpus_cloud':lambda:plot_word_cloud(full,'Облако слов: весь корпус'),
        'chunk_cloud':lambda:plot_word_cloud(chunks[10].text,'Облако слов: один чанк'),
        'development':lambda:plot_development(dev),
        'test':lambda:plot_test(test),
        'augmentation':lambda:plot_augmentation(test)}
    for name,func in figure_functions.items():save_figure(func(),ROOT/'figures'/f'{name}.png')
    selected=details[(details.system=='selected')&(details.variant=='original')&(details.k==5)].sort_values('question_id')
    baseline=details[(details.system=='baseline')&(details.variant=='original')&(details.k==5)].sort_values('question_id')
    interval=paired_group_interval(selected.hit.tolist(),baseline.hit.tolist(),selected.group.tolist())
    records=[]
    for d in docs:
        pages=audit[audit.document_id==d.document_id]
        records.append(dict(document=d.document_id,pages=len(pages),search_pages=len(d.pages),tokens=int(pages[pages.included].tokens.sum())))
    corpus_table=table(pd.DataFrame(records))
    metric_table=table(test[test.variant=='original'][['system','k','hit','precision','recall','mrr','ndcg','questions']])
    variants=table(test[test.k==5][['system','variant','hit','mrr','questions']])
    doc_scores=table(details[(details.variant=='original')&(details.k==5)].groupby(['system','document_id'])[['hit','mrr']].mean().reset_index())
    errors=baseline[baseline.hit==0].question_id.tolist()
    error_examples='\n'.join('- '+q['question_id']+': '+q['question']+' (PDF '+str(q['page'])+', '+q['document_id']+')' for q in questions if q['question_id'] in errors[:3])
    report=f'''# Чекпойнт 02. Анализ ГОСТов и поиск BM25

10 ноября 2026. Три документа, 105 страниц PDF, 90 страниц в поиске. Основной вывод: настройка на малой выборке не гарантировала улучшение на отложенных вопросах.

## 1. Данные и происхождение

{corpus_table}

Копии закреплены в [data/sources.json](data/sources.json): адрес, SHA-256 и границы страниц. Титул, служебные сведения и задние обложки не включены в поиск. Все 105 страниц остаются в аудите. Полные PDF не публикуются; загрузчик проверяет хеш перед расчетами.

Источники: [9000-2015](https://files.stroyinf.ru/Data2/1/4293759/4293759339.pdf), [9001-2015](https://files.stroyinf.ru/Data2/1/4293759/4293759338.pdf), [10013-2024](https://files.stroyinf.ru/Data/830/83016.pdf).

Это закрепленные редакции для эксперимента. Их действуущий нормативный статус на дату применения не проверялся.

## 2. Качество текста и дубликаты

![Page lengths](figures/page_lengths.png)

В корпусе есть определения, требования и рекомендации. Короткие страницы с диаграммами не считаются автоматически мусором. В текстовом слое встречаются переносы, латиница среди русских терминов и разрывы слов пробелами. Очередность таблиц и рисунков не восстанавливается.

Очистка объединяет переносы и нормализует пробелы; исходный PDF остается неизменным. Пропуски, символ замены, доля латиницы и длина каждой страницы сохранены в [page_audit.csv](results/page_audit.csv).

Проверка SHA-256 находит точные повторы нормализованного текста. Нормализованное сходство Левенштейна при пороге 0.85 дает {len(page_pairs)} пар страниц и {len(chunk_pairs)} пар базовых чанков. Пары сохранены в [duplicate_candidates.json](results/duplicate_candidates.json). Они не удалялись автоматически: повтор определения в приложении может быть полезным, а соседние чанки перекрываются по настройке.

Векторные представления рассмотрены как способ поиска смысловых повторов, но не вычислялись в этом запуске. Векторная близость сама по себе не доказывает дубль.

## 3. Словарь корпуса

![Corpus word cloud](figures/corpus_cloud.png)
![Term frequencies](figures/word_frequencies.png)
![Chunk word cloud](figures/chunk_cloud.png)

Облака построены по словоформам, без лемматизации. Базовый стоп-список убирает служебные слова; доменные слова сохранены. В панели можно сравнить режимы без стоп-слов, с базовым и расширенным списком. Это меняет только визуализацию, не индекс BM25. Частота не равна важности: IDF снижает вес слов, которые есть во многих чанках.

## 4. Вопросы и разделение

Ассистент составил по 20 вопросов на документ и по одной переформулировке: 60 основных вопросов и 60 парных вариантов. У каждого есть страница, раздел, символьный интервал доказательства и его SHA-256. Все записи помечены `human_review=false`. Происхождение сохранено в [generation.json](data/generation.json).

По страничным группам зафиксированы 32 вопроса для подбора и 28 для теста. Одна страница не попадает в обе части. Переформулировки наследуют группу исходного вопроса. Это внутридокументный синтетический тест, а не независимые вопросы реальных пользователей. Повторы определений на других страницах могут сохранять зависимость.

Вопросы не регенерируются при смене сплиттера. Релевантным считается чанк, который целиком покрывает хотя бы один интервал доказательства. Вопрос без такого чанка получает нуль, но не исчезает из знаменателя. Это строгое координатное правило; семантически подходящие не размеченные фрагменты могут считаться ошибкой. Необходима проверка полноты разметки.

Адаптер `questions.py` умеет обработать каждый документ или каждый базовый чанк через внешний API. Живой вызов API не выполнялся. Сравнение двух стратегий генерации не проведено; в этом корпусе использован документный режим.

## 5. Разбиение и протокол

Два сплиттера: окно слов и окно с предпочтением границ предложений. Размер означает максимум слов по одному регулярному выражению, не BPE-токены модели. Сетка: 128/256/512 слов, перекрытие 0/32, нижний регистр или русский Snowball-стеммер, k1=1.2/1.8, b=0.35/0.75. Всего 96 конфигураций.

BM25 использует положительный IDF: `log(1 + (N-df+0.5)/(df+0.5))`. Формула соответствует варианту из документации [Lucene](https://lucene.apache.org/core/9_12_1/core/org/apache/lucene/search/similarities/BM25Similarity.html). Пустой запрос и неизвестные слова не возвращают случайные нулевые совпадения. Равные оценки разрешаются по стабильному ID.

![Development experiments](figures/development.png)

Базовый вариант: `token / 256 / overlap=32 / lower / k1=1.2 / b=0.75`.
По dev выбран: `{selection['selected']}`.
Правило выбора: максимальный Hit@5, затем MRR@5, затем меньше слов контекста. Тест открывается только после фиксации выбора.

## 6. Фактические метрики

{metric_table}

![Held-out retrieval](figures/test.png)

На 28 тестовых вопросах оба варианта находят доказательство в TOP-5 в 16 случаях, в TOP-10 в 19. В TOP-1 базовый вариант попадает 10 раз, выбранный на dev вариант 8 раз. Значит, выигрыш на подборе не подтвердился на основных тестовых формулировках. Менять выбор задним числом по тесту нельзя.

Парный групповой бутстреп по страницам для разности Hit@5 (selected - baseline): 95% интервал [{interval[0]:.3f}; {interval[1]:.3f}]. На такой малой выборке это разведочный интервал, не доказательство равенства систем.

Hit@k показывает наличие хотя бы одного доказательства. Recall@k считает долю всех размеченных релевантных чанков; при перекрытии их может быть несколько. Precision@k делится на к, даже когда выдача короче. MRR учитывает ранг первого попадания, nDCG оценивает весь порядок. Основной критерий между разбиениями: Hit@k вместе с числом слов контекста, поскольку число релевантных чанков зависит от разбиения.

### Срез по документам

{doc_scores}

### Примеры пропусков базового поиска в TOP-5

{error_examples}

Полные ранги и адреса кандидатов сохранены в [predictions.jsonl](results/predictions.jsonl). Перед следующей итерацией нужно прочитать кандидатов и разделить пропуски словарного поиска, ошибки границ и неполноту разметки.

## 7. Аугментация запросов

{variants}

![Query variants](figures/augmentation.png)

Изменение регистра и пунктуации не меняет результат: нормализация работает одинаково в запросе и корпусе. Удаление одной буквы в длинном слове проверяет чувствительность к опечатке. Переформулировки созданы вместе с основными вопросами. Рост на них не доказывает улучшение на реальных пользователях. Не нужно аугментировать индекс дублями документов: это меняет IDF и может исказить сравнение.

## 8. Прототип и воспроизведение

`streamlit run checkpoint_02/streamlit_app.py`

Панель строит облака по корпусу, документу и выбранному чанку; показывает частоты, исходный текст, поиск и пересчет dev-метрик по ползункам. Таблица теста зафиксирована и не меняется от ползунков. Экран работает с настоящими данными и теми же модулями, что ноутбук. Детали запуска: [README.md](README.md).

Проверки: модульные тесты формул метрик, границ чанков, пустых запросов, хешей и разметки; Streamlit AppTest проверяет смену параметров и уровень чанка. Тест API использует подставной ответ. Факты о прогонах записаны в `results/validation.json` после проверки. Запуск в самом сервисе Google Colab не проводился; ноутбук проверен полным локальным выполнением.

## 9. Что делать дальше

Сохранить этот BM25 как точку отсчета. Приоритет: ручная проверка 60 вопросов, дополнение альтернативных доказательств и реальные вопросы специалистов. После этого проверить векторный поиск и реранкер на тех же данных. Веб-прототип не включает загрузку произвольных корпусов, граф цитирования и генерацию ответа. Этот этап реализует обработку PDF, поиск и оценку; остальные компоненты архитектуры не объявляются готовыми.

Практическая ценность и риски: [practical_value.pdf](practical_value.pdf).
'''
    (ROOT/'report.md').write_text(report,encoding='utf-8')
    write_json(ROOT/'results/environment.json',dict(python=platform.python_version(),platform=platform.platform(),corpus_sha256=corpus_fingerprint(docs),question_count=len(questions),chunks=len(chunks)))
    practical_pdf()
    return audit,test


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');args=parser.parse_args()
    if args.run:
        docs,_=load_corpus();run_study(docs,load_questions(docs))
    build_report()

if __name__=='__main__': main()

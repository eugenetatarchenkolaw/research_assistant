# Чекпойнт 02: корпус ГОСТов и поиск BM25

Выполненный анализ: три документа, 105 страниц PDF, 90 страниц в поиске. 60 синтезированных вопросов, 96 конфигураций.

[EDA.ipynb](EDA.ipynb) | [report.md](report.md) | [practical_value.pdf](practical_value.pdf) | [Открыть в Colab](https://colab.research.google.com/github/eugenetatarchenkolaw/research_assistant/blob/main/checkpoint_02/EDA.ipynb)

## Запуск в Google Colab

Откройте ноутбук по ссылке выше и выполните все ячейки по порядку. Первая ячейка скачивает репозиторий, если его нет в окружении, и устанавливает зависимости. Ключ API не нужен: вопросы уже сохранены. В ноутбуке указана ветка `main`. При работе с веткой замените `REPO_REF` на `checkpoint-02-quality-gost`.

Ноутбук полностью выполнен локально и в чистом окружении GitHub Actions. Сам сервис Google Colab не проверялся. В [протоколе проверки](results/validation.json) сохранены фактические результаты: 39 тестов, ноль ошибок и отказов, включая взаимодействия со Streamlit.

## Локальный запуск

```bash
git clone https://github.com/eugenetatarchenkolaw/research_assistant.git
cd research_assistant
python -m pip install -r checkpoint_02/requirements.txt
python -c "from checkpoint_02.corpus import ensure_pdfs; ensure_pdfs()"
python -m pytest checkpoint_02/tests -q
streamlit run checkpoint_02/streamlit_app.py
```

Полный пересчёт:

```bash
python -m checkpoint_02.build_report --run
```

## Что в папке

| Файл | Назначение |
|---|---|
| `corpus.py` | Загрузка PDF, проверка SHA-256, текст и страницы |
| `splitters.py` | Разбиение по словам или границам предложений; размер и перекрытие |
| `audit.py` | Частоты, точные дубли, сходство Левенштейна |
| `retrieval.py` | BM25, нижний регистр или русский стеммер |
| `metrics.py` | Hit, Precision, Recall, MRR, nDCG @ 1/5/10 |
| `evaluation.py` | Проекция доказательств на чанки, подбор на dev, отдельный тест |
| `questions.py` | Адаптер генерации вопросов по документу или чанку |
| `plots.py`, `build_report.py` | Графики, таблицы, отчёт и PDF о практической ценности |
| `streamlit_app.py` | Облака слов, поиск, параметры и метрики в браузере |
| `data/` | Источники, вопросы и происхождение разметки |
| `results/`, `figures/` | Фактические результаты выполненного анализа |
| `tests/` | Тесты модулей и элементов Streamlit |

## Данные и границы выводов

PDF не публикуются в Git. Загрузчик берёт три закреплённые копии по адресам из `data/sources.json` и проверяет байты. Если источник недоступен, поместите тот же PDF в `checkpoint_02/data/raw/`. Другие редакции не подходят к сохранённым координатам доказательств. Сканы и OCR не поддержаны.

Вопросы подготовил ассистент по фрагментам документов. У всех записей `human_review=false`. Это пилотная оценка, не независимый эталон. Не включайте вопросы или ответы в индекс.

Дополнительный запуск генерации требует собственного ключа и явно выбранной модели:

```bash
export LLM_API_KEY='...'
python -m checkpoint_02.questions --scope document --n 5 \
  --model YOUR_MODEL --base-url https://YOUR_PROVIDER/v1 \
  --output checkpoint_02/results/new_questions.jsonl
```

Замените `document` на `chunk` для генерации по базовым чанкам. Выход требует проверки и разметки; он не заменяет зафиксированный тест автоматически. Сетевой вызов API в этой работе не выполнялся. Адаптер проверен тестами с подставным ответом.

## Продолжение архитектуры

Модули следуют цепочке PDF → Document/Page → Chunk → BM25 → метрики. Координаты доказательств независимы от разбиения. Позже `BM25Index` можно дополнить векторным поиском и реранкером, не перенося логику из ноутбука. Сбор библиографии, граф цитирования и генерация конечного ответа пока не реализованы.

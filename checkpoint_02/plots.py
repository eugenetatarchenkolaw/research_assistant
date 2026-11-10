"""Reusable figures; functions return objects and leave saving to the caller."""
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont, FontProperties
import pandas as pd
from wordcloud import WordCloud

from .audit import word_frequencies


def word_cloud(text: str, stop_mode: str='basic', max_words: int=80):
    frequencies=word_frequencies(text,stop_mode)
    if not frequencies:
        raise ValueError('No words left after filtering')
    return WordCloud(font_path=findfont(FontProperties(family='DejaVu Sans')),width=1000,height=480,
        random_state=42,max_words=max_words,collocations=False,prefer_horizontal=.95).generate_from_frequencies(frequencies)


def plot_word_cloud(text: str, title: str, stop_mode: str='basic'):
    fig,ax=plt.subplots(figsize=(10,5))
    ax.imshow(word_cloud(text,stop_mode).to_array(),interpolation='bilinear')
    ax.axis('off'); ax.set_title(title,pad=12)
    fig.tight_layout(); return fig


def plot_page_lengths(audit: pd.DataFrame):
    fig,ax=plt.subplots(figsize=(10,4.8))
    for doc,part in audit.groupby('document_id'):
        ax.hist(part.tokens,bins=15,histtype='step',linewidth=1.8,label=doc)
    ax.set(xlabel='Words per PDF page',ylabel='Pages',title='Page length: covers, prose and diagrams')
    ax.legend(fontsize=9); ax.grid(axis='y',alpha=.25)
    fig.tight_layout(); return fig


def plot_term_frequencies(text: str):
    terms=word_frequencies(text,'basic').most_common(15)[::-1]
    fig,ax=plt.subplots(figsize=(10,5.5))
    ax.barh([x[0] for x in terms],[x[1] for x in terms])
    ax.set(xlabel='Occurrences (word forms, not lemmas)',title='Most frequent words: 90 searchable pages')
    fig.tight_layout(); return fig


def plot_development(development: pd.DataFrame):
    # Fixed retrieval settings isolate chunking and preprocessing.
    frame=development[(development.k==5)&(development.k1==1.2)&(development.b==.75)&(development.overlap==32)]
    fig,ax=plt.subplots(figsize=(10,5))
    for (strategy,mode),part in frame.groupby(['strategy','mode']):
        part=part.sort_values('size')
        ax.plot(part['size'],part['hit'],marker='o',label=f'{strategy} / {mode}')
    ax.set(xticks=[128,256,512],ylim=(0,1.05),xlabel='Word tokens per chunk (maximum)',
           ylabel='Hit@5',title='Development only: fixed overlap=32, k1=1.2, b=0.75')
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); return fig


def plot_test(test: pd.DataFrame):
    frame=test[(test.variant=='original')]
    fig,ax=plt.subplots(figsize=(9,4.7))
    for system,part in frame.groupby('system'):
        part=part.sort_values('k')
        ax.plot(part['k'],part['hit'],marker='o',label=system)
    ax.set(xticks=[1,5,10],ylim=(0,1.05),xlabel='Retrieved chunks (k)',ylabel='Hit@k',
           title=f"Held-out questions: n={int(frame.questions.iloc[0])}")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); return fig


def plot_augmentation(test: pd.DataFrame):
    frame=test[(test.k==5)].pivot(index='variant',columns='system',values='hit')
    fig,ax=plt.subplots(figsize=(9,4.7))
    frame.plot.bar(ax=ax,rot=0)
    ax.set(ylim=(0,1.05),ylabel='Hit@5',xlabel='Query variant',title='Paired query changes on the same held-out source pages')
    ax.grid(axis='y',alpha=.25); fig.tight_layout(); return fig


def save_figure(fig, path: Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=140,bbox_inches='tight')
    plt.close(fig)

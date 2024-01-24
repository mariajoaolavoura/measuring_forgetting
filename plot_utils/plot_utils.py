from eval_implicit import EvaluateAndStore
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')


def lineplot_recallxholdout(df,
    title='Recall@20 for checkpoint models across Holdouts - model - data',
    filepath='images/lineplots/..'):
    
    plt.figure(figsize=(25,10))
    sns.lineplot(data=df.T, palette='tab20')
    x_t = np.arange(0,20)
    plt.xticks(x_t, labels=[str(i+1) for i in x_t])
    plt.xlim(0, 19)
    plt.xlabel('Holdout')
    plt.ylabel('Recall@20')
    plt.legend(bbox_to_anchor=(1.0,1), loc="upper left", labels=[f'model: T{model+1}' for model in range( df.shape[0] )])
    plt.title(title)
    if filepath:
        plt.savefig(filepath);

def recall_heatmap(df,
    round_point=2,
    title='Recall@20 for checkpoint models across Holdouts - model - data',
    filepath='images/heatmaps/..'):
    plt.figure(figsize=(15, 10))
    x_t = np.arange(0, df.shape[0])
    labels=[str(i+1) for i in x_t]
    sns.heatmap(df, vmin=0, vmax=df.max().max(), annot=True, fmt=f'0.{round_point}f', linewidths=.1, cmap='Spectral_r', xticklabels=labels, yticklabels=labels)
    plt.ylabel('model')
    plt.xlabel('holdout')
    plt.title(title)
    if filepath:
        plt.savefig(filepath);
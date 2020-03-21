import json
import locale
import os
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

locale.setlocale(locale.LC_ALL, 'pt_pt.UTF-8')


font_dirs = ['/home/johnme/blockchain-notebook/fonts/']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
font_list = font_manager.createFontList(font_files)
font_manager.fontManager.ttflist.extend(font_list)

colors = {'blue': '#30a2da', 'red': '#fc4f30',
          'yellow': '#e5ae38', 'green': '#6d904f', 'gray': '#8b8b8b'}


font = 'Clear Sans'

plt.rcParams["figure.figsize"] = [8.5, 4.5]

plt.rcParams['font.family'] = font
plt.rcParams['font.sans-serif'] = font

plt.style.use('fivethirtyeight')

plt.rcParams['axes.linewidth'] = 1

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

plt.rcParams['grid.linestyle'] = '--'

plt.rcParams['ytick.color'] = '#333333'
plt.rcParams['xtick.color'] = '#333333'

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.rcParams['axes.edgecolor'] = '#333333'

plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

plt.rcParams['xtick.major.size'] = 12
plt.rcParams['xtick.minor.size'] = 8
plt.rcParams['ytick.major.size'] = 12
plt.rcParams['ytick.minor.size'] = 8

plt.rcParams['xtick.major.pad'] = 15
plt.rcParams['ytick.major.pad'] = 15

plt.rcParams['axes.grid.which'] = 'major'

plt.rcParams['font.size'] = 20

plt.rcParams['lines.linewidth'] = 4

plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

pd.set_option('precision', 8)


def plot_area(data, ax=None):
    if not ax:
        _, ax = plt.subplots(nrows=1)

    plt.stackplot(
        data.index, [data.deaths.tolist(), (data.recovered - data.deaths).tolist(), (data.confirmed - data.recovered - data.deaths).tolist(), ], labels=['deaths', 'recovered', 'active'],
        colors=[colors['red'], colors['green'], colors['blue']])

    date_form = mdates.DateFormatter("%d %b")

    ax.xaxis.set_major_formatter(date_form)

    plt.legend(loc='upper left')
    plt.xticks(rotation=30, ha="center")

    return ax


def plot_line(x, y, xlabel=None, ylabel=None, xlog=False, ylog=False,
              label=None, ax=None, color=None, marker='o', markerfacecolor="None",
              markersize=8, linewidth=1.5):
    if not ax:
        _, ax = plt.subplots(nrows=1)

    plt.plot(x, y, color=color, label=label, marker=marker,
             markerfacecolor=markerfacecolor, markersize=markersize, linewidth=linewidth)

    date_form = mdates.DateFormatter("%d %b")

    ax.xaxis.set_major_formatter(date_form)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.grid(which='minor', axis='both', linestyle=':')

    if label:
        plt.legend()
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    plt.xticks(rotation=30, ha="center")
    return ax


def get_data(df_confirmed, df_recovered, df_deaths):
    dfs = list()
    df_confirmed = df_confirmed.drop(
        columns=['Province/State', 'Lat', 'Long']).groupby('Country/Region').sum()
    df_recovered = df_recovered.drop(
        columns=['Province/State', 'Lat', 'Long']).groupby('Country/Region').sum()
    df_deaths = df_deaths.drop(
        columns=['Province/State', 'Lat', 'Long']).groupby('Country/Region').sum()

    for country in df_confirmed.index.unique():
        confirmed = df_confirmed.loc[country]
        recovered = df_recovered.loc[country]
        deaths = df_deaths.loc[country]
        df = pd.DataFrame(
            {'confirmed': confirmed, 'recovered': recovered, 'deaths': deaths})
        df['country'] = country
        dfs.append(df)
    df = pd.concat(dfs)
    df.index = pd.to_datetime(df.index)
    df['actives'] = df['confirmed'] - df['recovered'] - df['deaths']
    return df


def load_files(url, filenames):
    df_confirmed = pd.read_csv(url+filenames['confirmed'], sep=',')
    df_deaths = pd.read_csv(url+filenames['deaths'], sep=',')
    df_recovered = pd.read_csv(url+filenames['recovered'], sep=',')
    df = get_data(df_confirmed, df_recovered, df_deaths)
    return df


def plot_confirmed_cases(data):
    ax = sns.lineplot(x="days_since_first_infection", y="confirmed",
                      hue="Países", data=data, legend="full")

    ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))

    ax.set(xlabel='Número de dias desde o primeiro caso reportado',
           ylabel='Número oficial de casos (log)')

    plt.ylim((1, 100000))

    plt.yscale('log')

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)

    for axis in [ax.yaxis]:
        formatter = ticker.ScalarFormatter()
        formatter.set_scientific(False)
        axis.set_major_formatter(formatter)

    plt.text(63, .09, 'http://johnnatan.me',
             {'color': colors['red'], 'fontsize': 15})
    plt.text(63, .04, f"Atualizado em {str(datetime.now().date())}", {
             'color': colors['green'], 'fontsize': 15})

    plt.savefig('./confirmed_cases.pdf', bbox_inches='tight')


def persist_dataset(df):
    data.index.name = 'date'
    data.index = data.index.strftime('%d %b')
    cols = ['confirmed', 'recovered', 'deaths', 'actives','days_since_first_infection']
    out = dict()
    for country in data['Países'].unique():
        out[country] = data.query('Países == @country')[cols].reset_index().to_dict(orient='list')

    with open('../data/data.json', 'w', encoding='utf8') as outfile:
        json.dump(out, outfile, ensure_ascii=False)


if __name__ == "__main__":
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
    filenames = {'confirmed': 'time_series_19-covid-Confirmed.csv',
                 'deaths': 'time_series_19-covid-Deaths.csv',
                 'recovered': 'time_series_19-covid-Recovered.csv'}

    df = load_files(url=url, filenames=filenames)
    countries = ['Brazil', 'Italy', 'Germany',
                 'France', 'Spain', 'US', 'Portugal']
    #data = df.query("country in @countries").query('confirmed > 0')
    data = df.query('confirmed > 0')
    data['days_since_first_infection'] = data.groupby(
        "country").confirmed.rank(method='first', ascending=True)

    data['Países'] = data['country'].replace(
        {'Brazil': 'Brasil', 'France': 'França', 'Germany': 'Alemanha',
         'Italy': 'Itália', 'Spain': 'Espanha', 'US': 'EUA'}
    )
    # plot_confirmed_cases(data)

    persist_dataset(df=data)


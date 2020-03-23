import json
import locale
import os
import time
import traceback
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')


font_dirs = ['/home/johnme/blockchain-notebook/fonts/']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
font_list = font_manager.createFontList(font_files)
font_manager.fontManager.ttflist.extend(font_list)

colors = {'blue': '#30a2da', 'red': '#fc4f30',
          'yellow': '#e5ae38', 'green': '#6d904f', 'gray': '#8b8b8b'}

url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
url_daily = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv"
filenames = {'confirmed': 'time_series_19-covid-Confirmed.csv',
             'deaths': 'time_series_19-covid-Deaths.csv',
             'recovered': 'time_series_19-covid-Recovered.csv'}
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
countries_to_pt = {'Brazil': 'Brasil', 'France': 'França', 'Germany': 'Alemanha',
                   'Italy': 'Itália', 'Spain': 'Espanha', 'US': 'EUA', 'Switzerland': 'Suíça',
                   'Netherlands': 'Holanda', 'Iran': 'Irã', 'Korea, South': 'Coréia do Sul',
                   'United Kingdom': 'Reino Unido', 'Belgium': 'Bélgica', 'Austria': 'Áustria',
                   'Norway': 'Noruega', 'Sweden': 'Suécia', 'Denmark': 'Dinamarca', 'Canada': 'Canadá',
                   'Malaysia': 'Malásia', 'Australia': 'Austrália', 'Japan': 'Japão', 'Ireland': 'Irland',
                   'Turkey': 'Turquia', 'Luxembourg': 'Luxemburgo', 'Pakistan': 'Paquistão', 'Czechia': 'Rep. Tcheca',
                   'Cruise Ship': 'Cruzeiro D. Princess', 'Ecuador': 'Equador'}


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

    return df


def load_files(url, filenames):
    df_confirmed = pd.read_csv(url+filenames['confirmed'], sep=',')
    df_deaths = pd.read_csv(url+filenames['deaths'], sep=',')
    df_recovered = pd.read_csv(url+filenames['recovered'], sep=',')
    df = get_data(df_confirmed, df_recovered, df_deaths)

    df.rename(columns={'country': 'countries'}, inplace=True)
    #df['countries'] = df['countries'].replace(countries_to_pt)
    #df.index = df.index.strftime('%d %b')
    df.index.name = 'date'
    df.index = pd.to_datetime(df.index)
    return df.reset_index()


def plot_confirmed_cases(data):
    ax = sns.lineplot(x="days_since_first_infection", y="confirmed",
                      hue="countries", data=data, legend="full")

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


def process_dataframe(df):
    df['countries'] = df['countries'].replace(countries_to_pt)
    df['days_since_first_infection'] = df.groupby(
        "countries").confirmed.rank(method='first', ascending=True)
    df['date'] = pd.to_datetime(df['date'])
    date_update = df['date'].max()
    date_max = date_update.strftime('%d %b')

    df['date'] = df['date'].dt.strftime('%d %b')
    df.set_index('date', inplace=True)

    prop = df.loc[date_max].sort_values(
        by='confirmed', ascending=False).head(30)

    #df.index = df.index.strftime('%d %b')
    cols = ['confirmed', 'recovered', 'deaths',
            'active', 'days_since_first_infection']
    out = dict(timeserie=dict(), fraction=dict(),
               last_update=date_update.strftime('%Y-%m-%dT%X'))
    countries = list(prop['countries'].unique())

    countries.remove('Brasil')
    countries.insert(0, 'Brasil')
    for country in countries:
        out['timeserie'][country] = df.query('countries == @country')[
            cols].reset_index().to_dict(orient='list')

    # build proportion of cases per country
    prop['active_frac'] = (100 * prop['active'] / prop['confirmed']).round(2)
    prop['recovered_frac'] = (
        100 * prop['recovered'] / prop['confirmed']).round(2)
    prop['deaths_frac'] = (100 * prop['deaths'] / prop['confirmed']).round(2)
    prop = prop[['countries', 'active_frac', 'recovered_frac', 'deaths_frac']]

    out['fraction'] = prop.to_dict(orient='list')
    return out


def load_daily_data(url):
    df = pd.read_csv(url)
    df.columns = [column.lower() for column in df.columns]
    df.rename(columns={'country_region': 'countries',
                       'last_update': 'date'}, inplace=True)
    df.drop(columns=['lat', 'long_', 'active'], inplace=True)
    #df['countries'] = df['countries'].replace(countries_to_pt)
    return df


def persist_dataset(data):
    with open('../data/data.json', 'w', encoding='utf8') as outfile:
        json.dump(data, outfile, ensure_ascii=False)


def push_file():
    os.system("git add ../data -u")
    os.system("git commit -m 'Update'")
    os.system("git push")

    # os.system("rsync -rvzP ../data/data.json mpi-contact:~/public_html/covid19/")


def run():
    df = load_files(url=url, filenames=filenames)
    df_daily = load_daily_data(url=url_daily)

    df = pd.concat([df, df_daily]).query('confirmed > 0')
    df['active'] = df['confirmed'] - df['recovered'] - df['deaths']

    # plot_confirmed_cases(data)

    data_json = process_dataframe(df=df)
    persist_dataset(data=data_json)
    push_file()


if __name__ == "__main__":
    while True:
        print('>>>', datetime.today())
        try:
            run()
        except:
            print(traceback.print_exc())
        time.sleep(15*60)

import json
import locale
import os
import time
import traceback
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings("ignore")

# locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
pd.set_option('precision', 8)

current_date = datetime.now().astimezone(
    pytz.timezone('America/Sao_Paulo'))

url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
url_daily = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv"
url_roylab = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQuDj0R6K85sdtI8I-Tc7RCx8CnIxKUQue0TCUdrFOKDw9G3JRtGhl64laDd3apApEvIJTdPFJ9fEUL/pub?gid=0&output=csv&sheet=CR_ROYLAB'
filenames = {'confirmed': 'time_series_covid19_confirmed_global.csv',
             'deaths': 'time_series_covid19_deaths_global.csv',
             'recovered': 'time_series_covid19_recovered_global.csv'}


countries_to_pt = {'Brazil': 'Brasil', 'France': 'França', 'Germany': 'Alemanha',
                   'Italy': 'Itália', 'Spain': 'Espanha', 'United States': 'EUA', 'Switzerland': 'Suíça',
                   'Netherlands': 'Holanda', 'Iran': 'Irã', 'Korea, South': 'Coréia do Sul',
                   'United Kingdom': 'Reino Unido', 'Belgium': 'Bélgica', 'Austria': 'Áustria',
                   'Norway': 'Noruega', 'Sweden': 'Suécia', 'Denmark': 'Dinamarca', 'Canada': 'Canadá',
                   'Malaysia': 'Malásia', 'Australia': 'Austrália', 'Japan': 'Japão', 'Ireland': 'Irlanda',
                   'Turkey': 'Turquia', 'Luxembourg': 'Luxemburgo', 'Pakistan': 'Paquistão', 'Czechia': 'Rep. Tcheca',
                   'Cruise Ship': 'Cruzeiro D. Princess', 'Ecuador': 'Equador', 'Poland': 'Polônia',
                   'Thailand': 'Tailândia', 'Romania': 'Romênia', 'Russia': 'Rússia', 'South Africa': 'África do Sul',
                   'Finland': 'Finlândia', 'Indonesia': 'Indonésia', 'Saudi Arabia': 'Arábia Saudita',
                   'Philippines': 'Filipinas', 'Greece': 'Grécia', 'India': 'Índia', 'Iceland': 'Islândia',
                   'Panama': 'Panamá', 'Singapore': 'Singapura', 'Mexico': 'México', 'Slovenia': 'Eslovênia',
                   'Estonia': 'Estônia', 'Croatia': 'Croácia', 'Peru': 'Perú', 'Dominican Republic': 'Rep. Dominicana',
                   'Qatar': 'Catar', 'Colombia': 'Colômbia', 'Egypt': 'Egito', 'Serbia': 'Sérvia', 'Iraq': 'Iraque',
                   'New Zealand': 'Nova Zelândia', 'Lebanon': 'Líbano', 'United Arab Emirates': 'Emirados Árabe',
                   'Lithuania': 'Lituânia', 'Armenia': 'Armênia', 'Morocco': 'Marrocos', 'Hungary': 'Hungria',
                   'Bulgaria': 'Bulgária', 'Ukraine': 'Ucrânia', 'Uruguay': 'Uruguai', 'Slovakia': 'Slováquia',
                   'Bosnia And Herzegovina': 'Bósnia'}


def get_data(df_confirmed, df_recovered, df_deaths):
    dfs = list()
    to_replace = {'Gambia, The': 'Gambia', 'The Gambia': 'Gambia',
                  'The Bahamas': 'Bahamas', 'Bahamas, The': 'Bahamas'}

    df_confirmed['Country/Region'] = df_confirmed['Country/Region'].replace(
        to_replace)
    df_recovered['Country/Region'] = df_recovered['Country/Region'].replace(
        to_replace)
    df_deaths['Country/Region'] = df_deaths['Country/Region'].replace(
        to_replace)

    df_confirmed = df_confirmed.drop(
        columns=['Province/State', 'Lat', 'Long']).groupby('Country/Region').sum()
    df_recovered = df_recovered.drop(
        columns=['Province/State', 'Lat', 'Long']).groupby('Country/Region').sum()
    df_deaths = df_deaths.drop(
        columns=['Province/State', 'Lat', 'Long']).groupby('Country/Region').sum()

    for country in df_confirmed.index.unique():
        if country == 'Belize':
            continue
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


def update_historical_data(filename='../data/dataset_timeseries.csv'):
    global current_date
    current_date = datetime.now().astimezone(pytz.timezone('America/Sao_Paulo'))
    # df = load_historical_data()
    df = load_files()
    df_daily = load_daily_data()
    # df_daily = load_daily_report()
    df = df[df['date'] < str(current_date.date())]
    df = pd.concat([df, df_daily]).sort_values(by=['country', 'date'])
    df['date'] = pd.to_datetime(df['date'])
    df.to_csv(filename, index=False)
    return df


def load_historical_data(filename='../data/dataset_timeseries.csv'):
    df = pd.read_csv(filename)
    return df


def load_daily_report():
    global current_date
    df = pd.read_csv(url_roylab)
    df.rename(columns={'Nation': 'country', 'Confirmed Case': 'confirmed',
                       'Recover': 'recovered', 'Death': 'deaths'}, inplace=True)
    df['country'] = df['country'].apply(lambda c: c.title())
    df['date'] = str(current_date.date())
    df = df[['date', 'country', 'confirmed', 'recovered', 'deaths']]
    to_replace = {'Bosnia-Herzegovina': 'Bosnia And Herzegovina', 'China, Mainland': 'China', 'Congo': 'Congo (Brazzaville)',
                  'Czech Republic': 'Czechia',
                  'S. Korea': 'Korea, South', 'S. Africa': 'South Africa', 'Taiwan': 'Taiwan*',
                  'Syrian Arab Republic': 'Syria', 'Saint Vincent': 'Saint Vincent And The Grenadines',
                  'Uae': 'United Arab Emirates', 'N. Macedonia': 'North Macedonia',
                  }
    df['country'] = df['country'].replace(to_replace)
    df.to_csv(f"../data/time-series/dataset_{current_date}.csv")
    return df


def load_files(url=url, filenames=filenames):
    df_confirmed = pd.read_csv(url+filenames['confirmed'], sep=',')
    df_deaths = pd.read_csv(url+filenames['deaths'], sep=',')
    df_recovered = pd.read_csv(url+filenames['recovered'], sep=',')
    df = get_data(df_confirmed, df_recovered, df_deaths)

    #df.rename(columns={'country': 'countries'}, inplace=True)
    #df['countries'] = df['countries'].replace(countries_to_pt)
    #df.index = df.index.strftime('%d %b')
    df.index.name = 'date'
    df.index = pd.to_datetime(df.index)
    df.to_csv('../data/dataset.csv', index=True)
    return df.reset_index()


def process_dataframe(df):
    global current_date
    df.fillna(0, inplace=True)
    df['country'] = df['country'].replace(countries_to_pt)
    df['days_since_first_infection'] = df.groupby(
        "country").confirmed.rank(method='first', ascending=True)
    df['date'] = pd.to_datetime(df['date'])
    date_update = df['date'].max()
    date_max = date_update.strftime('%d %b')
    print(date_max)

    df['date'] = df['date'].dt.strftime('%d %b')
    df.set_index('date', inplace=True)

    prop = df.loc[date_max].sort_values(
        by='confirmed', ascending=False).head(30)

    #df.index = df.index.strftime('%d %b')
    cols = ['confirmed', 'recovered', 'deaths',
            'active', 'days_since_first_infection']
    out = dict(timeserie=dict(), fraction=dict(),
               last_update=datetime.utcnow().strftime('%Y-%m-%dT%X'), stats=dict(),
               total=dict())
    countries = list(prop['country'].unique())

    for country in countries:
        out['timeserie'][country] = df.query('country == @country')[
            cols].reset_index().to_dict(orient='list')

    # build proportion of cases per country
    prop['active_frac'] = (100 * prop['active'] / prop['confirmed']).round(2)
    prop['recovered_frac'] = (
        100 * prop['recovered'] / prop['confirmed']).round(2)
    prop['deaths_frac'] = (100 - prop['active_frac'] -
                           prop['recovered_frac']).round(2)
    prop = prop[['country', 'active_frac', 'recovered_frac', 'deaths_frac']]

    total_df = df.loc[date_max].sort_values(
        by='confirmed', ascending=False)
    out['total'] = total_df[['confirmed', 'active',
                             'recovered', 'deaths']].sum().to_dict()
    out['stats'] = total_df[['country', 'confirmed',
                             'active', 'recovered', 'deaths']
                            ].set_index('country').to_dict(orient='index')

    out['fraction'] = prop.to_dict(orient='list')

    return out


def load_daily_data():
    global current_date
    df = pd.read_csv(url_daily)
    df.columns = [column.lower() for column in df.columns]
    df.rename(columns={'country_region': 'country',
                       'last_update': 'date'}, inplace=True)
    df.drop(columns=['lat', 'long_', 'active', 'date'], inplace=True)
    df['date'] = str(current_date.date())
    df.country = df.country.replace({'US': 'United States'})

    df['country'] = df['country'].apply(lambda c: c.title())

    df = df[['date', 'country', 'confirmed', 'recovered', 'deaths']]

    df.to_csv(f"../data/time-series/dataset_{str(current_date.date())}.csv")
    return df


def persist_dataset(data):
    with open('../data/data.json', 'w') as outfile:
        json.dump(data, outfile)


def push_file():
    os.system("git add ../data -u")
    os.system("git commit -m 'Automated Update'")
    os.system("git push")

    # os.system("rsync -rvzP ../data/data.json mpi-contact:~/public_html/covid19/")


def run():
    # df = load_files(url=url, filenames=filenames)
    df = update_historical_data()
    df = df.query('confirmed > 0')
    df['active'] = df['confirmed'] - df['recovered'] - df['deaths']

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

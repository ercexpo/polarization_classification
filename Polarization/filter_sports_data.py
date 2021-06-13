"""
filter_sports_data.py
Purpose: filter a collection of articles from sports section according to a set of keywords related to politics

Author/s:
 - Bernhard Clemm | github.com/bernhardclemm | b.f.d.clemm@uva.nl
"""
#==============================================================================
# MODULES -- DEPENDENCIES
#==============================================================================

import os
import json
import pandas as pd
import re

#==============================================================================
# MAIN
#==============================================================================

def filter_articles(folder, pattern):
    json_files = [x for x in os.listdir(folder) if x.endswith("json")]
    json_data = list()
    for json_file in json_files:
        json_file_path = os.path.join(folder, json_file)
        with open(json_file_path, "r") as f:
            json_data.append(json.load(f))
    df = pd.DataFrame(json_data)
    df_filtered = df[~df['text'].str.contains(pattern, regex=True)]
    return(df_filtered)

# US
folder_us = '/Users/bernhardclemm/Dropbox/Academia/EXPO/polarization/parsed/us_html_parsed'
pattern_us = 'Republican|Democrat|Trump|racism|sexism|politic'
us_filtered = filter_articles(folder_us, pattern_us)
us_filtered.to_csv('us_html_parsed_filtered.csv')

# NL
folder_nl = '/Users/bernhardclemm/Dropbox/Academia/EXPO/polarization/parsed/nl_html_parsed'
pattern_nl = 'VVD|D66|PVV|CDA|SP|PvdA|GL|FvD|PvdD|CU|Volt|JA21|SGP|DENK|BBB|BIJ1|OSF|Volkspartij voor Vrijheid en Democratie|Democraten 66|Partij voor de Vrijheid|Christen-Democratisch Appèl|Socialistische Partij|Partij van de Arbeid|GroenLinks|Forum voor Democratie|Partij voor de Dieren|Volt Nederland|Juiste Antwoord 2021|Staatkundig Gereformeerde Partij|BoerBurgerBeweging|Onafhankelijke Senaatsfractie'
nl_filtered = filter_articles(folder_nl, pattern_nl)
nl_filtered.to_csv('nl_html_parsed_filtered.csv')

# PL
folder_pl = '/Users/bernhardclemm/Dropbox/Academia/EXPO/polarization/parsed/pl_html_parsed'
pattern_pl = 'PiS|polityka|Andrzej Duda|Jarosław Kaczyński|Platforma Obywatelska|Prawo i Sprawiedliwość'
pl_filtered = filter_articles(folder_pl, pattern_pl)
pl_filtered.to_csv('pl_html_parsed_filtered.csv')

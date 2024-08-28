#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Yiru Xiong

This script contains codes to conduct ad-hoc data transformation and analytics
    1. Map zip codes to active service states, MSA (Metropolitan statistical area), cities
    2. create zip codes mapping dictionary for future use
    3. update active service area lists
    4. perform ad-hoc data analytics tasks
Note: 
The script has been updated and redacted to eliminate any client-sensitive information
"""
import pandas as pd
import numpy as np
import math
import json
import sqlalchemy as db
from sqlalchemy import *
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.sql import text
from sqlalchemy import inspect
from collections import defaultdict
#!pip install uszipcode
from uszipcode import SearchEngine, SimpleZipcode, Zipcode
from collections import Counter

# connect to the aws database 
personal = "/Users-Selected-Work-Directory/ "
data_path = str(personal)+"/database/"
script_path = str(personal)+"/database/script/"

database = "dbName"

access_info = pd.read_csv(script_path+"connect_to_database.csv", header=None)
pd.options.display.max_colwidth = 60
username = access_info.iloc[0].to_string(header=False, index=False).lstrip()
password = access_info.iloc[3].to_string(header=False, index=False).lstrip()
host = access_info.iloc[1].to_string(header=False, index=False).lstrip()
port = int(access_info.iloc[2])
aws_engine = create_engine(('postgresql://{user}:{pwd}@{h}:{p}/{d}').format(user=username, pwd=password, h=host,p=port, d=database))
aws_conn = aws_engine.connect()
aws_conn

aws_engine.table_names()

metadata = MetaData()
# retrieve zipcodes from user_details
user_details = db.Table('user_details_v2', metadata, autoload=True, autoload_with=aws_engine)
print(user_details.columns.keys())
zipcode_query = db.select([user_details.columns.ZIP])
fetched_info = aws_conn.execute(zipcode_query).fetchall() 
zipcodes_user = [r[0] for r in fetched_info]

# retrieve zipcodes from job_requests
job_requests = db.Table('job_requests', metadata, autoload=True, autoload_with=aws_engine)
print(job_requests.columns.keys())
zipcode_query2 = db.select([job_requests.columns.ZIP])
fetched_info2 = aws_conn.execute(zipcode_query2).fetchall()
zipcodes_jr = [r[0] for r in fetched_info2] # 451298 before dedup

# read in zip_to_MSA file
zip_to_msa = pd.read_excel(data_path+"ZIP_MSA_MAPPING.xls", dtype={"ZIP CODE":str, "MSA No.":str})
zip_to_msa = zip_to_msa.iloc[:,:-1]

#####################################
####  Geographic Info Mapping   #####
#####################################
#  1. check MSA, NON-MSA and those without lables in mappings
#  2. Active city/MSA/state are considered active if it has at least 20 job postings or user profiles mapped to it

all_msa_values = list(zip_to_msa["MSA Name"].unique())
non_msa = []

for i in all_msa_values:
    if i is not None:
        if not str(i).endswith('MSA') and not str(i).endswith("NONMETROPOLITAN AREA"):
            non_msa.append(i)

# modify MSA values from zip_to_MSA file
zip_to_msa["MSA Name"].replace(to_replace= np.nan, value = '', inplace=True )
zip_to_msa["MSA Name"].replace(to_replace= 'ARMED FORCES', value = '', inplace=True )
zip_to_msa["MSA Name"].replace(to_replace= 'All other territories and foreign countries', value = '', inplace=True )
zip_to_msa["MSA Name"].replace(to_replace= 'Tampa-St. Petersburg-Clearwater, FL', value = 'Tampa-St. Petersburg-Clearwater, FL MSA', inplace=True )

# every value in MSA column now either ends with MSA or NONMETROPOLITAN AREA
zip_to_msa_mod = zip_to_msa[["ZIP CODE", "STATE", "MSA Name"]]
zip_to_msa_dict = dict(zip(zip_to_msa["ZIP CODE"], zip_to_msa["MSA Name"]))
zip_to_state_dict = dict(zip(zip_to_msa["ZIP CODE"], zip_to_msa["STATE"]))

# combine unique zipcodes to a new list 
# unique zipcodes in zipcodes from user_details
zipcodes_user_dedup = set(zipcodes_user) # 18377
zipcodes_jr_dedup = set(zipcodes_jr) # 7959
zipcodes=(zipcodes_user_dedup).union(zipcodes_jr_dedup) #19745
zipcodes_list = [zip for zip in zipcodes if zip is not None] # 19744

# mapping each zip code in zipcodes_list (19744) to city, state and MSA
zipcode_mapping_dict = defaultdict(list)
state_notmatched = []
msa_notmatched = []
city_notmatched = []
state_diff_count = 0
search = SearchEngine()
for i in zipcodes_list:
    # try matching with the first 5 characters 
    # since some zip codes are longer or shorter, preprocess zipcodes
    if len(i) > 5:
        i = str(i.strip()[:5])
    elif len(i) < 5:
        i = i.zfill(5)
    # check if zipcode can be matched to a state in imported zip codes file first
    # if not found, try using zipcode search engine
    zipcode = search.by_zipcode(i)
    try:
        matched_state = zip_to_state_dict[i]
        matched_msa = zip_to_msa_dict[i]
    # zip code not found in the file
    # try using zipcode search engine
    except:
        matched_msa = None
        if zipcode.state is None:
            state_notmatched.append(i)
            matched_state = None
        # if zip code not found in file but found by search engine
        else:
            matched_state = zipcode.state
            if str(matched_state) != str(zipcode.state):
                state_diff_count += 1
                
    if not matched_state is None and matched_msa is None:
        msa_notmatched.append(i)
        
    if (not matched_state is None) and (zipcode.major_city is None):
        city_notmatched.append(i)
    
    zipcode_mapping_dict[i]=[zipcode.major_city, matched_msa, matched_state]

key_to_remove = ['=""="', 'CA 90']
# 18932 keys - (unique zipcodes)
for k in key_to_remove:
    zipcode_mapping_dict.pop(k) 

zipcode_mapping_pd = pd.DataFrame.from_dict(zipcode_mapping_dict, orient='index', columns=['CITY', 'MSA', 'STATE'])
zipcode_mapping_pd.reset_index(inplace=True)
zipcode_mapping_pd = zipcode_mapping_pd.rename(columns={'index':'ZIPCODE'})

# output dataframe to csv file
zipcode_mapping_pd.to_csv(data_path+"updated_zipcode_msa_mapping.txt", index=False)
# output the dictionary
with open(data_path+"updated_zipcode_msa_mapping_dict.json", "w") as outfile:
    json.dump(zipcode_mapping_dict, outfile)

# import the saved dictionary
#with open(data_path+"zipcode_msa_mapping_dict.json") as json_file:
#    m_dict = json.load(json_file)

# 56 states
all_states = [st for st in zipcode_mapping_pd["STATE"].unique() if st is not None]

# updated active city and state list 
active_states = defaultdict(list, {key:[] for key in all_states})

# by the definition of "active states" stated above in 9/19 updates
# for all zipcodes
for zip in zipcode_mapping_dict.keys():
    # get state
    state = zipcode_mapping_dict[zip][2]
    if not state is None:
        active_states[state].append(zip)

jr_zipcounts = Counter(zipcodes_jr) #7959
user_zipcounts = Counter(zipcodes_user) #18377

active_states_zip_counts = defaultdict(list, {key:0 for key in all_states})
for state, zl in active_states.items():
    if len(zl)>0:
        jr_cnt = 0
        user_cnt = 0
        max_cnt = 0
        for i in zl:
            jr_cnt += jr_zipcounts[i]
            user_cnt += user_zipcounts[i]
        max_cnt = max(jr_cnt, user_cnt)
        active_states_zip_counts[state]=max_cnt

# 51 active states
active_states_final = []
for state, zip_counts in active_states_zip_counts.items():
    if zip_counts > 20:
        active_states_final.append(state)

##########################
#####   add cities   #####
##########################
        
active_states_cities = defaultdict(list, {key:defaultdict(lambda:[])for key in active_states_final})
for zip in zipcode_mapping_dict.keys():
    # get city
    city = zipcode_mapping_dict[zip][0]
    state = zipcode_mapping_dict[zip][2]
    if (not city is None) and state in active_states_final:
        active_states_cities[state][city].append(zip)
        
active_states_cities_zip_counts = defaultdict(list, {key:defaultdict(lambda:0)for key in active_states_final})
# find counts of user profiles or job requests with mapped zipcodes to each city
for state, cities in active_states_cities.items():
    if cities is not None:
        for city, zl in cities.items():
            if len(zl)>0:
                jr_cnt = 0
                user_cnt = 0
                max_cnt = 0
                for i in zl:
                    jr_cnt += jr_zipcounts[i]
                    user_cnt += user_zipcounts[i]
                max_cnt = max(jr_cnt, user_cnt)
                if max_cnt > 20:
                    active_states_cities_zip_counts[state][city]=max_cnt     

active_states_cities_dict = defaultdict(list, {key:[]for key in active_states_final})
for state in active_states_cities_zip_counts.keys():
    if not state is None:
        active_states_cities_zip_sorted = sorted(active_states_cities_zip_counts[state].items(), key=lambda kv: kv[1],reverse=True)
        for l in range(len(active_states_cities_zip_sorted)):
            active_states_cities_dict[state].append(active_states_cities_zip_sorted[l][0])     

active_states_cities_pd = pd.DataFrame(active_states_cities_dict.items(), columns =['ACTIVE STATES','ACTIVE CITIES'])
active_states_cities_pd['ACTIVE CITIES'] = [','.join(map(str, c)) for c in active_states_cities_pd['ACTIVE CITIES']]    
active_states_cities_pd.to_csv(data_path+'updated_active_states_cities_list.csv', index=False)   

######################
#####     MSA    #####
######################
active_states_msa = defaultdict(list, {key:defaultdict(lambda:[])for key in active_states_final})
for zip in zipcode_mapping_dict.keys():
    # get city
    msa = zipcode_mapping_dict[zip][1]
    state = zipcode_mapping_dict[zip][2]
    if (not msa is None) and state in active_states_final:
        active_states_msa[state][msa].append(zip)

active_states_msa_zip_counts = defaultdict(list, {key:defaultdict(lambda:0)for key in active_states_final})
# find counts of user profiles or job requests with mapped zipcodes to each city
for state, msa in active_states_msa.items():
    if msa is not None:
        for m, zl in msa.items():
            if len(zl)>0:
                jr_cnt = 0
                user_cnt = 0
                max_cnt = 0
                for i in zl:
                    jr_cnt += jr_zipcounts[i]
                    user_cnt += user_zipcounts[i]
                max_cnt = max(jr_cnt, user_cnt)
                if max_cnt > 20:
                    active_states_msa_zip_counts[state][m]=max_cnt    
                    
# sort the msa list
active_states_msa_dict = defaultdict(list, {key:[]for key in active_states_final})
for state in active_states_msa_zip_counts.keys():
    if not state is None:
        active_states_msa_zip_sorted = sorted(active_states_msa_zip_counts[state].items(), key=lambda kv: kv[1],reverse=True)
        for l in range(len(active_states_msa_zip_sorted)):
            active_states_msa_dict[state].append(active_states_msa_zip_sorted[l][0])     

active_states_msa_pd = pd.DataFrame(active_states_msa_dict.items(), columns =['ACTIVE STATES','ACTIVE MSA'])
active_states_msa_pd['ACTIVE MSA'] = [','.join(map(str, c)) for c in active_states_msa_pd['ACTIVE MSA']]    
active_states_msa_pd.to_csv(data_path+'updated_active_states_msa_list.csv', index=False)

# active_states_final
# update mapping dict 
# include only active state, city and MSA
active_msa_final = []
for m in active_states_msa_dict.values():
    active_msa_final.extend(m)
    
active_cities_final=[]
for c in  active_states_cities_dict.values():
    active_cities_final.extend(c)

# update zipcode_mapping_pd 
zipcode_mapping_pd.loc[~zipcode_mapping_pd ["STATE"].isin(active_states_final), "STATE"] = ""
zipcode_mapping_pd.loc[~zipcode_mapping_pd ["CITY"].isin(active_cities_final), "CITY"] = ""
zipcode_mapping_pd.loc[~zipcode_mapping_pd ["MSA"].isin(active_msa_final), "MSA"] = ""

# zipcode_mapping_pd["STATE"].unique() # 52 (51+"")
# output dataframe to csv file
zipcode_mapping_pd.to_csv(data_path+"zipcode_mapping_active_service_areas.csv", index=False)
# output the dictionary
with open(data_path+"zipcode_mapping_active_service_areas.json", "w") as outfile:
    json.dump(zipcode_mapping_dict, outfile)

#################################################
# codes before 9/19 updates
top_active_states = list(active_states["STATE"])
active_cities = defaultdict(list, {key:defaultdict(lambda:0)for key in top_active_states})
for zip in zipcode_mapping_dict.keys():
    # get city
    city = zipcode_mapping_dict[zip][0]
    state = zipcode_mapping_dict[zip][2]
    if (not city is None) and (state in top_active_states):
        active_cities[state][city] += 1

active_cities_top =  defaultdict(list,{key: [] for key in top_active_states})
for state in active_cities.keys():
    if not state is None:
        city_dict = active_cities[state]
        active_cities_sorted = sorted(city_dict.items(), key=lambda kv: kv[1],reverse=True)
        for l in range(len(active_cities_sorted)):
            if active_cities_sorted[l][1] > 15:
                active_cities_top[state].append(active_cities_sorted[l][0])
                
# zipcodes unmatched in job_requests table
state_notmatched_jr = set(state_notmatched).intersection(zipcodes_jr_dedup)
msa_notmatched_jr = set(msa_notmatched).intersection(zipcodes_jr_dedup)
city_notmatched_jr = set(city_notmatched).intersection(zipcodes_jr_dedup)

state_not_in_list= set()
city_not_in_list= set() # zipcode indicates state in the list but a city not in the active list
missing_state = set()
missing_cities = set()

for key, value in zipcode_mapping_dict.items():
    if not any(str(value[0]) in cities for cities in all_cities):
        # print("value 0 is:", str(value[0]))
        # city_not_in_list.append(key)
        if (not str(value[1]) is None) and (str(value[1]) not in all_states) :
            print("zipcode is", key, "state is", value[1])
            state_not_in_list.add(key)
            missing_state.add(value[1])
        else:
            city_not_in_list.add(key)
            missing_cities.add(value[0])

all_states = []
for key in zipcode_mapping_dict.keys() :
    all_states.append(zipcode_mapping_dict[key][1])        

state_counts = Counter(all_states)
state_counts.most_common()
missing_states_dict = defaultdict(list,{ k:[] for k in list(missing_state)})
for key in missing_states_dict.keys():
    if key in state_counts:
        missing_states_dict[key] = state_counts[key]
sorted_missing_states = sorted(missing_states_dict.items(), key=lambda kv: kv[1],reverse=True)
    
with open(data_path+'missing_states_counts.txt', 'w') as convert_file:
     convert_file.write(json.dumps(sorted_missing_states))    


# Notes: run state_abbrev located in the end of this script first before proceeding 
# run codes within ====== blocks below if need to compare and update the active cities list
# =============================================================================
# active cities
active_cities = pd.read_excel(data_path+"state_city_mapping.xlsx", header=None)
active_cities[2] = [state_abbrev[x] for x in active_cities[0]]
active_cities[3] = [str(x).split(",") for x in active_cities[1]]
active_cities[3] = [[s.strip() for s in l] for l in active_cities[3]]
# state - city mapping 
state_cities_mapping = dict(zip(active_cities[2], active_cities[3]))

# initialize dictionary with cities 
cities=[]
all_cities=[]
for values in state_cities_mapping.values():
    cities.extend(values)

all_cities = [x.strip() for x in cities if x != 'nan']
all_states = list(set(active_cities[2]))
# =============================================================================

# adapted from https://gist.github.com/rogerallen/1583593
state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'Washington, DC': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

###################
## Reference   ####
# source file from https://www.dol.gov/owcp/regs/feeschedule/fee/fee11/fs11_gpci_by_msa-zip.xls 
# see reference file for details

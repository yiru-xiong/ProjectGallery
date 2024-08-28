 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
local version: XXXXX
relational database design based on schema XXXX

@author: Yiru Xiong

This script facilitates the end-to-end process of transitioning raw data from an on-premise environment to a cloud-based relational database. 
It encompasses data gathering, database design, ingestion, cleaning, transformation, and preparation for querying.
Note: 
The script has been updated and redacted to eliminate any client-sensitive information
"""

import re
import io
import os
import csv
import numpy as np
import pandas as pd
import glob
from os import listdir
from io import StringIO
import sqlalchemy as db
from sqlalchemy import *
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.sql import text
from sqlalchemy import inspect

# Note: please change the data path to data folder, script path to script folder before proceeding 
# data path where raw data files are stored in original csv formats
personal = "/Users-selected-work-directory"
data_path = str(personal)+"/database/"
script_path = str(personal)+"/database/script/"

database = "dbName"

# connect to the database on localhost
engine = create_engine(('postgresql://localhost/{d}').format(d=database))
conn = engine.connect()
#conn

# connect to aws server 
# retrieve access information from file
access_info = pd.read_csv(script_path+"connect_to_database.csv", header=None)
pd.options.display.max_colwidth = 60
username = access_info.iloc[0].to_string(header=False, index=False).lstrip()
password = access_info.iloc[3].to_string(header=False, index=False).lstrip()
host = access_info.iloc[1].to_string(header=False, index=False).lstrip()
port = int(access_info.iloc[2])
aws_engine = create_engine(('postgresql://{user}:{pwd}@{h}:{p}/{d}').format(user=username, pwd=password, h=host,p=port, d=database))
aws_conn = aws_engine.connect()
aws_conn

# local database - all tables
engine.table_names()

# aws database - all tables
metadata = MetaData()
metadata.reflect(bind=aws_engine)
for table in metadata.sorted_tables:
    print(table)

# Querying examples
# retrieve entry from 'client_details' table where CLIENT_ID = 42 in aws database 
client_details = db.Table('client_details', metadata, autoload=True, autoload_with=aws_engine)
query = db.select([client_details]).where(client_details.columns.CLIENT_ID == 29653)
res = aws_conn.execute(query)
requested_info = res.fetchall()
print(requested_info)
#conn.execute("commit")
#conn.close()

###################
# load the schema #
###################
# to design a relation database, create a Schema with table mappings first 
design = open(data_path+"dbSchema.sql")

command = ''

for c in design:
    if c.strip('\n'):
        command += c.strip('\n')
#        print(command)
        
        if command.endswith(';'):
            try:
                engine.execute(text(command))
                #engine.commit()
            except:
                print(str(command)+' unable to proceed with this command!')
            finally:
                command = ''

# modify schemas
ref = """ALTER TABLE "jobs" ADD FOREIGN KEY ("JOB_ID") REFERENCES "client_ratings" ("JOB_ID");"""         
engine.execute(ref)

# temporarily modify constraints

# fix attributes in the schema
engine.execute("""ALTER TABLE "user_details_v2" ALTER COLUMN "USER_ID" TYPE BIGINT;""")
engine.execute("""ALTER TABLE "user_details_v2" ALTER COLUMN "ZIP" TYPE VARCHAR;""")


#########################
#    Helper Functions   #
#########################
# run helper functions before preceeding 
def contains_non_numeric(df, col_name):
    res = df[df[col_name].apply(lambda x: not str(x).isnumeric())]
    print("This column contains non numeric values:", res)


def fix_linebreak(sentence):
    pattern.sub(lambda x: x.group().replace('\n', ''), sentence)
    

# clear entries without dropping tables in dataset
def clear_rows_in_existing_tables(eng, table_name):
    meta = MetaData(engine)
    meta.reflect(engine)
    table = meta.tables[table_name]
    del_entry = table.delete()
    conn = eng.connect()
    res = conn.execute(del_entry)


# count the number of delimiters(comma) in a line and find lines with excess delimiters
def line_with_extra_comma(infile,expected_cnt):
    issue_lines=[]
    with open(infile, 'rt') as file:
        for line in file:
            if line.count(',') != expected_cnt:
                issue_lines.append(line)
    return issue_lines

def remove_quotations_header(df):
    df.columns = [col[1:-1] for col in df.columns]
    

###################################################
##      read in and load data into database      ##
###################################################   

##################
# client_details #
##################
client_details = pd.read_csv(data_path+"client_details_v3.csv")
# remove duplicates by unique client_id-parent_id pair
# keep the first record 
# (2162, 6) - 2162 unique pairs with 6 attributes
client_details_dedup = client_details.drop_duplicates(subset=['CLIENT_ID','PARENT_ID'], keep ='first').reset_index(drop=True)
#client_details_dedup = client_details.groupby(['CLIENT_ID','PARENT_ID']).first()
# (2162, 4) - 2162 unique pairs with 4 attributes 
client_details_df = pd.DataFrame(client_details_dedup.drop(columns=['SIZE','INDUSTRY']))
# check and clean data 
# check if all IDs are numeric - all values are numeric.
contains_non_numeric(client_details_df, 'CLIENT_ID')

# insert data into database 
# client_details_df.to_sql(name="client_details", con=engine, index = False,if_exists='append')

####################
# parent_companies #
####################
# create parent_companies table
# (2162, 3) 
parent_companies = client_details_dedup[['PARENT_ID','SIZE','INDUSTRY']].copy()
# remove duplicates by unique parent_id
# (1827, 3)
parent_companies_dedup = parent_companies.drop_duplicates(subset='PARENT_ID',keep='first').reset_index(drop=True)
parent_companies_df = pd.DataFrame(parent_companies_dedup)

# insert data
# parent_companies_df.to_sql(name="parent_companies", con=engine, index = False, if_exists='append')

# load all required tables

################
# user_details #
################
user_details = pd.read_csv(data_path+"user_details_v2.csv",dtype={"BADGE_NAMES":"string", "ZIP":"string"})
user_details_df = pd.DataFrame(user_details)

# data checking  
# USER_ID: all numeric
contains_non_numeric(user_details_df, 'USER_ID')
# SUSPENDED - contains only False and True
user_details_df['SUSPENDED'].unique()
# GENDER - 'female', 'male', 'unknown'
user_details_df['GENDER'].fillna('unknown', inplace=True)
user_details_df['GENDER'].unique()
# clear_rows_in_existing_tables("user_details_v2")
# user_details_df.to_sql(name="user_details_v2", con=engine, index = False, if_exists='append')

  
######################
#    user_activity   #
######################
# TO_DO: fix line breaks in user_activity
ua_files = glob.glob(data_path+"user_activity/*.csv")
ua_list = []
# column (8) have mixed types
for filename in ua_files:
    ua_df = pd.read_csv(filename, index_col=None, header=0, dtype={"NUM_JOBS_VIEWED":"Int64", "NUM_JOBS_ACCEPTED":"Int64"})
    ua_df = ua_df.replace({r'\\N': 0}, regex=True)
    ua_list.append(ua_df)

user_activities = pd.concat(ua_list, axis = 0, ignore_index = True)
user_activities.replace({r'\\N': 0}, regex=True)

# unresolved USER_ID issues
# (5375171,9)
user_activities = user_activities[user_activities['USER_ID'].isin(user_id_list)]

# check the last column-NUM_NOTIFS - only contains numbers
contains_non_numeric(user_activities, 'NUM_NOTIFS')
user_activities["USER_ID"] = pd.to_numeric(user_activities["USER_ID"], errors='coerce').fillna(0).astype(np.int64)
user_activities_df = pd.DataFrame(user_activities)
# user_activities_df.to_sql(name="user_activity", con=aws_engine, index = False, if_exists='append',method = 'multi')


#################
#     jobs      #
#################
# created a separate folder to store formatted job files
new_folder = "new_jobs"
jobs_path = os.path.join(data_path, new_folder)
if not os.path.exists(jobs_path):
    os.makedirs(jobs_path)


# preprocess all job files and save formatted files to new folder
# remove line breaks in quotations, extra whitespaces etc.
   
for f in listdir(data_path+"jobs/"):
    content = open(data_path+"jobs/"+str(f), "r").read()
    #content = open(data_path+'jobs_4.csv', "r").read().replace(r'\r+|\n+|\t+','')
    #content = "\n".join([x.replace("\n"," ") for x in re.findall('".*?"',content,flags=re.DOTALL)])
    # convert all single quotations to double first
    content1 = content.replace("'",'"')
    # remove quotations around null 
    content2 = content1.replace('"null"',"null")
    # remove unwanted line breaks
    content3 = re.sub(r'"[^"]*(?:""[^"]*)*"', lambda m: m.group(0).replace("\n", ""), content2)
    content4 = re.sub(r'"[^"]*(?:""[^"]*)*"', lambda m: re.sub(r'\n+', '', m.group(0)), content3)
    content5 = re.sub(r'"[^"]*"', lambda x: x.group(0).replace(",", ""), content4) 
    content6 = re.sub(r'"[^"]*(?:""[^"]*)*"', lambda k: re.sub(' +',' ', k.group(0)), content5)
    content7 = content6.replace('"', "'")

    with open(jobs_path+"/"+str(f), 'w') as f:
        for line in StringIO(content7):
            f.write(line)


jb_list = []

# read in formatted data 
for filename in jb_files:
    print("reading .."+str(filename))
    jb_df = pd.read_csv(filename, index_col=None, header=0, sep=',', error_bad_lines=False, low_memory=False)
    jb_list.append(jb_df)
    print("finishing .."+str(filename))

jobs_raw = pd.concat(jb_list, axis=0, ignore_index=True)

# cleaning and checking data 

remove_quotations_header(jobs_raw)
jb_clean = jobs_raw[jobs_raw["JOB_ID"].astype('str').str.isnumeric()]

# datetime 
jb_clean["STARTED_AT"] = pd.to_datetime(jb_clean["STARTED_AT"])
jb_clean["COMPLETED_AT"] = pd.to_datetime(jb_clean["COMPLETED_AT"])

# value: nan, False, True, 'false', 'true'
jb_clean["BAN"].unique()
jb_clean["BAN"].replace({"'false'": "False", "'true'":"True"},inplace=True)

# final checkings
# all JOB_ID are numeric
contains_non_numeric(jb_clean, "JOB_ID")

# created jobs table
jobs = jb_clean.drop(columns=["RATING_OF_CLIENT","RATING_OF_CLIENT_COMMENT","RATING", "RATING_OF_COMMNET"])
jobs['BAN'] = jobs ['BAN'].astype('bool')
# check for all existing values
jobs.BAN.unique()
allowed_value = [True, False, pd.NaT]
jobs.BAN = np.where(jobs.BAN == 'false', False,jobs.BAN)
jobs.BAN = np.where(jobs.BAN == 'true', True,jobs.BAN)
jobs["BAN"]=np.where(jobs.BAN.isin([True, False]), jobs.BAN, pd.NA)
jobs_df = pd.DataFrame(jobs)

#################
#  job_requests #
#################
# created a separate folder to store formatted job files
new_folder = "new_job_requests"
jr_path = os.path.join(data_path, new_folder)
if not os.path.exists(jr_path):
    os.makedirs(jr_path)

# preprocess all job request files and save formatted files to new folder
# remove line breaks in quotations, extra whitespaces in columns with large text
for f in listdir(data_path+"job_requests/"):
    content = open(data_path+"job_requests/"+str(f), "r").read()
    content1 = content.replace("'",'"')
    content2 = content1.replace('"null"',"null")
    # remove unwanted line breaks
    #pattern = re.compile(r'".*?"', re.DOTALL)
    content3 = re.sub(r'"[^"]*(?:""[^"]*)*"', lambda m: m.group(0).replace("\n", ""), content2)
    content4 = re.sub(r'"[^"]*(?:""[^"]*)*"', lambda m: re.sub(r"\t+\n+", " ", m.group(0)), content3)
    # remove comma in columns of long text values
    content5 = re.sub(r'"[^"]*"', lambda x: x.group(0).replace(",", " "), content4) 
    # remove possible punctuations
    content6 = re.sub(r'"[^"]*"', lambda y: re.sub(r'[?!;*]', " ", y.group(0)), content5)
    content8 = re.sub(r'"[^"]*(?:""[^"]*)*"', lambda k: re.sub(' +',' ', k.group(0)), content6)
    content9 = content8.replace('"', "'")

    
    with open(jr_path+"/"+str(f), 'w') as f:
        for line in StringIO(content9):
            f.write(line)


jr_files = glob.glob(jr_path+"/*.csv")
jr_list = []

# read in formatted data 
for filename in jr_files:
    print("reading .."+str(filename))
    jr_df = pd.read_csv(filename, index_col=None, header=0, sep=',',error_bad_lines=False, low_memory=False)
    jr_list.append(jr_df)
    print("finishing .."+str(filename))

jr_raw = pd.concat(jr_list, axis=0, ignore_index=True)

# checking and cleaning 
remove_quotations_header(jr_raw)
jr_clean1 = jr_raw[jr_raw["JOB_REQUEST_ID"].astype('str').str.isnumeric()]
jr_clean2 = jr_clean1[jr_clean1["CLIENT_ID"].astype('str').str.isnumeric()]

jr_clean2["CREATED_AT"] = pd.to_datetime(jr_clean2["CREATED_AT"])
jr_clean2["POSTED_AT"] = pd.to_datetime(jr_clean2["POSTED_AT"])
jr_clean2["START_AT"] = pd.to_datetime(jr_clean2["START_AT"])
jr_clean2["END_AT"] = pd.to_datetime(jr_clean2["END_AT"])
jr_clean2["CANCELLED_AT"] = pd.to_datetime(jr_clean2["CANCELLED_AT"],errors='coerce')
# convert ID columns to numeric
jr_clean2["CLIENT_ID"] = pd.to_numeric(jr_clean2["CLIENT_ID"], errors='coerce').fillna(0).astype(np.int64)
jr_clean2["JOB_REQUEST_ID"] = pd.to_numeric(jr_clean2.JOB_REQUEST_ID).fillna(0).astype(np.int64)
contains_non_numeric(jr_clean2, "CLIENT_ID")
contains_non_numeric(jr_clean2, "JOB_REQUEST_ID")

# dedup primary key
jr_dedup = jr_clean2.drop_duplicates(subset=['JOB_REQUEST_ID'], keep ='first').reset_index(drop=True)
jr = jr_dedup.drop(columns=['PARENT_ID','DESCRIPTION_SKILLS','DESCRIPTION_TASKS'])

# unresolved client ID issues
jr_df = pd.DataFrame(jr)


#########################################
######### Custom Data Cleaning ##########
#########################################
## Add placeholders in client_details table for missing CLIENT_ID and PARENT_ID
cid_fb = list(f_and_b["CLIENT_ID"].unique())
cid_c = list(client_details_df["CLIENT_ID"].unique())
cid_diff_list = list(set(cid_fb) - set(cid_c))
cid_diff_list_final = [0 if pd.isna(x) else x for x in cid_diff_list]
# step 1. get CIENT_ID and PARENT_ID in favorites_and_blocks table
fb_subset = f_and_b[f_and_b['CLIENT_ID'].isin(cid_diff_list_final)]
# step 2. dedup by CLIENT_ID and PARENT_ID- (1041,6)
fb_subset_dedup = fb_subset.drop_duplicates(subset=['CLIENT_ID','PARENT_ID'], keep ='first').reset_index(drop=True)
# step 3. retrieve all missing CLIENT_ID and PARENT_ID 
fb_missing_cid = fb_subset_dedup[['CLIENT_ID','PARENT_ID']]
# step 4. prepare table of entries to be appended to the client_details_df table 
client_columns = list(client_details_df.columns.values)
print(client_columns)
# numpy: np.NaN
fb_add_to_client = fb_missing_cid.assign(CREATED_DATE=pd.NaT, ACTIVATED_DATE=pd.NaT)
# step 5. add to client_details_df table 
client_details_mod1 = client_details_df.append(fb_add_to_client)

## repeat the same process step 1- step 5 as above to all related tables

###########################################
######### Check for Missing Data ##########
###########################################

# check for missing PARENT_IDs in related tables
c_pid = client_details_mod3["PARENT_ID"].unique()
client_rating_df["CLIENT_ID"] = pd.to_numeric(client_rating_df["CLIENT_ID"], errors='coerce').fillna(0).astype(np.int64)
pn_pid_diff_list = list(set(p_n["PARENT_ID"].unique()) - set(c_pid)) #0
fb_pid_diff_list = list(set(f_and_b["PARENT_ID"].unique()) - set(c_pid)) #0

# finalize client_details table
client_details_mod3.CLIENT_ID=pd.to_numeric(client_details_mod3.CLIENT_ID)
# check for all Primary Key - no NULL value
client_details_mod3.CLIENT_ID.isnull().values.any()
# check for all Primary Key - unique values 
client_details_mod3.CLIENT_ID.nunique()
client_details_mod4 = client_details_mod3.drop_duplicates(subset=['CLIENT_ID'], keep ='first').reset_index(drop=True)
client_details_final = pd.DataFrame(client_details_mod4)
client_details_final.shape

## repeat steps for all tables 
## updating job_request_tasks table using finalzied job_request table
jr_tasks_df["JOB_REQUEST_ID"] = pd.to_numeric(jr_tasks_df["JOB_REQUEST_ID"], errors='coerce').fillna(0).astype(np.int64)
rid_jr = list(jr_final["JOB_REQUEST_ID"].unique())
rid_task = list(jr_tasks_df["JOB_REQUEST_ID"].unique())
rid_diff_list = list(set(rid_jr) - set(rid_task))
task_rid_diff_list = [0 if pd.isna(x) else x for x in rid_diff_list]
jr_subset = jr_final[jr_final["JOB_REQUEST_ID"].isin(task_rid_diff_list)]
jr_subset_dedup = jr_subset.drop_duplicates(subset=["JOB_REQUEST_ID"], keep ='first').reset_index(drop=True)

task_missing_rid = jr_subset_dedup[["JOB_REQUEST_ID"]]

task_columns = list(jr_tasks_df.columns.values)
print(task_columns)

jr_add_to_task = task_missing_rid.assign(DESCRIPTION_SKILLS='', DESCRIPTION_TASKS='')
jr_task_mod = jr_tasks_df.append(jr_add_to_task)
jr_task_mod.JOB_REQUEST_ID.isnull().values.any() #False
contains_non_numeric(jr_task_mod, "JOB_REQUEST_ID")
jr_task_final = pd.DataFrame(jr_task_mod)

#######################################
###    load data into database      ###
#######################################
## Notes: back up latest version of the existing database first before preceeding
#  drop contents without dropping table and dependencies
metadata = MetaData()
#metadata.reflect(bind=aws_engine)
metadata.reflect(bind=engine)
for table in metadata.sorted_tables:
    #print(table)
    try:
        clear_rows_in_existing_tables(engine, str(table))
        print(str(table)+" has been cleared successfully")
    except:
        print(str(table)+" need to be cleared manually")
    
# manually clear tables that has not been cleared from the previous step
clear_rows_in_existing_tables(engine,"parent_companies")
clear_rows_in_existing_tables(engine,"client_details")
clear_rows_in_existing_tables(engine, "parent_companies")
clear_rows_in_existing_tables(engine, "job_requests")


# load finalized tables into the database schema

# parent_companies_final (2709 unique PARENT_ID)
parent_companies_final.to_sql(name="parent_companies", con=engine, index=False, if_exists='append')
# client_details_final (3305 unique CLIENT_ID)
client_details_final.to_sql(name="client_details", con=engine, index=False, if_exists='append')
# user_details_final (665680 unique USER_ID)
user_details_final.to_sql(name="user_details_v2", con=engine, index=False, if_exists='append')
f_and_b_df.to_sql(name="favorites_and_blocks", con=engine, index=False, if_exists='append')
# jr_final (451298, 15)
jr_final.to_sql(name="job_requests", con=engine, index=False, if_exists='append')
jr_task_final.to_sql(name="job_request_tasks", con=engine, index=False, if_exists='append')
p_n_df.to_sql(name="preferred_notifications", con=engine, index=False, if_exists='append')
user_activities_df.to_sql(name="user_activity", con=engine, index=False, if_exists='append')
jobs_df.to_sql(name="jobs", con=engine, index=False, if_exists='append')
client_rating_df.to_sql(name="client_ratings", con=engine, index=False, if_exists='append')

#########################
##   Data Validation   ##
#########################
# check for USER_ID 
uid_u = list(user_details_df["USER_ID"].unique())
uid_ua = list(user_activities["USER_ID"].unique())
uid_fb = list(f_and_b["USER_ID"].unique())
uid_diff_ua_u = list(set(uid_ua) - set(uid_u))
uid_diff_fb_u = list(set(uid_fb) - set(uid_u))
uid_diff_fb_ua = list(set(uid_fb) - set(uid_ua))

not_in_u = []
for i in uid_ua:
    if i not in uid_u:
        #print(i)
        not_in_u.append(i)

# check CLIENT_ID
#f_and_b["cid_exists"].unique()
cid_fb = list(f_and_b["CLIENT_ID"].unique())
cid_c = list(client_details_df["CLIENT_ID"].unique())
cid_diff_list = list(set(cid_fb) - set(cid_c))

# check CLIENT_PARENT pairs - 1041 pairs unmatched
f_and_b["CP_ID"] =f_and_b["CLIENT_ID"].astype(str)+f_and_b["PARENT_ID"]
client_details_dedup["CP_ID"] = client_details_dedup["CLIENT_ID"].astype(str)+client_details_dedup["PARENT_ID"]
cp_id_fb = list(f_and_b["CP_ID"].unique())
cp_id_c = list(client_details_dedup["CP_ID"].unique())
cp_id_diff_list = list(set(cp_id_fb) - set(cp_id_c))

# output results to excel workbook
uid_diff_ua_u_df = pd.DataFrame(uid_diff_ua_u) 
uid_diff_fb_u_df = pd.DataFrame(uid_diff_fb_u) 
uid_diff_fb_ua_df = pd.DataFrame(uid_diff_fb_ua) 
with pd.ExcelWriter(data_path+'Missing User IDs.xlsx') as writer:
    uid_diff_ua_u_df.to_excel(writer,sheet_name='1.Only_in_Activities',index=False)
    uid_diff_fb_u_df.to_excel(writer, sheet_name='2.Only_in_fav_blocks', index=False)
    uid_diff_fb_ua_df.to_excel(writer, sheet_name='3.In_2_Not_In_1', index=False)
writer.save()

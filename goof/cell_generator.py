#this generates cells from the dataset file test.txt's locations using utils.py's thingamazig
import json, pdb, os, sys
from location_helper import generate_cell, lookup_address_only
#from utils2 import get_db_connection

# ../venv/bin/python cell_generator.py twt_jun25_jul24_2018_offline_geo.json twt_jun25_jul24_2018_offline_cell.json offline
# offline vs online --> for offline, check news_litmus
# for online, no need to check news_litmus


INPUT_FILE = sys.argv[1]
OUTPUT_FILE = sys.argv[2]
'''
if sys.argv[3] == 'offline':
    OFFLINE_ = 1
elif sys.argv[3] == 'online':
    OFFLINE_ = 0
else:
    OFFLINE_ = 1
'''
MS_IN_DAYS = 86400000

if not os.path.exists(OUTPUT_FILE):
    open(OUTPUT_FILE,'w').close()

'''
db = get_db_connection()
cursor = db.cursor()
'''
counter = 0
missing = 0
extant = 0
extras=0

json_d = {}
with open(INPUT_FILE, 'r') as tweet_file:
    for line in tweet_file:
        line_ = json.loads(line.strip())
        json_d[line_['id']] = line_
#json_d is a dict of each line. now for each entry, we'll figure out its cell location, and store that as well

ex_cells = {}
for entry in json_d:
    p_line = json_d[entry]
    
    #now we lookup the location provided and generate cell. Then we need to check if this cell exists in our news_litmus dataset
    
    if type(p_line['locations']) is list:
        p_line['locations'] = ','.join(p_line['locations'])
    addr = lookup_address_only(p_line['locations'].encode('utf-8'))
    
    if addr[0] is not None and addr[1] is not None:
        cell = generate_cell(addr[0],addr[1])
        if cell not in ex_cells:
            if OFFLINE_:
                select_s = 'SELECT location from news_litmus where cell = %s and timestamp > %s and timestamp < %s'
                params = (cell, str(int(p_line['timestamp'])-15*MS_IN_DAYS), str(int(p_line['timestamp'])+15*MS_IN_DAYS))
                cursor.execute(select_s, params)
                res = cursor.fetchall()
                if len(res) > 0:
                    ex_cells[cell] = 1
                else:
                    #Dis no good
                    missing+=1
                    continue
            else:
                ex_cells[cell] = 1
        #at this point, either cell is preexisting, or new but in news (if offline), so all good
        extant+=1
        p_line['cell'] = cell
        if OFFLINE_:
            p_line['label'] = 1
        with open(OUTPUT_FILE, 'a') as out_file:
            out_file.write(json.dumps(p_line) + '\n')
    else:
        print p_line
        counter+=1

print '\n\n Null: ', counter
print 'Stuff thats no good: ', missing
print 'Yay: ', extant


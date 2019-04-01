import glob
import pdb
import time
import calendar
import json

def str2utc(s):
    # parse twitter time string into UTC seconds, unix-style
    # python's bizarro world of dates, times and calendars
    return calendar.timegm(time.strptime(s, "%a %b %d %H:%M:%S +0000 %Y"))

def utc2snowflake(stamp):
    return (int(round(stamp * 1000)) - 1288834974657) << 22

def snowflake2utc(sf):
    return ((sf >> 22) + 1288834974657) / 1000.0

def str2utcms(s):
    return 1000 * calendar.timegm(time.strptime(s, "%a %b %d %H:%M:%S +0000 %Y"))

def snowflake2utcms(sf):
    return ((sf >> 22) + 1288834974657)

# really is the best way to get utc timestamp?
#   (minus changing your box to be UTC)
def utcnow():
    calendar.timegm(datetime.datetime.utcnow().timetuple())

fList = glob.glob('data/gooddata/*.json')

totalData = []
negative_data = []
for file in fList:
    # Some data files have fake negatives (namely july-december2018)
    if 'aibek' in file:
        continue
        # Use aibek_test_data as training data
    fakeNegatives = True
    if 'aibek' in file or 'apriori' in file:
        fakeNegatives = False
    
    print file, 'fakes: ', fakeNegatives
    with open(file, 'r') as file_:
        for line in file_:
            json_ = json.loads(line.strip())
            if 'timestamp' not in json_:
                json_['timestamp'] = long(snowflake2utcms(long(json_['id_str'])))
            else:
                json_['timestamp'] = long(json_['timestamp'])
            # skip zero labek for fakeNegs
            if json_["label"] == 0:
                negative_data.append(json_)
            else:
                totalData.append(json_)


newTotalData = sorted(totalData,key=lambda k:k['timestamp'])

with open('apriori_to_december_sorted_positive.json', 'w') as allWrite:
    for entry in newTotalData:
        allWrite.write(json.dumps(entry) + '\n')

with open('apriori_to_december_negatives.json', 'w') as allWrite:
    for entry in negative_data:
        allWrite.write(json.dumps(entry) + '\n')








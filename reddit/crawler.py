import datetime
import time
import json
import requests


REDDIT_NEW_URL = "https://www.reddit.com/r/gifs/new.json"
REDDIT_HOT_URL = "https://www.reddit.com/r/gifs/hot.json"


def main():
    while True:
        ts = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        new = requests.get(REDDIT_NEW_URL, headers={"User-agent": "ConceptDriftBot v0.1"})
        with open("new_%s.json" % ts, 'w') as new_file:
            new_file.write(json.dumps(new.json(), indent=2))
        hot = requests.get(REDDIT_HOT_URL, headers={"User-agent": "ConceptDriftBot v0.1"})
        with open("hot_%s.json" % ts, 'w') as hot_file:
            hot_file.write(json.dumps(hot.json(), indent=2))
        time.sleep(60 * 10)

if __name__ == "__main__":
    main()

import datetime
import json
import os
import random
import string
import sys
import time
import urllib.request


class RedditPost:
    """A Reddit post from /r/gifs."""

    def __init__(self, timestamp, permalink, video_url, thumbnail_url, title):
        """Initialize a Reddit post.

        timestamp -- [datetime.datetime] Timestamp of the Reddit post.
        permalink -- [str] Permalink of the Reddit post.
        video_url -- [str] Video URL of the Reddit post.
        thumbnail_url -- [str] Thumbnail URL of the Reddit post.
        title -- [str] Title of the Reddit post.
        """
        # Copy initialization parameters.
        self._timestamp = timestamp
        self._permalink = permalink
        self._video_url = video_url
        self._thumbnail_url = thumbnail_url
        self._title = title
        # Download and store video.
        if self._video_url:
            self._video_local_path = os.path.join(sys.argv[3], permalink.split('/')[4] + "_vid")
            if not os.path.isfile(self._video_local_path):
                try:
                    urllib.request.urlretrieve(video_url, self._video_local_path)
                    time.sleep(1)
                except:
                    self._video_local_path = ""
        else:
            self._video_local_path = ""
        # Download and store thumbnail.
        if self._thumbnail_url:
            self._thumbnail_local_path = os.path.join(sys.argv[3], permalink.split('/')[4] + "_img")
            if not os.path.isfile(self._thumbnail_local_path):
                try:
                    urllib.request.urlretrieve(thumbnail_url, self._thumbnail_local_path)
                    time.sleep(1)
                except:
                    self._thumbnail_local_path = ""
        else:
            self._thumbnail_local_path = ""

    def timestamp(self):
        """Return this Reddit post's timestamp."""
        return self._timestamp

    def permalink(self):
        """Return this Reddit post's permalink."""
        return self._permalink

    def video_url(self):
        """Return this Reddit post's video URL."""
        return self._video_url

    def video_local_path(self):
        """Return the storage location of this Reddit post's video."""
        return self._video_local_path

    def thumbnail_url(self):
        """Return this Reddit post's thumbnail URL."""
        return self._thumbnail_url

    def thumbnail_local_path(self):
        """Return the storage location of this Reddit post's thumbnail."""
        return self._thumbnail_local_path

    def title(self):
        """Return this Reddit post's title."""
        return self._title


def main():
    """Parse JSON files downloaded from the Reddit website.

    $1 -- [str] Input directory containing the JSON files downloaded from the Reddit website.
    $2 -- [str] Output (pseudo) JSON file containing data to be fed into MLEP.
    $3 -- [str] Path to the directory where videos and images will be stored.
    """
    posts = {}
    hot_permalinks = set()
    for filename in os.listdir(sys.argv[1]):
        if filename.startswith("new_"):
            with open(os.path.join(sys.argv[1], filename)) as new_file:
                new_data = json.load(new_file)
                for post in new_data["data"]["children"]:
                    posts[post["data"]["permalink"]] = RedditPost(
                        datetime.datetime.strptime(
                            filename[4:filename.find('.')],
                            "%Y-%m-%d-%H:%M:%S"
                        ),
                        post["data"]["permalink"],
                        post["data"]["preview"]["reddit_video_preview"]["fallback_url"]
                            if post["data"].get("preview", None) is not None and \
                                "reddit_video_preview" in post["data"]["preview"]
                            else post["data"]["media"]["reddit_video"]["fallback_url"]
                            if post["data"].get("media", None) is not None and \
                                "reddit_video" in post["data"]["media"]
                            else "",
                        post["data"]["thumbnail"],
                        post["data"]["title"]
                    )
        elif filename.startswith("hot_"):
            with open(os.path.join(sys.argv[1], filename)) as hot_file:
                hot_data = json.load(hot_file)
                for post in hot_data["data"]["children"]:
                    hot_permalinks.add(post["data"]["permalink"])
    with open(sys.argv[2], 'w') as out_file:
        for post in posts.values():
            out_file.write("%s\n" % json.dumps({
                "timestamp": post.timestamp().strftime("%Y-%m-%d-%H:%M:%S"),
                "permalink": post.permalink(),
                "title": post.title(),
                "video_url": post.video_url(),
                "video_local_path": post.video_local_path(),
                "thumbnail_url": post.thumbnail_url(),
                "thumbnail_local_path": post.thumbnail_local_path(),
                "label": 1 if post.permalink() in hot_permalinks else 0
            }))


if __name__ == "__main__":
    main()

import os
import re
import datetime

from mitmproxy import http


PATTERN_LOCATION = re.compile(r'filepath":"(.+?)需交文件电子版')


def response(flow: http.HTTPFlow) -> None:
    if not "api/mongo/resource/normal" in flow.request.url:
        return flow
    text = flow.response.text
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    location = PATTERN_LOCATION.search(text)
    if location:
        location = location.group(1).replace("/", "_")
        path = "./data/{}.json".format(location)
    else:
        path = "./data/{}.json".format(time_str)
    if os.path.exists(path):
        return flow
    with open(path, mode="wt", encoding="utf-8") as f:
        f.write(text)
    return flow

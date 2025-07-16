import os
import platform
import json
import random
import datetime


html_template = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Task</title>
</head>
<body>
    <ul>
##ul##
    </ul>
    <pre>
##pre##
    </pre>
</body>
</html>
"""

js_template = """
// 这里填入链接
const links = [
##js:links##
];

function getDateTime() {
    var now = new Date();
    var year = now.getFullYear();
    var month = String(now.getMonth() + 1).padStart(2, "0")
    var day = String(now.getDate()).padStart(2, "0");
    var hour = String(now.getHours()).padStart(2, "0");
    var minute = String(now.getMinutes()).padStart(2, "0");
    var second = String(now.getSeconds()).padStart(2, "0");
    return `${year}.${month}.${day} ${hour}:${minute}:${second}`; 
}

function getRandomInterval() {
    return Math.floor(Math.random() * 60) + 120;
}

function openLinksSequentially(index) {
    // 所有链接打开完毕
    if (index >= links.length) {
        console.log(`[${getDateTime()}] 所有链接打开完毕`);
        setTimeout(() => {
            openLinksSequentially(index + 1);
        }, 3600 * 1000);
    }
    // 等待随机时间打开下一个
    var rand_secs = getRandomInterval() * 1000;
	if (index % 16 == 0) {
        rand_secs += 60 * 1000;
	}
    console.log(`[${getDateTime()}] ${rand_secs / 1000}s -> ${links[index]} (${index}/${links.length})`);
    setTimeout(() => {
        window.open(links[index], "_blank");
        openLinksSequentially(index + 1);
    }, rand_secs);
}

// 立即打开第一个链接
window.open(links[0], "_blank");
// 然后从第二个链接开始依次打开
openLinksSequentially(1);
"""


with open("locations.json", mode="rt", encoding="utf-8") as f:
    locations = json.load(f)

targets = []
for location_id, locations_name in locations:
    if os.path.exists(f"./data/{locations_name}.json"):
        continue
    url = "https://zhongguoyuyan.cn/point/{}".format(location_id)
    targets.append((url, locations_name))


FIRST_N_URLS = 256

random.seed(FIRST_N_URLS)
random.shuffle(targets)

body_ul = ""
body_pre = ""
for url, name in targets[:FIRST_N_URLS]:
    body_ul += f"<li><a href='{url}'><kbd>{url}</kbd> {name}</a></li>\n"
    body_pre += f'"{url}",\n'
body_ul = body_ul.rstrip("\n")
body_pre = body_pre.rstrip(",\n")
body_pre = js_template.replace("##js:links##", body_pre)

with open("task.html", mode="wt", encoding="utf-8") as f:
    html = html_template.replace("##ul##", body_ul).replace("##pre##", body_pre)
    f.write(html)


# if platform.system() == "Windows":
#     os.system("color")
# 
# print(datetime.datetime.now().strftime("\033[1;93m%Y.%m.%d %H:%M:%S\033[0m"))
# for url, name in targets[:FIRST_N_URLS]:
#     print(url, name)

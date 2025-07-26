import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

FILE = "ys.md"
URL = "https://news.cctv.com/2025/07/26/ARTIOLoDJWpA8YKX2DfKItBU250726.shtml?spm=C94212.P4YnMod9m2uD.ENPMkWvfnaiV.35"

def clean(text):
    """清理markdown文本"""
    lines = text.splitlines()
    cleaned_lines = []
    start = False
    for line in lines:
        if len(line) >= 2 and line[0] == "#" and line[1] == " ":
            cleaned_lines.append(line[2:])
        if line == "正在加载":
            start = True
            continue
        if len(line) >= 3 and line[0] == "编" and line[1] == "辑" and line[2] == "：":
            start = False
            continue
        if start:
            if not line.startswith("![]("):
                cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

async def main():
    browser_conf = BrowserConfig(headless=True)  # or False to see the browser

    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
    )

    with open(FILE, "w", encoding="utf-8") as f:
        async with AsyncWebCrawler(config=browser_conf) as crawler:
            # 爬取链接
            result = await crawler.arun(
                url=URL,
                config=run_config
            )
            # 保存到文件
            content = clean(result.markdown)
            f.write(content)
            print(f"Processed: {result.url}")

if __name__ == "__main__":
    asyncio.run(main())
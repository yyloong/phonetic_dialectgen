import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

FILE = "ys.md"
URL = "https://news.cctv.com/2025/07/26/ARTI5H7gVikaCP6VcMtHc7gs250726.shtml?spm=C94212.P4YnMod9m2uD.ENPMkWvfnaiV.14"

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
            f.write(result.markdown)
            print(f"Processed: {result.url}")

if __name__ == "__main__":
    asyncio.run(main())
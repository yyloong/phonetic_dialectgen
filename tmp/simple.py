import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

FILE = "zhihu.md"
URL = "https://www.zhihu.com/question/19731948/answer/1930918298020738153"

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
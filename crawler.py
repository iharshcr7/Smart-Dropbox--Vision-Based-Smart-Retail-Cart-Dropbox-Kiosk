from icrawler.builtin import BingImageCrawler

keywords = [
    "quality street box sealed",
    "quality street tin closed",
    "quality street chocolate box front",
    "quality street tub closed",
    "quality street box on table"
]

for i, keyword in enumerate(keywords):
    crawler = BingImageCrawler(storage={'root_dir': f'qs_{i}'})
    crawler.crawl(keyword=keyword, max_num=100)
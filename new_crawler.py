from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import urljoin
from requests.models import PreparedRequest
import requests
import json
from pathlib import Path

class ReadMangaCrawler():
    def __init__(self, options: Options = Options(), required: int = 10) -> None:
        self.START = "https://readmanga.live/list/genres/sort_name"
        self.RESTRICTED_GENRES = {"гарем", "гендерная интрига", "арт", "додзинси", "кодомо", "сёдзё-ай", "сёнэн-ай", "этти", "юри",
                     "сянься", "уся"}
        self.REQUIRED = required
        self.PAGE_SIZE = 50
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),
                          options=options)
        self.crawled = {}
        
    def _join_with_host(self, host, relative):
        return urljoin(host, relative)

    def _text_and_link(self, anchor):
        return anchor.get_attribute("title"), anchor.get_attribute("href")

    def _normalize(self, text: str):
        return text.replace("\n", "").replace(" ", "")
    
    def _add_query(self, url, params):
        req = PreparedRequest()
        req.prepare_url(url, params)
        return req.url
    
    def _get_mature(self, chapter_link):
        self.driver.get(chapter_link)
        try:
            mature = self.driver.find_element("xpath", "//a[contains(text(), 'нажмите сюда')]")
            self.driver.get(mature.get_attribute("href"))
        except:
            print("Not mature")

    def _get_genres(self):
        genre_list = self.driver.find_element("xpath", "//p[contains(@class, 'elementList')]")
        print(genre_list)
        result = [g.text for g in genre_list.find_elements("xpath", "//a[contains(@href, 'genre')]")]
        return [r for r in result if r != ""]
    
    def _get_pages(self, genre, normalized_name):
        chapter_table = self.driver.find_element("xpath", "//table")

        # no manga contents
        if chapter_table is None:
            return

        chapter_anchor = chapter_table.find_element("xpath", "//a[contains(@class, 'chapter-link')]")
        chapter_title, chapter_link = self._text_and_link(chapter_anchor)
        chapter_title = self._normalize(chapter_title)

        self._get_mature(chapter_link)

        images = [img.get_attribute("src") for img in self.driver.find_elements("xpath", "//img[@id='mangaPicture']")]


        print(images)
        print(f"fount {len(images)} images")
        for index, image_url in enumerate(images):

            if image_url == "#":
                break

            print(f"\t\tDownloading {index}/{len(images)} {image_url}", flush=True)

            try:
                response = requests.get(image_url)

                image_data = response.content

                Path(f"data/{genre}/{normalized_name}").mkdir(parents=True, exist_ok=True)
                with open(f"data/{genre}/{normalized_name}/{index}.jpg", "wb+") as fout:
                    fout.write(image_data)
            except:
                print("Error Occurred")
            
    
    def crawl_manga(self, genre, name, url):
        print(f"\t{name}")

        self.driver.get(url)

        genres = self._get_genres()

        self.crawled[genre].append((name, genres))


    
    def crawl_genre_page(self, url, page_number: int) -> list[tuple[str, str]]:
        offset = self.PAGE_SIZE * page_number

        wquery = self._add_query(url, {"sortType": "POPULARITY", "offset": offset})

        self.driver.get(wquery)

        manga_elements = self.driver.find_elements("xpath", "//div[contains(@class, 'desc')]/h3/a")

        return [self._text_and_link(m) for m in manga_elements]

    
    def crawl_genre(self, name: str, url: str):
        print(f"{name}")

        new_mangas = []
        page_number = 0
        
        while len(new_mangas) < self.REQUIRED:
            page_mangas = self.crawl_genre_page(url, page_number)

            for title, link in page_mangas:
                if title in self.visited_mangas:
                    continue
            
                self.visited_mangas.add(title)
                new_mangas.append((title, link))

                if len(new_mangas) == self.REQUIRED:
                    break
            
            page_number += 1
        
        for title, link in new_mangas:
            self.crawl_manga(name, title, link)
        
    def crawl(self) -> None:
        self.driver.get(self.START)
        self.driver.maximize_window()

        genres = [(g.get_attribute("innerHTML"), g.get_attribute("href")) for g in self.driver.find_elements("xpath", "//a[@class = 'element-link']")]

        genres = genres[0:2]
        self.visited_mangas = set()

        for g in genres:
            name, link = g
            print(name, link)

            self.crawled[name] = []

            if name not in self.RESTRICTED_GENRES:
                self.crawl_genre(name, link)


        print(self.crawled)
        
        with open('crawled_genres.txt', 'w+', encoding='utf-8') as f:
            json.dump(self.crawled, f, ensure_ascii=False)
      
options = Options()
# options.add_experimental_option("detach", True)

crawler = ReadMangaCrawler(options=options, required=1)
crawler.crawl()
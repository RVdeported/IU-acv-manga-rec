from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import urljoin
from requests.models import PreparedRequest
import requests
import json
from pathlib import Path
import time
from src.utils import split_ds

#=================================================#
# Manga Crawler class                             #
#=================================================#
# Realisation of a class for web crawling and Manga download  
# via selenium browser simulation
class ReadMangaCrawler():
    def __init__(self, 
                 options: Options = Options(), # Selenium options 
                 required: int = 10 # Required number of title per genre
                 ) -> None:
        self.START = "https://readmanga.live/list/genres/sort_name" # Starting page of the web crawler. Contains list of genres
        self.RESTRICTED_GENRES = {"сёнэн", "сёдзё", "сэйнэн", "дзёсэй"} # The list of genres we crawl
        self.REQUIRED = required 
        self.PAGE_SIZE = 50 # Average size of page (required for offsets in web crawling)
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),
                          options=options)
        self.crawled = {}

    def _text_and_link(self, anchor):
        # Helper function to get manga title and link from anchor
        return anchor.get_attribute("title"), anchor.get_attribute("href")

    def _normalize(self, text: str):
        # Helper function to normalize text
        return text.replace("\n", "").replace(" ", "")
    
    def _add_query(self, url, params):
        # Helper function to add query parameters to request
        req = PreparedRequest()
        req.prepare_url(url, params)
        return req.url
    
    def _get_mature(self, chapter_link):
        # Helper function to get around mature warnings on the web site
        # If manga has rating of 18+, it requires a manual touch of I agree to view the content button
        self.driver.get(chapter_link)
        try:
            # This may or may not appear, so we would get an exception if there is none
            # We basically need to find a button on xpath
            mature = self.driver.find_element("xpath", "//a[contains(text(), 'нажмите сюда')]")
            self.driver.get(mature.get_attribute("href"))
        except:
            print("Not mature")

    def _get_genres(self):
        # Helper function to get the list of genres with links to them
        genre_list = self.driver.find_element("xpath", "//p[contains(@class, 'elementList')]")
        print(genre_list)
        result = [g.text for g in genre_list.find_elements("xpath", "//a[contains(@href, 'genre')]")]
        return [r for r in result if r != ""]
    
    def _set_reader_settings(self):
        # Setting reader setting
        # Required because manga and webtoons load differently in selenium
        # For webtoon, all pages are available on a single screen
        # However, manga loads page by page. But changing the settings, can make manga load on a single page
        self.driver.get("https://readmanga.live/evoliuciia_bolshogo_dereva__A385fe/vol1/123")
        self.driver.maximize_window()
        # Opening settigns
        el = self.driver.find_element("xpath", "//a[@data-original-title='Настройки читалки. Масштабирование клавишей Z']")
        # Clicking
        self.driver.execute_script("arguments[0].click();", el)
        # Waiting a bit (may take time to open)
        time.sleep(1)
        # Finding the necessary setting
        el = self.driver.find_element("xpath", "//input[@value='web']")
        # Clicking the setting
        self.driver.execute_script("arguments[0].click();", el)
        # Reloading the page saves teh settigs. It is easier to use to close settings
        el = self.driver.find_element("xpath", "//i[@title='перезагрузить страницу']")
        self.driver.execute_script("arguments[0].click();", el)
    
    def _get_pages(self, genre, normalized_name):
        chapter_table = self.driver.find_element("xpath", "//table")

        # no manga contents
        if chapter_table is None:
            return
        # Finding all chapter links and normalizing names
        chapter_anchor = chapter_table.find_element("xpath", "//a[contains(@class, 'chapter-link')]")
        chapter_title, chapter_link = self._text_and_link(chapter_anchor)
        chapter_title = self._normalize(chapter_title)
        # Setting settings
        self._set_reader_settings()
        # Consideting case mature screen pops up
        self._get_mature(chapter_link)
        # Getting images
        images = [img.get_attribute("src") for img in self.driver.find_elements("xpath", "//img[@id='mangaPicture']")]

        # Printing all images found
        print(images)
        # Saving them one by one
        print(f"fount {len(images)} images")
        for index, image_url in enumerate(images[:10]):

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
        # Crawling manga individually
        self.driver.get(url)
        # Getting list of genres for manga (may be of use later) 
        genres = self._get_genres()

        self.crawled[genre].append((name, genres))
        # Crawling pages
        self._get_pages(genre, name)


    
    def crawl_genre_page(self, url, page_number: int) -> list[tuple[str, str]]:
        offset = self.PAGE_SIZE * page_number
        # Offsetting, taking 50 title names per page
        wquery = self._add_query(url, {"sortType": "POPULARITY", "offset": offset})
        # Getting titles
        self.driver.get(wquery)

        manga_elements = self.driver.find_elements("xpath", "//div[contains(@class, 'desc')]/h3/a")
        # Crawling pages
        return [self._text_and_link(m) for m in manga_elements]

    
    def crawl_genre(self, name: str, url: str):
        print(f"{name}")

        new_mangas = []
        page_number = 0
        # Crawling manga until required number of titles is crawled
        while len(new_mangas) < self.REQUIRED:
            page_mangas = self.crawl_genre_page(url, page_number)
            # Sometimes genres overlap, so we need to skip visited title
            for title, link in page_mangas:
                if title in self.visited_mangas:
                    continue
                # Adding the visited manga to the list
                self.visited_mangas.add(title)
                new_mangas.append((title, link))
                # There may be way more than required titles per genre and we need to stop
                if len(new_mangas) == self.REQUIRED:
                    break
            
            page_number += 1
        
        for title, link in new_mangas:
            self.crawl_manga(name, title, link)
        
    def crawl(self) -> None:
        # Getting starter page
        self.driver.get(self.START)
        # Making fullscreen
        self.driver.maximize_window()
        # Finding all genres
        genres = [(g.get_attribute("innerHTML"), g.get_attribute("href")) for g in self.driver.find_elements("xpath", "//a[@class = 'element-link']")]

        self.visited_mangas = set()

        for g in genres:
            name, link = g
            print(name, link)

            self.crawled[name] = []
            # Crawling only the required genres
            if name in self.RESTRICTED_GENRES:
                self.crawl_genre(name, link)


        print(self.crawled)
        
        with open('crawled_genres.txt', 'w+', encoding='utf-8') as f:
            json.dump(self.crawled, f, ensure_ascii=False)
      
options = Options()
# options.add_experimental_option("detach", True)

crawler = ReadMangaCrawler(options=options, required=30)
crawler.crawl()

# split_ds('data', 'data')
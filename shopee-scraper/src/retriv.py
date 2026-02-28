#!/usr/bin/env python3
# retriv.py - ShopeeScraper (fixed for Streamlit)

import os
import sys
import time
import json
import pickle
import logging
import re
import datetime
import threading
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By

class ShopeeScraper:
    def __init__(self, search_term, max_products, index_only,
                 review_limit,
                 all_star_types=False,
                 star_limit_per_type=0,
                 chrome_user_data_dir=None):

        self.all_star_types = all_star_types
        self.star_limit_per_type = star_limit_per_type
        self.driver = None
        self.cookies_file = 'cookies_shopee.dat'
        self.search_term = search_term
        self.max_products = max_products
        self.index_only = index_only
        self.review_limit = review_limit
        self.chrome_user_data_dir = chrome_user_data_dir

        self.stop_requested = False
        self.start_time = time.time()
        self.total_pages_scraped = 0

        # setup logging
        self._setup_logging()
        self.options = uc.ChromeOptions()
        self._configure_options()

        self.output_data = {}
        self.out_file = f"shopee_{re.sub(r'[^a-z0-9_]+', '', self.search_term.lower())}.json"
        self._load_existing_data()

        # register signal handler only if main thread
        if threading.current_thread() == threading.main_thread():
            self._register_signal()

    # -------------------------
    # SIGNAL HANDLER
    # -------------------------
    def _register_signal(self):
        try:
            import signal
            signal.signal(signal.SIGINT, self._handle_interrupt)
        except Exception as e:
            logging.warning(f"Signal handler gagal diatur: {e}")

    def _handle_interrupt(self, signum, frame):
        print("\n⛔ CTRL+C detected. Finishing safely...")
        self.stop_requested = True

    # -------------------------
    # LOGGING
    # -------------------------
    def _setup_logging(self):
        os.makedirs("logs", exist_ok=True)
        log_filename = datetime.datetime.now().strftime("shopee_%d_%m_%H_%M_%S.log")
        log_filepath = os.path.join("logs", log_filename)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(log_filepath), logging.StreamHandler(sys.stdout)]
        )

    # -------------------------
    # CHROME OPTIONS
    # -------------------------
    def _configure_options(self):
        if sys.platform.startswith('linux'):
            self.options.add_argument("--disable-gpu")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument("--disable-blink-features=AutomationControlled")
        self.options.add_argument("--start-maximized")

    # -------------------------
    # COOKIE
    # -------------------------
    def _save_cookies(self):
        try:
            with open(self.cookies_file, 'wb') as file:
                pickle.dump(self.driver.get_cookies(), file)
        except:
            pass

    def _load_cookies(self):
        if os.path.exists(self.cookies_file):
            with open(self.cookies_file, 'rb') as file:
                cookies = pickle.load(file)
            for cookie in cookies:
                try:
                    self.driver.add_cookie(cookie)
                except:
                    continue

    # -------------------------
    # DATA
    # -------------------------
    def _load_existing_data(self):
        if os.path.exists(self.out_file):
            try:
                with open(self.out_file, 'r', encoding='utf-8') as f:
                    self.output_data = {
                        item['link']: item
                        for item in json.load(f)
                        if 'link' in item
                    }
                logging.info(f"Loaded {len(self.output_data)} existing products")
            except:
                self.output_data = {}

    def _periodic_save(self):
        try:
            with open(self.out_file, 'w', encoding='utf-8') as f:
                json.dump(
                    list(self.output_data.values()),
                    f,
                    ensure_ascii=False,
                    indent=2
                )
        except Exception as e:
            logging.error(f"Save error: {e}")

    # -------------------------
    # LOGIN / CAPTCHA
    # -------------------------
    def _wait_for_login(self):
        self.driver.get("https://shopee.co.id/")
        time.sleep(3)

        if os.path.exists(self.cookies_file):
            logging.info("Loading cookies...")
            self._load_cookies()
            self.driver.refresh()
            time.sleep(3)
            if "login" not in self.driver.current_url.lower():
                logging.info("Session valid.")
                return

        print("\n=== LOGIN REQUIRED ===")
        print("1. Login Shopee")
        print("2. Selesaikan captcha")
        print("3. Pastikan sudah masuk homepage")
        input("Tekan ENTER jika sudah login...")
        time.sleep(3)
        self._save_cookies()

    def _check_captcha(self):
        blacklist = ["login", "captcha", "verify", "security"]
        if any(x in self.driver.current_url.lower() for x in blacklist):
            input("Captcha detected. Solve then press ENTER...")
            return True
        return False

    def _safe_get(self, url):
        self.driver.get(url)
        time.sleep(3)
        while self._check_captcha():
            self.driver.get(url)
            time.sleep(3)

    # -------------------------
    # SCRAPE
    # -------------------------
    def _retrieve_products(self):
        result = []
        items = self.driver.find_elements(By.CSS_SELECTOR, "li.col-xs-2-4")

        for li in items:
            if self.stop_requested:
                break
            try:
                link = li.find_element(By.TAG_NAME, "a").get_attribute("href")
                if not link:
                    continue

                name = price = rating = location = img = ""
                try: name = li.find_element(By.CSS_SELECTOR, "div._10Wbs-").text
                except: pass
                try: price = li.find_element(By.CSS_SELECTOR, "div._1w9jLI").text
                except: pass
                try: rating = li.find_element(By.CSS_SELECTOR, "div._3LWZlK").text
                except: pass
                try: location = li.find_element(By.CSS_SELECTOR, "div._2E5i1q").text
                except: pass
                try: img = li.find_element(By.TAG_NAME, "img").get_attribute("src")
                except: pass

                result.append({
                    "link": link,
                    "name": name,
                    "price": price,
                    "rating": rating,
                    "location": location,
                    "img": img
                })
            except:
                continue
        return result

    def _scrape_page(self):
        logging.info("Starting auto pagination...")
        base_url = "https://shopee.co.id/search?keyword="
        kw = re.sub(r'\s+', '%20', self.search_term.strip())
        page = 0

        while len(self.output_data) < self.max_products and not self.stop_requested:
            url = f"{base_url}{kw}&page={page}&sortBy=sales"
            logging.info(f"Scraping page {page}")
            self._safe_get(url)
            self.total_pages_scraped += 1

            # Scroll
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            for _ in range(5):
                if self.stop_requested:
                    break
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height

            products = self._retrieve_products()
            if not products:
                break

            for p in products:
                if self.stop_requested:
                    break
                if p["link"] not in self.output_data:
                    self.output_data[p["link"]] = p
                    self._periodic_save()

                if len(self.output_data) >= self.max_products:
                    break

            page += 1
            if page > 50:
                break

    # -------------------------
    # EXECUTE
    # -------------------------
    def execute(self):
        try:
            self.driver = uc.Chrome(options=self.options, headless=False, version_main=145)
            self.driver.maximize_window()
            self._wait_for_login()
            self._scrape_page()
        finally:
            if self.driver:
                self._save_cookies()
                self._periodic_save()
                self.driver.quit()

            duration = round(time.time() - self.start_time, 2)
            print("\n====================================")
            print("SCRAPING SELESAI ✅")
            print(f"Keyword        : {self.search_term}")
            print(f"Total Produk   : {len(self.output_data)}")
            print(f"Total Halaman  : {self.total_pages_scraped}")
            print(f"Durasi         : {duration} detik")
            print(f"File Output    : {self.out_file}")
            print("====================================\n")
            logging.info("Scraper stopped safely.")


# -------------------------
# MAIN (standalone run)
# -------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--keyword", default="Laptop Lenovo Acer")
    parser.add_argument("-n", "--num", type=int, default=100)
    parser.add_argument("--index-only", action="store_true", default=False)
    parser.add_argument("-r", "--review-limit", type=int, default=100)
    args = parser.parse_args()

    scraper = ShopeeScraper(
        args.keyword,
        args.num,
        args.index_only,
        args.review_limit
    )
    scraper.execute()
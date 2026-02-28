#!/usr/bin/env python3
# app.py - Gabungan: CSV reader (user file) + Tokopedia & Shopee scrapers + CSI analysis + product matching
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import time
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import sys
import logging
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin
import string
from difflib import SequenceMatcher

# --- ensure retriv path discoverable (optional) ---
repo_path = Path(__file__).resolve().parent / "shopee-scraper" / "src"
if str(repo_path) not in sys.path:
    sys.path.append(str(repo_path))

# --- optional imports ---
SHOPEE_AVAILABLE = False
try:
    from retriv import ShopeeScraper
    SHOPEE_AVAILABLE = True
except Exception:
    SHOPEE_AVAILABLE = False

TOKOPEDIA_AVAILABLE = False
try:
    from tokopaedi import search as tp_search
    TOKOPEDIA_AVAILABLE = True
except Exception:
    TOKOPEDIA_AVAILABLE = False

# --- Streamlit page config ---
st.set_page_config(page_title="CSI Laptop Multi-Platform", layout="wide")

# --- CSS ---
st.markdown("""
<style>
.interpretation-card { padding: 15px; border-radius: 10px; text-align: center; color: white; font-weight: bold; font-size: 20px; margin-top: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}
.thumbnail-card { text-align:center; font-size:12px; margin-bottom:8px; }
.small-note { font-size:12px; color:#666; }
.conclusion-box { background-color: #000000; padding: 20px; border-radius: 12px; color: #fff; }
</style>
""", unsafe_allow_html=True)

# --- Helpers (shared) ---
def format_rupiah(val):
    try:
        return f"Rp{val:,.0f}".replace(",", ".")
    except:
        return str(val)

def get_status_info(score):
    if score >= 81: return "SANGAT PUAS", "#28a745"
    elif score >= 66: return "PUAS", "#007bff"
    elif score >= 51: return "CUKUP PUAS", "#ffc107"
    else: return "TIDAK PUAS", "#dc3545"

def parse_sold_num(text):
    if pd.isna(text): return 0
    text = str(text).strip().lower()
    text = text.replace('.', '').replace(' ', '')
    m = re.match(r'^([\d,]+)(k|rb)?', text)
    if m:
        num, unit = m.groups()
        num = num.replace(',', '.')
        try:
            val = float(num)
        except:
            return 0
        if unit in ['k', 'rb']:
            val *= 1000
        return int(val)
    digits = re.findall(r'\d+', text)
    if digits:
        try:
            return int(digits[0])
        except:
            return 0
    return 0

def _normalize_img_url(img):
    if not img or str(img).strip() == '':
        return None
    img = str(img).strip()
    if img.startswith('//'):
        return 'https:' + img
    if img.startswith('/'):
        return 'https://shopee.co.id' + img
    return img

# --- Primary process_data (used by scrapers) ---
def process_data(df, platform_name='unknown'):
    for col in ['product_name','product_price','product_rating','product_sold','product_img']:
        if col not in df.columns:
            df[col] = np.nan

    df['product_rating'] = pd.to_numeric(df['product_rating'], errors='coerce')
    if df['product_rating'].notna().any():
        median_rating = df['product_rating'].median()
    else:
        median_rating = 4.0
    df['product_rating'] = df['product_rating'].fillna(median_rating)

    df['sold_num'] = df['product_sold'].apply(parse_sold_num)

    df['price_clean'] = pd.to_numeric(
        df['product_price'].astype(str).replace(r'[^\d,.-]', '', regex=True)
            .str.replace(',', '.', regex=False)
            .replace(r'^\s*[-–—]\s*$', '0', regex=True),
        errors='coerce'
    ).fillna(0)

    r_min, r_max = df['product_rating'].min(), df['product_rating'].max()
    if pd.isna(r_min) or pd.isna(r_max) or r_max == r_min:
        df['rating_norm'] = 0.5
    else:
        df['rating_norm'] = (df['product_rating'] - r_min) / (r_max - r_min)

    df['vfm_score'] = 0
    mask = df['price_clean'] > 0
    df.loc[mask, 'vfm_score'] = (df.loc[mask, 'product_rating'] / (df.loc[mask, 'price_clean'] / 1000000)).replace([np.inf, -np.inf], 0)

    bins = [0, 7000000, 15000000, 1000000000]
    labels = ['Budget (<7jt)', 'Mid-Range (7-15jt)', 'Premium (>15jt)']
    df['Segmen Harga'] = pd.cut(df['price_clean'], bins=bins, labels=labels, include_lowest=True)

    df['platform'] = platform_name
    df['ws'] = 0
    if 'product_img' in df.columns:
        df['product_img'] = df['product_img'].apply(_normalize_img_url)
    return df

# --- CSV-specific process_data (from user's small file) ---
def process_data_from_csvfile(df_raw):
    """
    This function follows the CSV-reading logic you provided:
    - skiprows=2 is handled at read_csv call
    - minimal cleaning and normalization similar to the small CSV example
    """
    df = df_raw.copy()
    # ensure expected columns exist (fallback names)
    # Accept both 'product_price' or 'Harga Asli' etc.
    col_map = {}
    # normalize column names to expected keys if possible
    lower_cols = {c.lower(): c for c in df.columns}
    if 'product_price' not in df.columns and 'harga asli' in lower_cols:
        col_map[lower_cols['harga asli']] = 'product_price'
    if 'product_name' not in df.columns and 'nama produk' in lower_cols:
        col_map[lower_cols['nama produk']] = 'product_name'
    if 'product_rating' not in df.columns and 'rating produk' in lower_cols:
        col_map[lower_cols['rating produk']] = 'product_rating'
    if 'product_sold' not in df.columns and 'jumlah terjual' in lower_cols:
        col_map[lower_cols['jumlah terjual']] = 'product_sold'
    if col_map:
        df = df.rename(columns=col_map)

    # Fill missing expected columns
    for col in ['product_name','product_price','product_rating','product_sold','product_img','platform']:
        if col not in df.columns:
            df[col] = np.nan

    # Convert rating
    df['product_rating'] = pd.to_numeric(df['product_rating'], errors='coerce')
    # If rating missing, fill with median or default 4.0
    if df['product_rating'].notna().any():
        df['product_rating'] = df['product_rating'].fillna(df['product_rating'].median())
    else:
        df['product_rating'] = df['product_rating'].fillna(4.0)

    # sold parsing
    df['sold_num'] = df['product_sold'].apply(parse_sold_num)

    # price cleaning
    df['price_clean'] = pd.to_numeric(
        df['product_price'].astype(str).replace(r'[Rp,."\s]', '', regex=True),
        errors='coerce'
    ).fillna(0)

    # rating normalization
    r_min, r_max = df['product_rating'].min(), df['product_rating'].max()
    if r_max > r_min:
        df['rating_norm'] = (df['product_rating'] - r_min) / (r_max - r_min)
    else:
        df['rating_norm'] = 0.5

    # vfm
    df['vfm_score'] = (df['product_rating'] / (df['price_clean'] / 1000000)).replace([np.inf, -np.inf], 0)

    bins = [0, 7000000, 15000000, 1000000000]
    labels = ['Budget (<7jt)', 'Mid-Range (7-15jt)', 'Premium (>15jt)']
    df['Segmen Harga'] = pd.cut(df['price_clean'], bins=bins, labels=labels, include_lowest=True)

    # platform: if missing, set to CSV
    if 'platform' not in df.columns or df['platform'].isna().all():
        df['platform'] = 'csv'
    else:
        df['platform'] = df['platform'].astype(str).str.lower()

    df['product_img'] = df.get('product_img', np.nan).apply(_normalize_img_url)
    df['product_name'] = df['product_name'].astype(str)
    return df

# --- Utility: show thumbnails grid ---
def show_thumbnails(df, cols_per_row=4, thumb_w=150, max_items=12):
    if df is None or df.empty:
        return
    rows = df.head(max_items)
    cols = st.columns(cols_per_row)
    for i, (_, r) in enumerate(rows.iterrows()):
        col = cols[i % cols_per_row]
        img = r.get("product_img", None)
        name = r.get("product_name", "")[:80]
        price = r.get("product_price", "")
        if img:
            try:
                col.image(img, width=thumb_w)
            except Exception:
                col.write("No image")
        else:
            col.write("No image")
        col.markdown(f"<div class='thumbnail-card'><b>{name}</b><br>{price}</div>", unsafe_allow_html=True)

# --- Scraper wrappers & fallback scrapers (unchanged from earlier) ---
def fetch_tokopedia_via_lib(query: str, limit: int = 50, ui_status=None, ui_progress=None):
    try:
        if ui_status: ui_status.text("Menggunakan library tokopaedi...")
        res = tp_search(query, max_result=limit)
        if hasattr(res, "json"):
            items = res.json()
        else:
            items = res
        if not items:
            if ui_status: ui_status.text("Tokopaedi: tidak ada hasil dari library.")
            if ui_progress: ui_progress.progress(0)
            return pd.DataFrame()
        df_temp = pd.DataFrame(items)
        rename_map = {}
        if 'price' in df_temp.columns: rename_map['price'] = 'product_price'
        if 'rating' in df_temp.columns: rename_map['rating'] = 'product_rating'
        if 'sold' in df_temp.columns: rename_map['sold'] = 'product_sold'
        if 'main_image' in df_temp.columns:
            rename_map['main_image'] = 'product_img'
        elif 'img' in df_temp.columns:
            rename_map['img'] = 'product_img'
        elif 'image' in df_temp.columns:
            rename_map['image'] = 'product_img'
        elif 'images' in df_temp.columns:
            df_temp['product_img'] = df_temp['images'].apply(lambda x: x[0] if isinstance(x, (list, tuple)) and x else (x if isinstance(x, str) else ""))
        df_temp = df_temp.rename(columns=rename_map)
        if 'product_img' in df_temp.columns:
            df_temp['product_img'] = df_temp['product_img'].astype(str).replace('nan', '')
        if ui_progress: ui_progress.progress(100)
        if ui_status: ui_status.text("Tokopaedi: selesai.")
        return df_temp.head(limit)
    except Exception as e:
        logging.warning(f"fetch_tokopedia_via_lib failed: {e}")
        if ui_status: ui_status.text(f"Tokopaedi error: {e}")
        if ui_progress: ui_progress.progress(0)
        return pd.DataFrame()

def fetch_shopee_via_class(query: str, limit: int = 50, index_only: bool = True,
                           chrome_user_data_dir: str = None, profile_directory: str = None,
                           ui_status=None, ui_progress=None, wait_for_user_click=False):
    def _run():
        try:
            repo_path = Path(__file__).resolve().parent / "shopee-scraper" / "src"
            if str(repo_path) not in sys.path:
                sys.path.append(str(repo_path))
            from retriv import ShopeeScraper
        except Exception as e:
            logging.warning(f"Cannot import retriv.ShopeeScraper: {e}")
            if ui_status: ui_status.text("Error: retriv module not found")
            if ui_progress: ui_progress.progress(0)
            return pd.DataFrame()

        try:
            if ui_status: ui_status.text("Mempersiapkan scraper...")
            if ui_progress: ui_progress.progress(5)

            scraper = ShopeeScraper(
                search_term=query,
                max_products=limit,
                index_only=index_only,
                review_limit=0,
                all_star_types=False,
                star_limit_per_type=0,
                chrome_user_data_dir=chrome_user_data_dir
            )
            if profile_directory:
                try:
                    scraper.options.add_argument(f"--profile-directory={profile_directory}")
                except Exception:
                    pass

            if ui_status: ui_status.text("Menjalankan scraper (Chrome akan terbuka)...")
            if ui_progress: ui_progress.progress(20)

            scraper.execute()

            if ui_status: ui_status.text("Mengumpulkan output dari scraper...")
            if ui_progress: ui_progress.progress(50)

            items = []
            try:
                out_data = getattr(scraper, "output_data", {}) or {}
                total = len(out_data) if out_data else 0
                processed = 0
                for link, item in out_data.items():
                    items.append({
                        "product_name": item.get("name", ""),
                        "product_price": item.get("price", ""),
                        "product_rating": item.get("rating", ""),
                        "product_sold": item.get("sold", ""),
                        "product_location": item.get("location", ""),
                        "product_img": item.get("img", "")
                    })
                    processed += 1
                    if ui_progress and total:
                        ui_progress.progress(50 + int(40 * (processed / total)))
            except Exception:
                items = []

            if not items:
                out_file = getattr(scraper, "out_file", None)
                if out_file and os.path.exists(out_file):
                    try:
                        with open(out_file, "r", encoding="utf-8") as f:
                            raw = json.load(f)
                        for item in raw[:limit]:
                            items.append({
                                "product_name": item.get("name", ""),
                                "product_price": item.get("price", ""),
                                "product_rating": item.get("rating", ""),
                                "product_sold": item.get("sold", ""),
                                "product_location": item.get("location", ""),
                                "product_img": item.get("img", "")
                            })
                    except Exception as e:
                        logging.warning(f"Failed reading out_file {out_file}: {e}")

            if ui_status: ui_status.text("Selesai.")
            if ui_progress: ui_progress.progress(100)

            if not items:
                return pd.DataFrame()
            return pd.DataFrame(items).head(limit)
        except Exception as e:
            logging.warning(f"fetch_shopee_via_class failed: {e}")
            if ui_status: ui_status.text(f"Error saat scraping: {e}")
            if ui_progress: ui_progress.progress(0)
            return pd.DataFrame()

    if wait_for_user_click:
        return _run
    else:
        return _run()

# --- Fallback parallel scrapers (requests + BeautifulSoup) ---
def _get_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"})
    return s

def _parse_shopee_card(c):
    try:
        img = ""
        img_tag = c.select_one("img")
        if img_tag:
            img = img_tag.get("src") or img_tag.get("data-src") or img_tag.get("data-image") or img_tag.get("data-srcset")
        name_el = c.select_one("div._10Wbs-._5SSWfi.UjjMrh, a")
        name = name_el.get_text(strip=True) if name_el else (img_tag.get("alt") if img_tag else "")
        price_el = c.select_one("div._1w9jLI.QbH7Ig._1f1QKk, span")
        price = price_el.get_text(strip=True) if price_el else ""
        rating_el = c.select_one("div._3Oj5_n, span._3Oj5_n")
        rating = rating_el.get_text(strip=True) if rating_el else ""
        sold_el = c.select_one("div._18SLBt, span._18SLBt")
        sold = sold_el.get_text(strip=True) if sold_el else ""
        return {"product_name": name, "product_price": price, "product_rating": rating, "product_sold": sold, "product_img": img}
    except Exception:
        return None

def fetch_shopee_simple_parallel(query: str, limit: int = 50, ui_status=None, ui_progress=None, max_workers=4):
    session = _get_session()
    base_url = f"https://shopee.co.id/search?keyword={requests.utils.requote_uri(query)}"
    try:
        if ui_status: ui_status.text("Mengambil halaman Shopee (parallel)...")
        if ui_progress: ui_progress.progress(0)
        resp = session.get(base_url, timeout=12)
        if resp.status_code != 200:
            if ui_status: ui_status.text("Shopee fallback: respons tidak 200")
            if ui_progress: ui_progress.progress(0)
            return pd.DataFrame()
        soup = BeautifulSoup(resp.text, "html.parser")
        cards = soup.select("div.shopee-search-item-result__item, div._1NoI8_")
        results = []
        for c in cards:
            parsed = _parse_shopee_card(c)
            if parsed:
                results.append(parsed)
            if len(results) >= limit:
                break
        if len(results) < limit:
            pages_needed = min(5, (limit // 20) + 2)
            urls = [f"{base_url}&page={p}" for p in range(2, pages_needed + 1)]
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(session.get, u, timeout=12): u for u in urls}
                processed = 0
                for f in as_completed(futures):
                    try:
                        r = f.result()
                        if r.status_code != 200:
                            continue
                        s = BeautifulSoup(r.text, "html.parser")
                        cards2 = s.select("div.shopee-search-item-result__item, div._1NoI8_")
                        for c in cards2:
                            parsed = _parse_shopee_card(c)
                            if parsed:
                                results.append(parsed)
                            if len(results) >= limit:
                                break
                        processed += 1
                        if ui_progress:
                            ui_progress.progress(min(100, 50 + int(50 * (processed / len(futures)))))
                        if len(results) >= limit:
                            break
                    except Exception:
                        continue
        if ui_status: ui_status.text("Shopee fallback: selesai.")
        if ui_progress: ui_progress.progress(100)
        if not results:
            return pd.DataFrame()
        return pd.DataFrame(results).head(limit)
    except Exception as e:
        logging.warning(f"fetch_shopee_simple_parallel failed: {e}")
        if ui_status: ui_status.text("Shopee fallback: error")
        if ui_progress: ui_progress.progress(0)
        return pd.DataFrame()

def _parse_tokopedia_card(c):
    try:
        img = ""
        img_tag = c.select_one("img")
        if img_tag:
            img = img_tag.get("src") or img_tag.get("data-src") or img_tag.get("data-image")
        name_el = c.select_one("a")
        name = name_el.get_text(strip=True) if name_el else ""
        price_el = c.select_one("div.css-rhd610, span[data-testid='spnSRPProdPrice']")
        price = price_el.get_text(strip=True) if price_el else ""
        rating_el = c.select_one("span.css-1f4mp12, span[data-testid='spnSRPProdRating']")
        rating = rating_el.get_text(strip=True) if rating_el else ""
        sold_el = c.select_one("div.css-1s8d1b6, span[data-testid='spnSRPProdSold']")
        sold = sold_el.get_text(strip=True) if sold_el else ""
        return {"product_name": name, "product_price": price, "product_rating": rating, "product_sold": sold, "product_img": img}
    except Exception:
        return None

def fetch_tokopedia_simple_parallel(query: str, limit: int = 50, ui_status=None, ui_progress=None, max_workers=4):
    session = _get_session()
    per_page = 20
    pages = max(1, (limit + per_page - 1) // per_page)
    urls = [f"https://www.tokopedia.com/search?st=product&q={requests.utils.requote_uri(query)}&page={p}" for p in range(1, pages+1)]
    results = []
    try:
        if ui_status: ui_status.text("Mengambil Tokopedia (parallel)...")
        if ui_progress: ui_progress.progress(0)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(session.get, u, timeout=12): u for u in urls}
            processed = 0
            for f in as_completed(futures):
                try:
                    r = f.result()
                    if r.status_code != 200:
                        continue
                    s = BeautifulSoup(r.text, "html.parser")
                    cards = s.select("div.css-1f4mp12, div.css-1b6t4dn, div[data-testid='lstCL2ProductList'] a")
                    for c in cards:
                        parsed = _parse_tokopedia_card(c)
                        if parsed:
                            results.append(parsed)
                        if len(results) >= limit:
                            break
                    processed += 1
                    if ui_progress:
                        ui_progress.progress(min(100, int(100 * (processed / len(futures)))))
                    if len(results) >= limit:
                        break
                except Exception:
                    continue
        if ui_status: ui_status.text("Tokopedia: selesai.")
        if ui_progress: ui_progress.progress(100)
        if not results:
            return pd.DataFrame()
        return pd.DataFrame(results).head(limit)
    except Exception as e:
        logging.warning(f"fetch_tokopedia_simple_parallel failed: {e}")
        if ui_status: ui_status.text("Tokopedia fallback: error")
        if ui_progress: ui_progress.progress(0)
        return pd.DataFrame()

# --- Sidebar & UI inputs ---
st.title("📊 Analisis CSI Laptop Multi-Platform")
st.subheader("Upload CSV atau Scraping Live (Tokopedia & Shopee)")

with st.sidebar:
    source_option = st.radio("Pilih Sumber Data:", ["CSV","Scraping Live"])
    file = None
    fetch_btn = False
    default_user_data = None
    chrome_user_data_dir = st.text_input("Chrome user-data-dir (folder User Data)", default_user_data)
    profile_directory = st.text_input("Profile directory (mis. Default atau Profile 1)", "Profile 1")
    if source_option == "CSV":
        # Use the user's CSV-reading behavior: allow skiprows=2 by default (user file may have header lines)
        file = st.file_uploader("Unggah CSV (jika file memiliki 2 baris header, centang opsi di bawah)", type="csv")
        skip_two = st.checkbox("CSV memiliki 2 baris header (gunakan skiprows=2)", value=False)
    else:
        keyword = st.text_input("Masukkan keyword produk", "Laptop Lenovo")
        limit = st.slider("Jumlah Produk per Merk", 10, 200, 50, 10)
        platforms_to_fetch = st.multiselect("Pilih platform untuk scraping", ["Tokopedia", "Shopee"], default=["Tokopedia","Shopee"])
        max_workers = st.slider("Max parallel workers (fallback)", 1, 8, 4)
        prefer_retriv = st.checkbox("Prefer retriv Shopee scraper (may open Chrome / slower)", value=False)
        fetch_btn = st.button("Fetch Data")
    st.markdown("---")
    st.markdown("Catatan: Tutup semua jendela Chrome sebelum menjalankan scraper jika menggunakan profil Chrome. Jika captcha muncul, selesaikan secara manual di browser yang terbuka.")

# --- Main flow variables ---
df = pd.DataFrame()
df_tokopedia = pd.DataFrame()
df_shopee = pd.DataFrame()

# --- CSV branch (integrated with user's CSV logic) ---
if source_option == "CSV" and file is not None:
    try:
        # read CSV with optional skiprows=2 as user file indicated
        if 'skip_two' in locals() and skip_two:
            df_raw = pd.read_csv(file, skiprows=2, skip_blank_lines=True)
        else:
            df_raw = pd.read_csv(file)
        # Use the CSV-specific processor to normalize columns
        df_csv = process_data_from_csvfile(df_raw)
        # ensure platform column normalized
        df_csv['platform'] = df_csv['platform'].astype(str).str.lower()
        df = df_csv.copy()
        st.success(f"Data CSV berhasil dimuat: {len(df)} produk")
        st.markdown("**Preview CSV (Top 10)**")
        st.dataframe(df.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"Gagal membaca CSV: {e}")

# --- Scraping branch (paralel Tokopedia + Shopee) ---
if source_option == "Scraping Live" and fetch_btn:
    status_tok = st.empty()
    prog_tok = st.progress(0)
    status_sp = st.empty()
    prog_sp = st.progress(0)

    def worker_tokopedia(keyword, limit, max_workers, chrome_user_data_dir, profile_directory):
        try:
            if TOKOPEDIA_AVAILABLE:
                df_tp = fetch_tokopedia_via_lib(keyword, limit=limit, ui_status=None, ui_progress=None)
            else:
                df_tp = fetch_tokopedia_simple_parallel(keyword, limit=limit, ui_status=None, ui_progress=None, max_workers=max_workers)
            if df_tp is None or df_tp.empty:
                return {"df": pd.DataFrame(), "error": None}
            df_proc = process_data(df_tp, platform_name='tokopedia')
            return {"df": df_proc, "error": None}
        except Exception as e:
            logging.exception("Tokopedia worker error")
            return {"df": pd.DataFrame(), "error": str(e)}

    def worker_shopee(keyword, limit, max_workers, chrome_user_data_dir, profile_directory, prefer_retriv):
        try:
            if prefer_retriv and SHOPEE_AVAILABLE:
                df_sp = fetch_shopee_via_class(keyword, limit=limit, index_only=True,
                                               chrome_user_data_dir=chrome_user_data_dir,
                                               profile_directory=profile_directory,
                                               ui_status=None, ui_progress=None,
                                               wait_for_user_click=False)
            else:
                df_sp = fetch_shopee_simple_parallel(keyword, limit=limit, ui_status=None, ui_progress=None, max_workers=max_workers)
            if df_sp is None or df_sp.empty:
                return {"df": pd.DataFrame(), "error": None}
            df_proc = process_data(df_sp, platform_name='shopee')
            return {"df": df_proc, "error": None}
        except Exception as e:
            logging.exception("Shopee worker error")
            return {"df": pd.DataFrame(), "error": str(e)}

    if "Tokopedia" in platforms_to_fetch:
        status_tok.text("Menyiapkan Tokopedia...")
        prog_tok.progress(0)
    if "Shopee" in platforms_to_fetch:
        status_sp.text("Menyiapkan Shopee...")
        prog_sp.progress(0)

    futures = {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        if "Tokopedia" in platforms_to_fetch:
            futures["tokopedia"] = ex.submit(worker_tokopedia, keyword, limit, max_workers, chrome_user_data_dir, profile_directory)
        if "Shopee" in platforms_to_fetch:
            futures["shopee"] = ex.submit(worker_shopee, keyword, limit, max_workers, chrome_user_data_dir, profile_directory, prefer_retriv)

        for name, fut in futures.items():
            try:
                res = fut.result(timeout=300)
            except Exception as e:
                res = {"df": pd.DataFrame(), "error": str(e)}

            if name == "tokopedia":
                if res.get("error"):
                    status_tok.text(f"Tokopedia error: {res['error']}")
                    prog_tok.progress(0)
                elif res["df"].empty:
                    status_tok.text("Tokopedia: tidak ada data yang diambil")
                    prog_tok.progress(0)
                else:
                    df_tokopedia = res["df"]
                    status_tok.text(f"Tokopedia: {len(df_tokopedia)} produk diambil")
                    prog_tok.progress(100)
                    st.markdown("**Preview Tokopedia (Top 10)**")
                    st.dataframe(df_tokopedia.head(10), use_container_width=True)
                    st.markdown("**Produk Tokopedia**")
                    show_thumbnails(df_tokopedia, cols_per_row=4, thumb_w=140, max_items=12)

            if name == "shopee":
                if res.get("error"):
                    status_sp.text(f"Shopee error: {res['error']}")
                    prog_sp.progress(0)
                elif res["df"].empty:
                    if prefer_retriv and SHOPEE_AVAILABLE:
                        status_sp.text("Shopee: retriv tersedia, jalankan retriv jika ingin menggunakan browser-based scraper.")
                    else:
                        status_sp.text("Shopee: tidak ada data yang diambil")
                    prog_sp.progress(0)
                else:
                    df_shopee = res["df"]
                    status_sp.text(f"Shopee: {len(df_shopee)} produk diambil")
                    prog_sp.progress(100)
                    st.markdown("**Preview Shopee (Top 10)**")
                    st.dataframe(df_shopee.head(10), use_container_width=True)
                    st.markdown("**Produk Shopee**")
                    show_thumbnails(df_shopee, cols_per_row=4, thumb_w=140, max_items=12)

    frames = []
    if not df_tokopedia.empty:
        frames.append(df_tokopedia)
    if not df_shopee.empty:
        frames.append(df_shopee)
    if frames:
        df = pd.concat(frames, ignore_index=True)
        st.success(f"Total produk gabungan: {len(df)}")
    else:
        st.error("Tidak ada data berhasil diambil dari platform yang dipilih")

# --- Normalize platform column early for downstream checks ---
if not df.empty:
    if 'platform' not in df.columns:
        df['platform'] = 'unknown'
    df['platform'] = df['platform'].astype(str).str.lower()
    platforms = sorted(df['platform'].unique())
else:
    platforms = []

# --- Product matching (fuzzy) ---
try:
    from rapidfuzz import fuzz
    _HAS_RAPIDFUZZ = True
except Exception:
    _HAS_RAPIDFUZZ = False

def _normalize_text(s):
    if s is None:
        return ""
    s = str(s).lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = " ".join(s.split())
    return s

def _name_similarity(a, b):
    a_n = _normalize_text(a)
    b_n = _normalize_text(b)
    if _HAS_RAPIDFUZZ:
        return fuzz.token_set_ratio(a_n, b_n)
    else:
        return int(SequenceMatcher(None, a_n, b_n).ratio() * 100)

def match_products(df_a, df_b,
                   name_weight=0.7,
                   price_weight=0.3,
                   name_threshold=65,
                   price_tolerance_pct=0.35,
                   top_k=1):
    if df_a is None or df_b is None or df_a.empty or df_b.empty:
        return pd.DataFrame()
    for col in ['product_name','price_clean','product_img','platform']:
        if col not in df_a.columns:
            df_a[col] = ""
        if col not in df_b.columns:
            df_b[col] = ""
    rows = []
    df_b = df_b.copy().reset_index(drop=True)
    df_b['_norm_name'] = df_b['product_name'].fillna("").apply(_normalize_text)
    df_b['_price_val'] = df_b['price_clean'].fillna(0).astype(float)
    max_price = max(df_a['price_clean'].max() if 'price_clean' in df_a.columns else 0,
                    df_b['_price_val'].max() if not df_b.empty else 0, 1)
    for idx_a, a in df_a.reset_index(drop=True).iterrows():
        a_name = a.get('product_name', "")
        a_price = float(a.get('price_clean', 0) or 0)
        a_img = a.get('product_img', "")
        best_candidates = []
        for idx_b, b in df_b.iterrows():
            b_name = b.get('product_name', "")
            b_price = float(b.get('_price_val', 0) or 0)
            name_score = _name_similarity(a_name, b_name)
            if max(a_price, b_price) > 0:
                price_diff_pct = abs(a_price - b_price) / max(a_price, b_price)
            else:
                price_diff_pct = 0.0
            price_score = max(0.0, 1.0 - price_diff_pct) * 100
            combined = (name_weight * name_score) + (price_weight * price_score)
            best_candidates.append({
                "idx_b": idx_b,
                "b_name": b_name,
                "b_price": b_price,
                "b_img": b.get('product_img', ""),
                "name_score": name_score,
                "price_score": price_score,
                "combined_score": combined,
                "price_diff_pct": price_diff_pct
            })
        best_candidates = sorted(best_candidates, key=lambda x: x['combined_score'], reverse=True)
        selected = []
        for cand in best_candidates[:max(10, top_k)]:
            if cand['name_score'] >= name_threshold and cand['price_diff_pct'] <= price_tolerance_pct:
                selected.append(cand)
            elif cand['combined_score'] >= 80:
                selected.append(cand)
            if len(selected) >= top_k:
                break
        if selected:
            for cand in selected:
                b_row = df_b.loc[cand['idx_b']]
                rows.append({
                    "platform_a": a.get('platform', 'A'),
                    "name_a": a_name,
                    "price_a": a_price,
                    "img_a": a_img,
                    "platform_b": b_row.get('platform', 'B'),
                    "name_b": cand['b_name'],
                    "price_b": cand['b_price'],
                    "img_b": cand['b_img'],
                    "name_score": round(cand['name_score'], 2),
                    "price_score": round(cand['price_score'], 2),
                    "combined_score": round(cand['combined_score'], 2),
                    "price_diff_pct": round(cand['price_diff_pct'] * 100, 2)
                })
        else:
            rows.append({
                "platform_a": a.get('platform', 'A'),
                "name_a": a_name,
                "price_a": a_price,
                "img_a": a_img,
                "platform_b": None,
                "name_b": None,
                "price_b": None,
                "img_b": None,
                "name_score": 0,
                "price_score": 0,
                "combined_score": 0,
                "price_diff_pct": None
            })
    matches_df = pd.DataFrame(rows)
    matches_df = matches_df.sort_values(by='combined_score', ascending=False).reset_index(drop=True)
    return matches_df

# --- UI: Product matching (only if both platforms present) ---
if not df.empty and {'tokopedia','shopee'}.issubset(set(df['platform'].astype(str).str.lower())):
    st.divider()
    st.subheader("Pencocokan Produk Antar Platform")
    df_sp = df[df['platform'] == 'shopee'].copy().reset_index(drop=True)
    df_tp = df[df['platform'] == 'tokopedia'].copy().reset_index(drop=True)
    if df_sp.empty or df_tp.empty:
        st.info("Salah satu platform tidak memiliki data untuk dicocokkan.")
    else:
        with st.expander("Pengaturan Pencocokan"):
            name_w = st.slider("Bobot Nama Produk", 0.0, 1.0, 0.7, 0.05)
            price_w = 1.0 - name_w
            name_thresh = st.slider("Ambang Nama Minimum (0-100)", 40, 90, 65, 1)
            price_tol = st.slider("Toleransi Selisih Harga (%)", 0, 100, 35, 5) / 100.0
            top_k = st.number_input("Jumlah Kandidat Teratas per Produk", min_value=1, max_value=5, value=1, step=1)
        st.markdown(f"**Bobot nama**: {name_w}  •  **Bobot harga**: {price_w}  •  **Ambang nama**: {name_thresh}  •  **Toleransi harga**: {int(price_tol*100)}%")
        matches = match_products(df_sp, df_tp,
                                 name_weight=name_w,
                                 price_weight=price_w,
                                 name_threshold=name_thresh,
                                 price_tolerance_pct=price_tol,
                                 top_k=top_k)
        if matches.empty:
            st.warning("Tidak ditemukan pasangan produk.")
        else:
            st.markdown("**Hasil Pencocokan Teratas**")
            display_cols = ['name_a','platform_a','price_a','name_b','platform_b','price_b','combined_score','name_score','price_diff_pct']
            st.dataframe(matches[display_cols].head(50).rename(columns={
                'name_a':'Nama Shopee',
                'platform_a':'Platform A',
                'price_a':'Harga A',
                'name_b':'Nama Tokopedia',
                'platform_b':'Platform B',
                'price_b':'Harga B',
                'combined_score':'Skor Gabungan',
                'name_score':'Skor Nama',
                'price_diff_pct':'Selisih Harga (%)'
            }), use_container_width=True)
            unmatched = matches[matches['combined_score'] == 0]
            if not unmatched.empty:
                st.markdown(f"**Produk Shopee tanpa pasangan ({len(unmatched)})**")
                st.dataframe(unmatched[['name_a','price_a']].rename(columns={'name_a':'Nama Produk Shopee','price_a':'Harga'}).head(50), use_container_width=True)
            matched_b_names = matches[matches['name_b'].notna()]['name_b'].tolist()
            unmatched_b = df_tp[~df_tp['product_name'].isin(matched_b_names)]
            if not unmatched_b.empty:
                st.markdown(f"**Produk Tokopedia tanpa pasangan ({len(unmatched_b)})**")
                st.dataframe(unmatched_b[['product_name','price_clean']].rename(columns={'product_name':'Nama Produk Tokopedia','price_clean':'Harga'}).head(50), use_container_width=True)
            try:
                csv_data = matches.to_csv(index=False)
                st.download_button("Unduh Hasil Pencocokan CSV", data=csv_data, file_name="matches_shopee_tokopedia.csv", mime="text/csv")
            except Exception:
                st.text_area("Hasil Pencocokan CSV", value=matches.to_csv(index=False), height=200)

# --- ANALISIS CSI & VISUALISASI ---
if not df.empty:
    platforms = sorted(df['platform'].unique())
    scores_map = {}
    cols = st.columns(len(platforms) + 1)
    for i, plat in enumerate(platforms):
        p_df = df[df['platform'] == plat].copy()
        total_sales = p_df['sold_num'].sum()
        if total_sales > 0:
            p_df['wf'] = p_df['sold_num'] / total_sales
            p_df['ws'] = p_df['rating_norm'] * p_df['wf']
        else:
            p_df['wf'] = 0
            p_df['ws'] = p_df['rating_norm']
        csi_score = (p_df['ws']).sum() * 100
        scores_map[plat] = round(csi_score, 2)
        df.loc[df['platform'] == plat, 'ws'] = p_df['ws']
        status_txt, status_clr = get_status_info(csi_score)
        with cols[i]:
            st.metric(label=f"CSI {plat.upper()}", value=f"{scores_map[plat]}%")
            st.markdown(f'<div class="interpretation-card" style="background-color:{status_clr};">{status_txt}</div>', unsafe_allow_html=True)

    if len(scores_map) >= 2:
        vals = list(scores_map.values())
        gap = round(abs(vals[0] - vals[1]), 2)
        with cols[-1]:
            st.metric(label="GAP", value=f"{gap}%")
            st.markdown('<div class="interpretation-card" style="background-color:#6c757d;">KOMPARASI</div>', unsafe_allow_html=True)

    st.divider()
    v1, v2 = st.columns(2)
    with v1:
        st.subheader("Perbandingan CSI")
        fig_bar = px.bar(x=list(scores_map.keys()), y=list(scores_map.values()), color=list(scores_map.keys()), text_auto=True,
                         color_discrete_map={'shopee':'#EE4D2D','tokopedia':'#42B549','csv':'#6c757d'},
                         labels={'x':'Platform','y':'CSI (%)'})
        st.plotly_chart(fig_bar, use_container_width=True)
    with v2:
        st.subheader("Distribusi Segmen Harga")
        fig_pie = px.pie(df, names='Segmen Harga', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()
    st.subheader("Top 10 Produk")
    df['Harga Produk'] = df['price_clean'].apply(format_rupiah)
    df_final = df.rename(columns={'platform':'Platform','product_name':'Nama Produk','product_rating':'Rating Produk','sold_num':'Jumlah Terjual'})
    top10 = df_final.sort_values(by='ws', ascending=False).head(10)
    display_cols = ['Platform','Nama Produk','Harga Produk','Rating Produk','Jumlah Terjual']
    st.dataframe(top10[display_cols], use_container_width=True)

    # Kesimpulan otomatis
    def generate_conclusion(df, scores_map, top_n=5):
        conclusions = []
        recommendations = []
        if not scores_map:
            conclusions.append("Tidak ada skor CSI yang tersedia untuk dianalisis.")
            return conclusions, recommendations
        sorted_scores = sorted(scores_map.items(), key=lambda x: x[1], reverse=True)
        best_platform, best_score = sorted_scores[0]
        worst_platform, worst_score = sorted_scores[-1]
        conclusions.append(f"Platform terbaik berdasarkan CSI adalah **{best_platform.upper()}** dengan skor **{best_score}%**.")
        conclusions.append(f"Platform terendah adalah **{worst_platform.upper()}** dengan skor **{worst_score}%**.")
        if len(sorted_scores) >= 2:
            gap = round(abs(sorted_scores[0][1] - sorted_scores[1][1]), 2)
            conclusions.append(f"GAP antara dua platform teratas adalah **{gap}%**.")
            if gap >= 10:
                recommendations.append("Perbedaan CSI cukup besar; fokuskan analisis penyebab pada platform yang tertinggal (harga, rating, atau volume penjualan).")
            else:
                recommendations.append("Perbedaan CSI relatif kecil; strategi kompetitif dapat difokuskan pada diferensiasi produk dan promosi.")
        segmen_counts = df['Segmen Harga'].value_counts(dropna=True).to_dict()
        if segmen_counts:
            top_seg = max(segmen_counts.items(), key=lambda x: x[1])
            conclusions.append(f"Segmen harga yang paling banyak muncul adalah **{top_seg[0]}** ({top_seg[1]} produk).")
            if 'Budget' in str(top_seg[0]):
                recommendations.append("Pasar didominasi produk budget; pertimbangkan strategi volume dan promosi harga.")
            else:
                recommendations.append("Pertimbangkan strategi nilai tambah (bundle, garansi, layanan purna jual) untuk segmen ini.")
        if 'ws' in df.columns and not df.empty:
            top_products = df.sort_values(by='ws', ascending=False).head(top_n)
            conclusions.append(f"Top {min(top_n, len(top_products))} produk unggulan (berdasarkan skor gabungan rating x penjualan):")
            for idx, row in top_products.iterrows():
                name = row.get('product_name') or row.get('Nama Produk') or ''
                platform = row.get('platform', '')
                price = row.get('price_clean', None)
                sold = int(row.get('sold_num', 0)) if not pd.isna(row.get('sold_num', None)) else 0
                price_str = f"Rp{int(price):,}".replace(",", ".") if price and price > 0 else "Harga tidak tersedia"
                conclusions.append(f"- {name} ({platform}) — {price_str}, terjual {sold} unit.")
            recommendations.append("Pertahankan stok dan promosi untuk produk unggulan; gunakan sebagai benchmark untuk listing lain.")
        recommendations.append("Periksa listing dengan rating rendah namun volume tinggi untuk perbaikan kualitas produk atau deskripsi.")
        recommendations.append("Optimalkan gambar dan judul untuk produk dengan CTR rendah; jalankan A/B test pada beberapa listing.")
        recommendations.append("Pantau harga kompetitor pada platform dengan CSI lebih tinggi dan sesuaikan strategi harga bila perlu.")
        return conclusions, recommendations

    conclusions, recommendations = generate_conclusion(df, scores_map, top_n=5)
    st.divider()
    st.subheader("Kesimpulan Analisis Otomatis")
    for line in conclusions:
        st.markdown(line, unsafe_allow_html=True)
    st.subheader("Rekomendasi Singkat")
    for rec in recommendations:
        st.markdown(f"- {rec}", unsafe_allow_html=True)
    try:
        summary_text = "\n".join(conclusions) + "\n\nRekomendasi:\n" + "\n".join(f"- {r}" for r in recommendations)
        st.download_button("Unduh Ringkasan (TXT)", data=summary_text, file_name="kesimpulan_csi.txt", mime="text/plain")
    except Exception:
        st.text_area("Ringkasan (salin manual jika perlu)", value=summary_text, height=240)
else:
    st.warning("Silakan pilih sumber data dan upload CSV atau fetch live scraping.")

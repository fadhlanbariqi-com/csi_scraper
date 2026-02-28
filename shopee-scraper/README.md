
# Shopee Scraper

This Python application scrapes product data from Shopee. It retrieves basic product information (name, link, price, etc.), and can also collect product reviews in different modes.

## Installation Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/dtungpka/shopee-scraper.git
   cd shopee-scraper
   ```

2. **Create a Virtual Environment (Optional But Recommended):**
   - Linux/Mac:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Before running:
- Make sure all Chrome windows are fully closed.
- Prepare to log in and solve captchas manually if prompted.

### Basic Command

```bash
python src/retriv.py -k "your_search_term" -n 10 -r 30
```

When you see the search page loaded in the browser:
1. Log in to Shopee (if needed).
2. Solve any captcha presented.
3. After continuing to the main search page, press Enter in the terminal to proceed.
4. Keep an eye on the browser; if another captcha appears at any point, solve it to continue scraping.

### Scraping Modes

1. **Review Limit Mode:**
   - Use `-r` or `--review-limit` to collect reviews from 5-stars downwards until the limit is met:
     ```bash
     python src/retriv.py -k "laptop" -n 5 -r 10
     ```
   This collects up to 10 reviews per product, starting from the top ratings and moving downward.

2. **All-Star Types Mode:**
   - Combine `--all-star-types` with `--star-limit-per-type` to specify how many reviews to retrieve for each star rating:
     ```bash
     python src/retriv.py -k "laptop" -n 5 --all-star-types --star-limit-per-type 5
     ```
   This collects 5 reviews for 5-star, 4-star, 3-star, etc., in separate queries.

### Command-Line Arguments

- `-k`, `--keyword`: Search term (default: "Raspberry pi")
- `-n`, `--num`: Number of products to retrieve (default: 10)
- `-r`, `--review-limit`: Total reviews to collect per product (default: 30)
- `--index-only`: If set, only retrieve index data without details
- `--all-star-types`: Collect each star rating separately
- `--star-limit-per-type`: Reviews per star type (default: 10)
- `--chrome-user-data-dir`: Path to your Chrome profile directory

## Example Command

```bash
python src/retriv.py -k "laptop" -n 5 --all-star-types --star-limit-per-type 3
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.
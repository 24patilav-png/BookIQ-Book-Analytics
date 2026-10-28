# BookIQ – Book Price & Rating Analytics Dashboard

## Overview
BookIQ is an interactive data analytics dashboard built using Python and Streamlit. It helps users explore book prices and ratings through visualizations, filters, book comparisons, and statistical hypothesis testing.

## Features
- **Overview Dashboard:** Summary statistics and charts for book prices and ratings.
- **Book Discovery:** Search, filter, sort, and explore books.
- **Compare Books:** Compare prices, ratings, and availability indicators for two books.
- **Analytics:** Visualize price and rating distributions.
- **Hypothesis Testing Lab:** Perform statistical tests and interpret p-values.
- **Dataset & Data Quality:** Preview, inspect, and export dataset information.

## Technologies Used
- Python
- Streamlit
- Pandas
- NumPy
- SciPy
- CSV dataset

## Dataset
The project analyses 600 book records. The catalogue source is [Books to Scrape](https://books.toscrape.com/).

## Important Notes
- Currency conversion uses a manually configured rate of ₹128 per £1 by default.
- Availability information is simulated for demonstration and does not represent live stock.
- Prices and analysis results depend on the supplied dataset and selected settings.

## Installation and Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/24patilav-png/BookIQ-Book-Analytics.git
   ```

2. Open the project folder:

   ```bash
   cd BookIQ-Book-Analytics
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL displayed in the terminal.

## Repository
[BookIQ on GitHub](https://github.com/24patilav-png/BookIQ-Book-Analytics)

## Author
Anshul Patil  
Computer Science & Engineering  
Ramdeobaba University, Nagpur

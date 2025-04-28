# Machu Pouches Dashboard

A Streamlit dashboard for visualizing and analyzing Machu Pouches company data from Google Sheets.

## Setup Instructions

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Google Sheets API:
   - Go to the [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project
   - Enable the Google Sheets API
   - Create credentials (Service Account)
   - Download the JSON credentials file
   - Rename the downloaded file to `credentials.json` and place it in the project root directory

3. Share your Google Sheet:
   - Open your Google Sheet
   - Click the "Share" button
   - Add the service account email (found in your credentials.json) as an editor

4. Run the dashboard:
```bash
streamlit run app.py
```

## Features

- Real-time data from Google Sheets
- Interactive filters for date range and products
- Sales overview metrics
- Product performance visualization
- Sales trend analysis
- Raw data view

## Data Structure

The dashboard expects the following columns in your Google Sheet:
- Date
- Product
- Sales Amount
- Other relevant columns will be automatically included in the raw data view

## Note

Make sure to keep your `credentials.json` file secure and never commit it to version control. 
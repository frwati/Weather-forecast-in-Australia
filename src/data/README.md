# Evidently UI for Drift Monitoring

This project uses Evidently to monitor data drift and visualize the results. It provides an easy way to track performance metrics and assess model behavior over time.

## Prerequisites

Make sure you have the following installed:

- Python 3.7 or higher
- Evidently library (`evidently`)
- Your workspace with drift monitoring results (in this case, `Weather-Classification` folder)

### Install Evidently

To install Evidently, you can use `pip`:


pip install evidently

Then run :
evidently ui --workspace ./Weather-Classification/

it takes 1 minute to be able to view running on http://127.0.0.1:8000 

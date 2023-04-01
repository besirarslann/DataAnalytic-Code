###############################################################
# Customer Segmentation with RFM
###############################################################

# Customer Segmentation with RFM in 6 Steps

1. Business Problem
2. Data Understanding
3. Data Preparation
4. Calculating RFM Metrics
5. Calculating RFM Scores
6. Naming & Analysing RFM Segments

# Purpose of Working
An e-commerce company wants to segment its customers and determine marketing strategies according to these segments. 
To this end, we will define the behaviors of the customers and create segmentation according to the clusters in these behaviors. The segmentation to be made has been set up manually.
Recency : Time from customer's last contact to date --> Today's date - Last purchase
Frequency : Number of customer visits --> Invoice number of unique of visit 
Monetary : Total money earned by the customer

# Dataset Licence
https://archive.ics.uci.edu/ml/datasets/Online+Retail+II


The data set named Online Retail II was obtained from a UK-based online store.
# 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını içeriyor.

# Parametres

* InvoiceNo: Invoice number The unique number of each transaction, that is, the invoice NOTE: If this code starts with C, it means that the operation has been cancelled.
* StockCode: Product code. Unique number for each product.
* Description: Product name
* Quantity: Number of products. It expresses how many of the products on the invoices have been sold.
* InvoiceDate: Invoice date and time.
* UnitPrice: Product price (in GBP)
v CustomerID: Unique customer number
v Country: Country name. Country where the customer lives


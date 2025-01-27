#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# In[4]:


# Load the datasets
customers = pd.read_csv("Customers.csv")
products = pd.read_csv("Products.csv")
transactions = pd.read_csv("Transactions.csv")

# Display the first few rows of each dataset
print(customers.head())
print(products.head())
print(transactions.head())


# In[5]:


print(customers.isnull().sum())
print(products.isnull().sum())
print(transactions.isnull().sum())


# In[6]:


sns.countplot(data=customers, x='Region')
plt.title("Customer Distribution by Region")
plt.show()


# In[7]:


top_products = transactions.groupby('ProductID')['Quantity'].sum().sort_values(ascending=False).head(10)
top_products.plot(kind='bar')
plt.title("Top-Selling Products")
plt.xlabel("ProductID")
plt.ylabel("Quantity Sold")
plt.show()


# In[8]:


transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])
transactions.groupby(transactions['TransactionDate'].dt.month).sum()['TotalValue'].plot(kind='line')
plt.title("Transaction Trends by Month")
plt.xlabel("Month")
plt.ylabel("Total Transaction Value")
plt.show()


# In[10]:


# Merge datasets
merged_data = transactions.merge(customers, on="CustomerID").merge(products, on="ProductID")

# Create a pivot table for customer-product matrix
customer_product_matrix = merged_data.pivot_table(index='CustomerID', columns='ProductID', values='Quantity', fill_value=0)

# Calculate cosine similarity
similarity_matrix = cosine_similarity(customer_product_matrix)

# Convert to a DataFrame
similarity_df = pd.DataFrame(similarity_matrix, index=customer_product_matrix.index, columns=customer_product_matrix.index)


# In[12]:


lookalike_results = {}
for customer in similarity_df.index[:20]:
    similar_customers = similarity_df.loc[customer].sort_values(ascending=False).iloc[1:4]
    lookalike_results[customer] = list(similar_customers.index) + list(similar_customers.values)

# Save results as CSV
lookalike_df = pd.DataFrame.from_dict(lookalike_results, orient='index', columns=['Similar_Cust_1', 'Score_1', 'Similar_Cust_2', 'Score_2', 'Similar_Cust_3', 'Score_3'])
lookalike_df.to_csv("Lookalike.csv", index_label="CustomerID")


# In[ ]:





# In[ ]:





# In[ ]:





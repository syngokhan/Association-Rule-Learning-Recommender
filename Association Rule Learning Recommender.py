#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


pd.set_option("display.max_columns",None)
pd.set_option("display.float_format", lambda x : "%.3f" % x)
pd.set_option("display.width", 200)


# ## Task 1:
# 
# * Perform Data Preprocessing
# 
# **Important note!**
# 
# * Select 2010-2011 data and preprocess all data. The choice of Germany will be the next step.

# In[3]:


def check_df(dataframe , head = 5 ,tail = 5):
        print(" head ".upper().center(50,"#"),end = "\n\n")
        print(dataframe.head(head),end = "\n\n")
        
        print(" tail ".upper().center(50,"#"),end = "\n\n")
        print(dataframe.tail(tail),end = "\n\n")
        
        print(" shape ".upper().center(50,"#"),end = "\n\n")
        print(dataframe.shape,end = "\n\n")
        
        print(" dtypes ".upper().center(50,"#"),end = "\n\n")
        print(dataframe.dtypes,end = "\n\n")
        
        print(" ndim ".upper().center(50,"#"),end = "\n\n")
        print(f"{dataframe.ndim} Boyutlu",end = "\n\n")
        
        print(" na ".upper().center(50,"#"),end = "\n\n")
        print(dataframe.isnull().sum(),end = "\n\n")
        
        print(" quantiles ".upper().center(50,"#"),end = "\n\n")
        print(dataframe.describe(percentiles = [ 0.01, 0.05, 0.95 , 0.99 ]).T, end = "\n\n")


# In[4]:


path = "/Users/gokhanersoz/Desktop/VBO_Dataset/online_retail_II.xlsx"


# In[5]:


online_retail = pd.read_excel(path, sheet_name = "Year 2010-2011")


# In[6]:


df = online_retail.copy()


# In[7]:


check_df(df)


# In[8]:


print("Max Year : {}".format(df.InvoiceDate.max().year))
print("Min Year : {}".format(df.InvoiceDate.min().year))


# In[9]:


def outlier_thresholds(dataframe, col_name, q1 = 0.01, q3 = 0.99):
    quantile1 = dataframe[col_name].quantile(q1)
    quantile3 = dataframe[col_name].quantile(q3)
    interquantile = quantile3 - quantile1
    up_limit = quantile3 + 1.5*interquantile
    low_limit = quantile1 - 1.5*interquantile
    return up_limit, low_limit

def replace_with_thresholds(dataframe,col_name , q1 = 0.01, q3 = 0.99):
    up_limit , low_limit = outlier_thresholds(dataframe, col_name , q1 , q3)
    dataframe.loc[ (dataframe[col_name] > up_limit), col_name] = up_limit
    dataframe.loc[ (dataframe[col_name] < low_limit), col_name] = low_limit


# In[10]:


df.describe([0.01,0.99]).T


# In[11]:


def boxplot_outliers(dataframe, num_cols):
    
    plt.figure(figsize = (10,5))
    num=len(num_cols)
    i=1
    size = 15
    for col in num_cols:
        plt.subplot(1,num,i) 
        plt.boxplot(dataframe[col])
        plt.title(f"For {col} Outliers Values", fontsize = size)
        plt.xlabel(col, fontsize = size)
        plt.ylabel("Values", fontsize = size)
        i+=1
        plt.tight_layout()
    plt.show()


# In[12]:


num_cols = [col for col in df.columns if df[col].dtype == "int64" or df[col].dtype == "float64"]

boxplot_outliers(df, num_cols)


# In[13]:


def online_retail_prep(dataframe):
    dataframe.dropna(inplace = True , axis = 0)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C" , na = False)]
    dataframe = dataframe[dataframe["Quantity"] > 0 ]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


# In[14]:


df = online_retail_prep(df)
boxplot_outliers(df, num_cols)


# In[15]:


df.describe([.01, .99]).T


# ## Mission 2:
# 
# * Generate association rules through Germany customers.

# In[16]:


country = df["Country"].value_counts()
country = pd.DataFrame(country).sort_values(by="Country" , ascending = False)
country.head()


# In[17]:


#Germany countries we only receive...
# I just wanted to control...

df_de = df[df["Country"] == "Germany"]
df_de.Country.unique()


# In[18]:


# Let's go by understanding step by step what we want to achieve ...

df_de.groupby(["Invoice","Description"]).agg({"Quantity" : "sum"}).head(10)


# In[19]:


values = 536527
df_de[df_de["Invoice"] == 536527]


# In[20]:


# Let's look at the working logic of the stack !!!

index = pd.MultiIndex.from_tuples([("one","a"), ("one","b"), ("two","a"), ("two","b")])
s = pd.DataFrame(data = np.arange(1.0,5.0) , index = index)
s


# In[21]:


s.unstack()


# In[22]:


s.stack()


# In[23]:


df_de.groupby(["Invoice","Description"]).agg({"Quantity" : "sum"}).unstack()


# In[24]:


df_de.groupby(["Invoice","Description"]).agg({"Quantity" : "sum"}).unstack().fillna(0).iloc[:5,:5]


# In[25]:


# Is it normally bought here or not? Can we think of it as an answer to the question?

test= df_de.groupby(["Invoice","Description"]).agg({"Quantity" : "sum"}).unstack().fillna(0).iloc[20:30,10:30]
test.applymap(lambda x : 1 if x > 0 else 0)


# In[26]:


# I wanted to look at why there is a difference between stockcode and description.. !!!
# Normally, there is only one product per invoice, but it seems that more than one product is included here.

df.StockCode.value_counts()


# In[27]:


df.Description.value_counts()


# In[28]:


stockcode = df[df["StockCode"] == "85123A"][["Description","Quantity"]]
description = df[df["Description"] == "WHITE HANGING HEART T-LIGHT HOLDER"][["Description","Quantity"]]


# In[29]:


# I made the generalization here for the whole df, I would like to point out that (not only for the Germans...) ...
# As a different result from here, there may be a different description name belonging to that stockcode ...
# That's why we get a different result in value_counts() values ....

col ="Description"

print(f"For {col} Unique Values : {stockcode[col].unique()}\nAnd Count: {stockcode[col].count()} ",end = "\n\n")
print(f"For {col} Unique Values : {description[col].unique()}\nAnd Count : {description[col].count()} ")


# In[30]:


def create_invoice_product_df(dataframe, id = False):
        
    if id :
        col_name = "StockCode"
        return dataframe.groupby(["Invoice", col_name ])["Quantity"].sum().unstack().fillna(0).                                                                        applymap(lambda x : 1 if x > 0 else 0)
    
    else:
        col_name = "Description"
        return dataframe.groupby(["Invoice", col_name ])["Quantity"].sum().unstack().fillna(0).                                                                        applymap(lambda x : 1 if x > 0 else 0)


# In[31]:


df_de_desc = create_invoice_product_df(df_de)
df_de_id = create_invoice_product_df(df_de , id = True)

print("For Description Germany DataFrame Shape : {}".format(df_de_desc.shape))

print("For ID Germany DataFrame Shape : {}".format(df_de_id.shape))


# In[32]:


df_de.head()


# ## Mission 2:
# 
# ***Produce association rules through Germany customers.***
# 
# * antecedent support: X probability alone
# 
# * consequent support: Y probability alone
# 
# * support: probability of both occurring together
# 
# * confidence: probability of getting Y when X is taken.
# 
# * lift: When X is taken, the probability of getting Y increases .. times.
# 
# * conviction: expected frequency of X without Y

# In[33]:


# I looked at the difference ...
# When you do this, you may get an error in the function.... (in association_rules....)
# Quantity !!!!!!!

df_de.groupby(["Invoice", "StockCode"]).agg({"Quantity" : "sum"}).unstack().fillna(0).                                                                 applymap(lambda x : 1 if x > 0 else 0).iloc[:5]


# In[34]:


df_de_id.head()


# In[35]:


"""
  use_colnames : bool (default: False)
      If `True`, uses the DataFrames' column names in the returned DataFrame
      instead of column indices.
  
  min_support : float (default: 0.5)
      A float between 0 and 1 for minumum support of the itemsets returned.
      The support is computed as the fraction
      `transactions_where_item(s)_occur / total_transactions`.
      

  """


from mlxtend.frequent_patterns import apriori , association_rules


# In[36]:


frequent_itemsets = apriori(df= df_de_id,min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values(by = "support", ascending = False)


# In[37]:


"""
min_threshold : float (default: 0.8)
  Minimal threshold for the evaluation metric,
  via the `metric` parameter,
  to decide whether a candidate rule is of interest.

metric : string (default: 'confidence')
  Metric to evaluate if a rule is of interest.
  **Automatically set to 'support' if `support_only=True`.**
  Otherwise, supported metrics are 'support', 'confidence', 'lift',
  'leverage', and 'conviction'

"""

rules = association_rules(df = frequent_itemsets, metric="support", min_threshold=0.01)
rules


# In[38]:


rules.sort_values(by = "support", ascending =False).head(10)


# In[39]:


rules.sort_values(by = "lift", ascending =False).head(10)


# In[40]:


sorted_rules = rules.sort_values(by = "lift", ascending = False)
sorted_rules.head()


# ## Mission 3:
# 
# ***What are the names of the products whose IDs are given?***
# 
# * User 1 product id: 21987
# * User 2 product id: 23235
# * User 3 product id: 22747

# In[41]:


users = {"User_1" : 21987, "User_2" : 23235, "User_3" : 22747}


# In[42]:


def check_description(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code]["Description"].values[0]
    return product_name


# In[43]:


for user in users.keys():
    
    print(f"For {user} StockCode : '{users[user]}' Description : {check_description(df, users[user])}")


# In[44]:


df[df["StockCode"] == 21987]["Description"].values[0]


# ## Task 4 and Task 5:
# 
# * Make a product recommendation for the users in the cart.
# 
# * What are the names of the recommended products?

# In[45]:


sorted_rules.head(10)


# In[46]:


# ...[0] the reason we do it is to catch it from the list.... !!!!

# antecedent support: X probability alone
# consequent support: Y probability alone

product_id = 21987

recommendation_list = []
last_list = []
iloc = []

for i,products in enumerate(sorted_rules["antecedents"]):
    for j in list(products):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
            iloc.append(i)
            
recommendation_list[:10]


# In[47]:


# Unique values

set(recommendation_list)


# In[48]:


# Unique values
# Same Way

for i in recommendation_list:
    if i not in last_list:
        last_list.append(i)
        
last_list


# In[49]:


# Remember, we did this for the Germans.
# For german we used apriori and association_rules !!!!

def arl_recommender(dataframe , rules_df , product_id, rec_count = 1):
    
    product_name = dataframe[dataframe["StockCode"] ==  product_id]["Description"].values[0]
    print(" ID and Product Name used for the recommendation ".upper().center(100,"#"),end = "\n\n")
    print(f"Product Name : {product_name}\nProduct ID : {product_id}",end = "\n\n")
    
    recommendation_list = []
    last_list = []
    
    for i,products in enumerate(rules_df["antecedents"]):
        for j in list(products):
            if product_id == j:
                recommendation_list.append(list(rules_df.iloc[i]["consequents"])[0])
                
    for product in recommendation_list:
        if product not in last_list:
            last_list.append(product)
    
    print(" Recommended Total Product ID ".upper().center(100,"#"),end = "\n\n")
    
    for product_id in last_list:
        description = dataframe[dataframe["StockCode"] ==  product_id]["Description"].values[0]
        print(f"Product ID : {product_id} ,Description : {description}")
    print("\n")
        
    if 1 <=rec_count <= len(last_list):
        
        print(" Recommended Stock Code ID and Product Name ".upper().center(100,"#"),end = "\n\n")
        for num in range(rec_count):
            product_id = last_list[num]
            description = dataframe[dataframe["StockCode"] == product_id ]["Description"].values[0]
            print(f"Product ID : {product_id}, Description : {description}")
            
    elif rec_count == 0:
        
        print(" Recommended Stock Code ID and Product Name ".upper().center(100,"#"),end = "\n\n")
        print("You entered 0 as a recommended value !!!" , end = "\n\n")
        
    elif rec_count < 0 :
        
        print(" Attention ".upper().center(100,"#"),end = "\n\n")
        print("Please do not enter negative numbers !!!!", end = "\n\n")
    
    else:
        
        print(" Recommended Stock Code ID and Product Name ".upper().center(100,"#"),end = "\n\n")
        print(f"Recommended Product Quantity is Maximum {len(last_list)}!!!!!\n")
        
        rec_count = len(last_list)
        for num in range(rec_count):
            product_id = last_list[num]
            description = dataframe[dataframe["StockCode"] ==  product_id]["Description"].values[0]
            print(f"Product ID : {product_id}, Description : {description}")
            


# In[50]:


arl_recommender(df_de, sorted_rules, users['User_1'], rec_count = 6)


# In[51]:


arl_recommender(df_de, sorted_rules, users['User_1'], rec_count = 3)


# In[52]:


arl_recommender(df_de, sorted_rules, users['User_1'], rec_count = 0)


# In[53]:


arl_recommender(df_de, sorted_rules, users['User_1'], rec_count = -2)


# In[54]:


for product_id in users.values():
    
    arl_recommender(df_de, sorted_rules, product_id, rec_count=3)
    print("\n","".center(100,"*"),end="\n\n")


# In[55]:


while True:
    print("""
    
     Example Stock Codes: 23254,23255,22899
    
     Just Press "q" to exit....
    
     Transactions will be taken according to the sorted_rules dataframe.
    
     """)
    
    product_id = input("INPUT STOCK CODE:\n")
    rec_count = input("HOW MANY PRODUCT RECOMMENDATIONS DO YOU WANT:\n")
    
    if product_id == "q" or rec_count == "q":
        print("Logout !!!!!!")
        break
        
    else:
        product_id = int(product_id )
        rec_count = int(rec_count)
        
        arl_recommender(df_de, sorted_rules, product_id ,rec_count)


# ## Extra

# In[56]:


def dataframe_prep(dataframe, Country , id = True):
    
    from mlxtend.frequent_patterns import apriori, association_rules
    
    dataframe = dataframe.dropna(axis=0)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C",na = False)]
    dataframe = dataframe[dataframe["Quantity"] > 0 ]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    
    dataframe = dataframe[dataframe["Country"] == Country]
    print(dataframe.Country.unique())
    
    if id:
        
        dataframe =         dataframe.groupby(["Invoice","StockCode"])["Quantity"].sum().unstack().fillna(0).                                                                    applymap(lambda x : 1 if x > 0 else 0)
        
    else:
        
        dataframe =         dataframe.groupby(["Invoice","Description"])["Quantity"].sum().unstack().fillna(0).                                                                    applymap(lambda x : 1 if x > 0 else 0)
        
        
    frequent_itemsets = apriori(df = dataframe,
                                min_support=0.01, 
                                use_colnames=True)
    
    rules = association_rules(df = frequent_itemsets,
                              metric= "support",
                              min_threshold=0.01)
    
    sorted_rules = rules.sort_values(by = "lift" , ascending = False)
    
    return dataframe, sorted_rules


# In[57]:


df_test = online_retail.copy()


# In[58]:


df_test.Country.unique()


# In[59]:


df_test[df_test["Invoice"].str.contains("C", na = False)]["Invoice"].unique()


# In[60]:


df_fr , sorted_rules = dataframe_prep(df_test , "France", id = True)


# In[61]:


df_fr


# In[62]:


sorted_rules


# In[63]:


df_fr , sorted_rules = dataframe_prep(df_test , "France", id = False)


# In[64]:


df_fr


# In[65]:


sorted_rules


# In[ ]:





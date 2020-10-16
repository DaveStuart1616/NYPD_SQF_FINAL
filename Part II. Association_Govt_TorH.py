# %% read dataframe
import pandas as pd
import  seaborn as sns
import matplotlib.pyplot as plt 
from mlxtend.frequent_patterns import apriori, association_rules


df = pd.read_pickle("data.pkl")


# %% convert physical forces used columns to booleans
pfs = [col for col in df.columns if col.startswith("pf_")]
for col in pfs:
    df[col] = df[col] == "YES"


# %% manual one hot encoding for race / city
for val in df["race"].unique():
    df[f"race_{val}"] = df["race"] == val

for val in df["city"].unique():
    df[f"city_{val}"] = df["city"] == val

# %% convert inout/arstmade to boolean
df["inside"] = df["inout"] == "INSIDE"
df["arrest"] = df["arstmade"] == "Y"

# %% create armed column
df["armed"] = (
    (df["contrabn"] == "YES")
    | (df["pistol"] == "YES")
    | (df["riflshot"] == "YES")
    | (df["asltweap"] == "YES")
    | (df["knifcuti"] == "YES")
    | (df["machgun"] == "YES")
    | (df["othrweap"] == "YES")
)
#%%
#df["trhsloc"] = str(df["trhsloc"])

df = df[df["trhsloc"] != "NEITHER"]
#df = df.trhsloc == "HOUSING"


# %% select columns for association rules mining
cols = [
    col
    for col in df.columns
    if col.startswith("pf_") or col.startswith("race_") or col.startswith("city_")
] + ["inside", "armed", "arrest"]

# %% apply frequent itemset mining
frequent_itemsets = apriori(df[cols], min_support=0.05, use_colnames=True)

# %% apply association rules mining
rules = association_rules(frequent_itemsets, min_threshold=0.03)

# %% sort rules by confidence
rules.sort_values("confidence", ascending=False)

# %%Plot Support/Confidence
sns.lmplot(data=rules, x='support', y='confidence')
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence trhsloc Dataset')


# %%
sns.lmplot(data=rules, x='support', y='lift')
plt.xlabel('support')
plt.ylabel('lift')
plt.title('Support vs Lift trhsloc Dataset')
# %%

# %% Import and read
from datetime import datetime
from matplotlib.pyplot import xlabel
import pandas as pd
from seaborn.categorical import countplot
import matplotlib.pyplot as plt
from seaborn.regression import lmplot
df = pd.read_csv("2012.csv")

# %%Describe before df changes
df.describe(include="all")
df.describe(include="all").to_excel("statspre.xlsx", startrow=1)

# %%Integer Convert and Coerce
from tqdm import tqdm

cols = [
    "perobs",
    "perstop",
    "age",
    "weight",
    "ht_feet",
    "ht_inch",
    "datestop",
    "timestop",
    "xcoord",
    "ycoord",
]
for col in tqdm(cols, desc="convert to number"):
    df[col] = pd.to_numeric(df[col], errors="coerce")

# for col in cols:
#     df[col] = pd.to_numeric(df[col], errors="coerce")

#%%List of all columns
col_list = list(df)
print(col_list)

#%% INTEGER COLUMNS
df.select_dtypes(include=['int64'])

#%% OBJECT COLUMNS
df.select_dtypes(include=['object'])

#%%10 largest integer values
df[cols].nlargest(10, cols, keep='first').to_excel("nlarge.xlsx",startrow=1)

#%%10 smallest integer values 
df[cols].nsmallest(10, cols, keep='first').to_excel("nsmall.xlsx",startrow=1)

#%%isna() after conversion
df[cols].isna().sum().to_excel('isna.xlsx')

# %%dropna
df = df.dropna()

#%% Duplicate Records
duplicate_df = df[df.duplicated()]
print(duplicate_df)

# %%datestop/timestop astype.str(8/4)
df["datestop"] = df["datestop"].astype(str).str.zfill(8)
df["timestop"] = df["timestop"].astype(str).str.zfill(4)

# %%fx for date/timestop to y/m/d
from datetime import datetime

def make_datetime(datestop, timestop):
    year = int(datestop[-4:])
    month = int(datestop[:2])
    day = int(datestop[2:4])

    hour = int(timestop[:2])
    minute = int(timestop[2:])

    return datetime(year, month, day, hour, minute)

df["datetime"] = df.apply(
    lambda row: make_datetime(row["datestop"], row["timestop"]), axis=1
)

# %%height to cm
df["height"] = (df["ht_feet"] * 12 + df["ht_inch"]) * 2.54

# %% change x/y coord to lat/lon
import pyproj

srs = (
    "+proj=lcc +lat_1=41.03333333333333 "
    "+lat_2=40.66666666666666 +lat_0=40.16666666666666 +lon_0=-74 "
    "+x_0=300000.0000000001 +y_0=0 "
    "+ellps=GRS80 +datum=NAD83 +to_meter=0.3048006096012192 +no_defs"
)

p = pyproj.Proj(srs)

coords = df.apply(lambda r: p(r["xcoord"], r["ycoord"], inverse=True), axis=1)
df["lat"] = [c[1] for c in coords]
df["lon"] = [c[0] for c in coords]


# %% index fields from xl file
import numpy as np

value_labels = pd.read_excel(
    "2012 SQF File Spec.xlsx", sheet_name="Value Labels", skiprows=range(4)
)
value_labels["Field Name"] = value_labels["Field Name"].fillna(method="ffill")
value_labels["Field Name"] = value_labels["Field Name"].str.lower()
value_labels["Value"] = value_labels["Value"].fillna(" ")
vl_mapping = value_labels.groupby("Field Name").apply(
    lambda x: dict([(row["Value"], row["Label"]) for row in x.to_dict("records")])
)

cols = [col for col in df.columns if col in vl_mapping]

for col in tqdm(cols):
    df[col] = df[col].apply(lambda val: vl_mapping[col].get(val, np.nan))


# %%Plot height
import seaborn as sns

ax = sns.distplot(df["height"])
ax.get_figure().savefig("height.jpg")

# %%Out of range weight/ageperobs removed
df = df[(df["age"] <= 100) & (df["age"] >= 10)]
df = df[(df["weight"] <= 350) & (df["weight"] >= 50)]
df = df[(df["perobs"] <= 500)]

#%% original file size was 532,911
df.shape

# %%time Monthly count
ax = sns.countplot(df["datetime"].dt.month)
plt.title("Date Time Monthly Data")
ax.get_figure().savefig("dt.month.jpg")

# %%Day count
ax = sns.countplot(df["datetime"].dt.weekday)
ax.set_xticklabels(["Mon", "Tue", "Wed", "Thur", "Fri", "Sat", "Sun"])
ax.set(xlabel="day of week", title="# of incidents by day of weeks")
ax.get_figure().savefig("dt.day.jpg")

# %%Describe after all changes
df.describe(include="all").to_excel("statspost.xlsx", startrow=1)

# %%Hour
ax = sns.countplot(df["datetime"].dt.hour)
plt.title("Date Time Hourly Data")
ax.get_figure().savefig("dt.hour.jpg")


# %%X/Y Raw coordinates
sns.scatterplot(data=df[:100], x="xcoord", y="ycoord")

# %%Folium Convert x/y and plot v Murder
import folium

m = folium.Map((40.7128, -74.0060))

for r in df[["lat", "lon"]][df["detailcm"] == "MURDER"].to_dict("records"):
    folium.CircleMarker(location=(r["lat"], r["lon"]), radius=1).add_to(m)

m

# %%Folium plot V Terrorism

m = folium.Map((40.7128, -74.0060))

for r in df[["lat", "lon"]][df["detailcm"] == "TERRORISM"].to_dict("records"):
    folium.CircleMarker(location=(r["lat"], r["lon"]), radius=1).add_to(m)

m

# %%Count Race
ax = sns.countplot(data=df, y="race")
plt.title("Stops by Race")
ax.get_figure().savefig("race.jpg")

# %%Count Race/City
ax = sns.countplot(data=df, y="race", hue="city")
plt.title("Stops by Race and City")
ax.get_figure().savefig("race_city.jpg")

# %%Count Race/Sex
ax = sns.countplot(data=df, y="race", hue="sex")
plt.title("Stops by Race and Sex")
ax.get_figure().savefig("race_sex.jpg")

#%%Force df create/plot count
df_force = df[
    (df["pf_hands"] == "YES")
    | (df["pf_wall"] == "YES")
    | (df["pf_grnd"] == "YES")
    | (df["pf_drwep"] == "YES")
    | (df["pf_ptwep"] == "YES")
    | (df["pf_baton"] == "YES")
    | (df["pf_hcuff"] == "YES")
    | (df["pf_pepsp"] == "YES")
    | (df["pf_other"] == "YES")
]

pfs = [col for col in df.columns if col.startswith("pf_")]
pf_counts = (df[pfs] == "YES").sum()
ax = sns.barplot(y=pf_counts.index, x=pf_counts.values)
plt.title("Type of Force Use in Stop")
ax.get_figure().savefig("force_count.jpg")


#%% Race/Crime plot
ax = sns.countplot(
    data=df,
    y="detailcm",
    hue="race",
    order=df_force["detailcm"].value_counts(ascending=False).keys()[:10],
)
plt.title("Criminal Code Arrest for Stops, by Race")
ax.get_figure().savefig("race_crime_count.jpg")

#%%Sex/Crime Plot
ax = sns.countplot(
    data=df_force,
    y="detailcm",
    hue="sex",
    order=df_force["detailcm"].value_counts(ascending=False).keys()[:10],
)
plt.title("Criminal Code Arrests, by Sex")
ax.get_figure().savefig("crime_sex.jpg")

#%% Sex
ax = sns.countplot(x="sex", data=df)
_ = plt.title("Count of SQF by Sex")
ax.get_figure().savefig("sex.jpg")

#%% EDA Sex/Race
ax = sns.countplot(x="sex", hue="race", data=df)
_ = plt.title("Count of SQF by Sex and Race")
ax.get_figure().savefig("race_sex2.jpg")

#%% SQF Race/Location
ax = sns.countplot(x="trhsloc", hue="race", data=df)
plt.title("Count of SQF Stops by Gov't Location, and Race")
plt.xlabel("Public Housing or Transit Facilities")
ax.get_figure().savefig("location_hous_trans.jpg")

#%% Race/Arrests Made
ax = sns.countplot(x="arstmade", hue="race", data=df)
_ = plt.title("Count of SQF by Whether Arrest Made and Race")
plt.xlabel("Arrests Made")
ax.get_figure().savefig("arrests_race.jpg")

# %% ALL REASONS FOR STOPS BY RACE
_ = sns.catplot(x="race", hue="cs_bulge", data=df,  
    kind="count", height=4, aspect=1.3)
_ = plt.title("Stop Reason: Suspicious Bulge")
_ = sns.catplot(x="race", hue="cs_casng", data=df, 
    kind="count",height=4, aspect=1.3)
_ = plt.title("Stop Reason: Casing Victim/Location")
_ = sns.catplot(x="race", hue="cs_cloth", data=df, 
    kind="count",height=4, aspect=1.3)
_ = plt.title("Stop Reason: Crime Clothing")
_ = sns.catplot(x="race", hue="cs_descr", data=df, 
    kind="count",height=4, aspect=1.3)
_ = plt.title("Stop Reason: Fits a Description")
_ = sns.catplot(x="race", hue="cs_drgtr", data=df, 
    kind="count",height=4, aspect=1.3)
_ = plt.title("Stop Reason: Drug Transaction")

#%%
ax = sns.catplot(y="race", hue="cs_furtv", data=df, 
    kind="count",height=4, aspect=1.3)
_ = plt.title("Stop Reason: Furtive Movements")


#%%
_ = sns.catplot(x="race", hue="cs_lkout", data=df, 
    kind="count",height=4, aspect=1.3)
_ = plt.title("Stop Reason: Acting as Lookout")
_ = sns.catplot(x="race", hue="cs_objcs", data=df, 
    kind="count",height=4, aspect=1.3)
_ = plt.title("Stop Reason: Carry Suspicious Object")
_ = sns.catplot(x="race", hue="cs_other", data=df, 
    kind="count",height=4, aspect=1.3)
_ = plt.title("Stop Reason: Other Not Defined")
_ = sns.catplot(x="race", hue="cs_vcrim", data=df, 
    kind="count",height=4, aspect=1.3)
_ = plt.title("Stop Reason: Engaging Violent Crime")

#%%PRECINCT
ax =sns.countplot(data=df, y="pct", order = df['pct'].value_counts().index[:20])
_ = plt.title("Top 20 Count of SQF Cases, by Precinct ID")
_ = plt.xlabel("Count per Precinct ID")
ax.get_figure().savefig("precinct.jpg")

#%%Top/Bottom 20 Precincts
df_c = df.groupby(['pct']).size()
#df_c.rename(columns={'pct':'Precinct', '':'Count'})
df_l20 = df_c.nlargest(20).to_excel("20_high_pct.xlsx")
df_s20 = df_c.nsmallest(20).to_excel("20_low_pct.xlsx")

#"""EXPLORE CROSS RELATIONSHIPS OF IMPORTANT VARIABLES"""
#%%Cross Tab Hands/Sex
pd.crosstab(df['pf_hands'], df['sex']).to_excel("C_hands_sex.xlsx")

# %%Cross Tab Hands/Race
pd.crosstab(df['pf_hands'], df['race']).to_excel("C_hands_race.xlsx")


# %%Cross Tab Race/City
pd.crosstab(df['race'], df['city']).to_excel("C_race_city.xlsx")


# %%
forces = [col for col in df.columns if col.startswith("pf_")]
result = ["arstmade", "sumissue", "searched", "frisked"]

subset = df[forces + result]
subset = (subset == "YES").astype(int)
plt.title("Correlation Heat Map of Force Used")

fig = sns.heatmap(subset.corr())
fig.get_figure().savefig('heat.jpg')


# %%
top10_pct = df["pct"].value_counts()[:10]

palette = sns.color_palette("bright")
ax = sns.scatterplot(
    data=df[df["pct"].isin(top10_pct.index)], x="perobs", y="race", hue="pct"
)

plt.legend
plt.title("Period of Observation, by Race and Top 10 Precinct")
ax.get_figure().savefig("obs_race_pct.jpg")

# %%
#palette = sns.color_palette("bright")
#ax = sns.scatterplot(data=df, x="perobs", y="pct", hue="race")
#plt.title("Period of Observation, by Race and Precinct")
#ax.get_figure().savefig("obs_race_pct.jpg")

#%%
#sns.scatterplot(data=df, x="perobs", y="pct")


# %% drop columns that are not used in analysis
df = df.drop(
    columns=[
        # processed columns
        "datestop",
        "timestop",
        "ht_feet",
        "ht_inch",
        "xcoord",
        "ycoord",
        # not useful
        "year",
        "recstat",
        "crimsusp",
        "dob",
        "ser_num",
        "arstoffn",
        "sumoffen",
        "compyear",
        "comppct",
        "othfeatr",
        "adtlrept",
        "dettypcm",
        "linecm",
        "repcmd",
        "revcmd",
        # location of stop
        # only use coord and city
        "addrtyp",
        "rescode",
        "premtype",
        "premname",
        "addrnum",
        "stname",
        "stinter",
        "crossst",
        "aptnum",
        "state",
        "zip",
        "addrpct",
        "sector",
        "beat",
        "post",
    ]
)

# %% modify one column
df["trhsloc"] = df["trhsloc"].fillna("NEITHER")

# %% remove all rows with NaN
df = df.dropna()

#%%Rate of Force
pf_counts.sum()/df['sex'].count()

# %% save dataframe to a pkl file
df.to_pickle("data.pkl")



# %%Done
print ("done")
# %%

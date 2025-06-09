import calendar
from scipy.optimize import curve_fit
from scipy.stats import linregress 
import statsmodels.api as sm 
import datetime
import matplotlib.dates as pltdates
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
from matplotlib.ticker import MaxNLocator
from typing import Optional 

# Global graph formatting. 
bg_color = "seashell"
plt.rcParams["figure.facecolor"] = bg_color
plt.rcParams["axes.facecolor"] = bg_color

# Træningsvariabler 
vægt = 67
alder = 34
# Løb 
løb_1km = vægt*4.7 # Konstanten er estimeret kJ brugt på løb 5-6 min/km.  
# Energi i 1 kg kropsvægt
energi_vægt = 32_200 # kJ/kg
# Styrketræning
styrke_1min = 18 # kJ forbrændt på 1 minut i styrketræning
# Konversion fra kcal til kJ:
kcal_kj = 4.184

# Tid og perioder 
first_day = datetime.date(2022, 8, 22)
today = datetime.date.today()
current_month = datetime.date.today().month
current_year = datetime.date.today().year 
period_delta = today - first_day 
period_days = period_delta.days
period_months = period_days/(365/12)

##################### HEARTRATE ZONES ########################
def print_heart_zones(age, RHR=60, MHR_obs=None):
    # Determine Maximum Heart Rate: use observed if provided, else theoretical
    MHR_theoretical = 220 - age
    MHR = MHR_obs if MHR_obs is not None else MHR_theoretical
    HRR = MHR - RHR

    # Define zones: (lower_fraction, upper_fraction)
    pct_zones = {
        1: (0.0, 0.6),   # Zone 1: < 60% MHR
        2: (0.6, 0.7),   # Zone 2: 60-70%
        3: (0.7, 0.8),   # Zone 3: 70-80%
        4: (0.8, 0.9),   # Zone 4: 80-90%
        5: (0.9, 1.0),   # Zone 5: 90-100%
    }

    # Karvonen zones use same fractions but on HRR
    karo_zones = pct_zones.copy()

    # Descriptions for each zone
    descriptions = {
        1: "Recovery: very easy, promotes blood flow and clearing",
        2: "Endurance: easy-steady, builds aerobic base",
        3: "Tempo: comfortably hard, sustainable 45-90 min",
        4: "Threshold: hard, sustainable 20-30 min",
        5: "VO₂ Max: near all-out, sustainable 3-8 min intervals",
    }

    # Header
    print(f"Theoretical MHR: {MHR_theoretical:.0f} bpm")
    if MHR_obs is not None:
        print(f"Observed MHR:   {MHR_obs} bpm")
    print(f"Resting HR (RHR): {RHR} bpm\n")

    # Loop through zones 1-5
    for z in range(1, 6):
        lo_pct, hi_pct = pct_zones[z]
        lo_kf, hi_kf = karo_zones[z]

        lo_bpm_pct = lo_pct * MHR
        hi_bpm_pct = hi_pct * MHR
        lo_bpm_karo = RHR + lo_kf * HRR
        hi_bpm_karo = RHR + hi_kf * HRR

        print(f"Zone {z} ({descriptions[z]}):")
        print(f"  % of Max {int(lo_pct*100)}–{int(hi_pct*100)}%:"
              f" {round(lo_bpm_pct)}–{round(hi_bpm_pct)} bpm")
        print(f"  Karvonen {int(lo_kf*100)}–{int(hi_kf*100)}% HRR:"
              f" {round(lo_bpm_karo)}–{round(hi_bpm_karo)} bpm\n")

##################### SOURCE DATAFRAME #######################

df = pd.read_csv('training_data.csv', sep=',')
# Gør tid til datetime midlertidigt. 
df['tid'] = pd.to_datetime(df['tid'], format='%H:%M:%S')
# Beregn tid i minutter som float. 
df['t float'] = df['tid'].dt.hour * 60 + df['tid'].dt.minute + df['tid'].dt.second / 60
# Formater tid tilbage til string.
df['tid'] = df['tid'].dt.strftime('%H:%M:%S')
# Formater datokolonne. 
df['dato'] = pd.to_datetime(df['dato'], format="%d-%m-%Y")
# Sorter data.
df = df.sort_values(by="dato", ascending=False)
# Konverter km til numeriske værdier og X til NaN. 
df['km'] = pd.to_numeric(df['km'], errors='coerce')
# Lav kolonne med forbrændt kJ.
df['kj'] = df.apply(lambda row: row['km'] * løb_1km if row['navn'] == 'Løb' else row['t float'] * styrke_1min, axis=1)
# Lav kolonne med min/km for løb.
df['min/km'] = df.apply(lambda row: row['t float'] / row['km'] if row['navn'] == 'Løb' else 0, axis=1)
df['pace'] = pd.to_datetime(df['min/km'], unit='m').dt.strftime('%H:%M:%S')
# Arranger vigtigste kategorier først.
df = df[["navn", "dato", 'kategori', "km", "min/km", "pace", "tid", "HR", "kj", "lokation", "præcis tid", "t float", "split"]]
# Slet heart rate observationer før 30-10-2024, fordi de er falske. 
df.loc[df["dato"] < "30-10-2024", "HR"] = np.nan 

# Dataframe fra 2026 og frem. 
df_2026 = pd.read_csv("training_data_2026plus.csv", sep=",")
df_2026['tid'] = pd.to_datetime(df_2026['tid'], format='%H:%M:%S')
# Beregn tid i minutter som float. 
df_2026['t float'] = df_2026['tid'].dt.hour * 60 + df_2026['tid'].dt.minute + df_2026['tid'].dt.second / 60
# Formater tid tilbage til string.
df_2026['tid'] = df_2026['tid'].dt.strftime('%H:%M:%S')
# Formater datokolonne. 
df_2026['dato'] = pd.to_datetime(df_2026['dato'], format="%d-%m-%Y")
# Sorter data.
df_2026 = df_2026.sort_values(by="dato", ascending=False)
# Konverter km til numeriske værdier og X til NaN. 
df_2026['km'] = pd.to_numeric(df_2026['km'], errors='coerce')
# Lav kolonne med forbrændt kJ.
df_2026['kj'] = df_2026.apply(lambda row: row['km'] * løb_1km if row['navn'] == 'Løb' else row['t float'] * styrke_1min, axis=1)
df_2026["kj"] = df_2026["kj"].convert_dtypes(convert_integer=True)
# Lav kolonne med min/km for løb.
df_2026['min/km'] = df_2026.apply(lambda row: row['t float'] / row['km'] if row['navn'] == 'Løb' else 0, axis=1)
df_2026['pace'] = pd.to_datetime(df_2026['min/km'], unit='m').dt.strftime('%H:%M:%S')
# Konverter heartrate til integer.
df_2026["HR"] = df_2026["HR"].convert_dtypes(convert_integer=True)
# Arranger vigtigste kategorier først.
df_2026 = df_2026[["navn", "dato", 'kategori', "km", "min/km", "pace", "tid", "HR", "kj", "t float", "split"]]

# Stack de to dataframes. 
df = pd.concat([df, df_2026], ignore_index=True)

# Splits.
pattern = r"(?P<distance>[\d\.]+)\s*km\s*(?P<duration>\d{2}:\d{2}:\d{2})"
df[["split distance", "split tid str"]] = df["split"].str.extract(pattern)
df["split km"] = df["split distance"].astype(float)
df["split tid"] = pd.to_datetime(df["split tid str"], format="%H:%M:%S")
df = df.drop(columns=["split", "split tid str"])
df['split tid'] = pd.to_datetime(df['split tid'], format='%H:%M:%S')
df['split t float'] = df['split tid'].dt.hour * 60 + df['split tid'].dt.minute + df['split tid'].dt.second / 60
df['split min/km'] = df.apply(lambda row: row['split t float'] / row['split km'] if row['navn'] == 'Løb' else 0, axis=1)
df['split pace'] = pd.to_datetime(df['split min/km'], unit='m').dt.strftime('%H:%M:%S')
df['split tid'] = df['split tid'].dt.strftime('%H:%M:%S')

###################### CATEGORY DATAFRAMES #######################

# Dataframe til løb.
df_løb = df[df['navn'] == 'Løb'].copy()
df_løb = df_løb.sort_values(by="dato", ascending=False)
# Dataframe til styrketræning.
df_styrke = df[df['navn'] == 'Træning'].copy()

df_løb['km'] = df_løb['km'].astype(float) # Formater km. 
df_løb.set_index('dato', inplace=True) # Sæt dato som index.
cmlative_km = df_løb.resample('M')['km'].sum() # Gruppér efter måned og beregn summer.

# Reset index for at få dato som kolonne igen.
cmlative_km = cmlative_km.reset_index() 
df_løb = df_løb.reset_index() 

# Dataframe til plot.
cmlative_km_plt = cmlative_km[-12:] # Seneste 12 måneder.
cmlative_km_record = cmlative_km.copy()
cmlative_km_record = cmlative_km_record.sort_values(by='km', ascending=False)
cmlative_km_record['place'] = np.arange(1, len(cmlative_km_record) + 1) # Tilføj kolonne med placering.

# Dataframe til styrketræning.
df_styrke['t float'] = df_styrke['t float'].astype(float) # Formater tid. 
df_styrke.set_index('dato', inplace=True) # Sæt dato som index.

cmlative_str = df_styrke.resample('M')['t float'].cumsum() # Gruppér efter måned og beregn summer.
# Reset index for at få dato som kolonne igen.
cmlative_str = cmlative_str.reset_index()[-80:] # Medtag kun de seneste 80 rækker. 

# Gennemsnit km/måned
avg_km_måned = cmlative_km_plt['km'][0:-1].sum() / 11

############################## OLD CATEGORIES ##########################

# # Dataframe for løb med præcis tid og uden Reconstitution eller Distance. 
# df_løb_spd = df_løb.copy()
# df_løb_spd = df_løb_spd[
#                         (df_løb_spd['kategori'] == 'Tempo')
#                         & (df_løb_spd['min/km'] != 0) 
#                         & (df_løb_spd['præcis tid'] == 'Y') 
#                         # Kun tempotræning på distance under 10.01 km. 
#                         & (df_løb_spd['km'] < 10.01) 
#                         ]

# df_løb_recon = df_løb.copy()
# df_løb_recon = df_løb_recon[df_løb_recon['kategori'] == 'Reconstitution']

# # Dataframe for distance med præcis tid. 
# df_løb_dist = df_løb.copy()
# df_løb_dist = df_løb_dist[(df_løb_dist['kategori'] == 'Distance') & (df_løb_dist['præcis tid'] == 'Y') 
#     & (df_løb_dist['min/km'] != 0)]

# Dataframe til ugedistancer.
df_løb['km'] = df_løb['km'].astype(float) # Formater km. 
df_løb.set_index('dato', inplace=True) # Sæt dato som index.
uge_løb_df = df_løb.resample('W', closed="right")['km'].sum() # Gruppér efter måned og beregn summer.
uge_løb_df = uge_løb_df.reset_index() # Reset index for at få dato som kolonne igen.
df_løb = df_løb.reset_index()

# Formater dato til at vise dagens navn.
df_løb["dag_dato"] = df_løb["dato"].dt.strftime("%A, %d-%m-%Y")

# 2025 CPH Marathon training block: weekly distances. 
mara25_block = uge_løb_df[(uge_løb_df["dato"] >= pd.Timestamp(2025, 2, 23)) & (uge_løb_df["dato"] <= pd.Timestamp(2025, 5, 11))]

########################## NEW CATEGORIES #################################

##### 5 km
df_5k = df_løb.copy()
df_5k = df_5k[
    (df_5k["km"] <= 5.1)
    & (df_5k["min/km"] > 0.1)
    & ((df_5k["kategori"] == "Tempo") | (df_5k["kategori"] == "pace"))
]
# Filter out imprecise entries.
cutoff_date = pd.Timestamp(2025, 5, 11)
mask = ((df_5k["dato"] < cutoff_date) & (df_5k["præcis tid"] == "Y")) | (df_5k["dato"] >= cutoff_date)
df_5k = df_5k.loc[mask]

# Compute rank of pace runs. 
df_5k["rank"] = df_5k["min/km"].sort_values(ascending=False).rank(method="first").astype(int)
cols = ["rank"] + [col for col in df_5k.columns if col != 'rank']

def show_5k(chrono: bool=None, rank: bool=False, n_rows=10):
    df_5k_chrono = df_5k_rank = df_5k[["rank", "dag_dato", "kategori", "km", "pace", "tid", "HR", "kj"]]
    if chrono:
        return df_5k_chrono.head(n_rows).style.hide(axis="index")
    else:
        df_5k_rank = df_5k_rank.sort_values(by="pace", ascending=True)
        return df_5k_rank.head(n_rows).style.hide(axis="index")

##### 10 km 
df_10k = df_løb.copy()
b = 1.4 # A distance bound on runs to include. 
df_10k = df_10k[
    ((df_10k["km"] >= 10 - b) & (df_10k["km"] <= 10 + b))
    & (df_10k["min/km"] > 0.1)
    & ((df_10k["kategori"] == "Tempo") | (df_10k["kategori"] == "Distance") | (df_10k["kategori"] == "intervals")
       | (df_10k["kategori"] == "pace") | (df_10k["kategori"] == "progression"))
]
# Filter out imprecise entries.
cutoff_date = pd.Timestamp(2025, 5, 11)
mask = ((df_10k["dato"] < cutoff_date) & (df_10k["præcis tid"] == "Y")) | (df_10k["dato"] >= cutoff_date)
df_10k = df_10k.loc[mask]

# Compute rank of pace runs. 
df_10k["rank"] = df_10k["min/km"].sort_values(ascending=False).rank(method="first").astype(int)
cols = ["rank"] + [col for col in df_10k.columns if col != 'rank']

def show_10k(chrono: bool=None, rank: bool=False, n_rows=10):
    df_10k_chrono = df_10k_rank = df_10k[["rank", "dag_dato", "kategori", "km", "pace", "tid", "HR", "kj"]]
    if chrono:
        return df_10k_chrono.head(n_rows).style.hide(axis="index")
    else:
        df_10k_rank = df_10k_rank.sort_values(by="rank", ascending=True)
        return df_10k_rank.head(n_rows).style.hide(axis="index")

##### 15 km
df_15k = df_løb.copy()
df_15k = df_15k[
    ((df_15k["km"] >= 12.5) & (df_15k["km"] <= 17.5))
    & (df_15k["min/km"] > 0.1)
    & ((df_15k["kategori"] == "Distance") | (df_15k["kategori"] == "easy") | (df_15k["kategori"] == "long") 
       | (df_15k["kategori"] == "progression") | (df_15k["kategori"] == "strides") | (df_15k["kategori"] == "pace"))
]

cutoff_date = pd.Timestamp(2025, 5, 11)
mask = ((df_15k["dato"] < cutoff_date) & (df_15k["præcis tid"] == "Y")) | (df_15k["dato"] >= cutoff_date)
df_15k = df_15k.loc[mask]

df_15k["rank"] = df_15k["min/km"].sort_values(ascending=False).rank(method="first").astype(int)
cols = ["rank"] + [col for col in df_15k.columns if col != 'rank']

def show_15k(rank: bool=True, n_rows=10):
    df_15k_rank = df_15k[["rank", "dag_dato", "kategori", "km", "pace", "tid", "HR", "kj"]]
    if rank:
        df_15k_rank = df_15k_rank.sort_values(by="rank")
        return df_15k_rank.head(n_rows).style.hide(axis="index")
    

########################## Monthly distance projection ####################
# Projektion for hvad jeg vil løbe resten af måneden.
first_day_proj = today.replace(day=1)
remaining_days = datetime.date(current_year, current_month\
    , calendar.monthrange(current_year, current_month)[1]) - today

def month_distance_projection():
    default_proj = avg_km_måned
        
    sum_proj = cmlative_km_plt.iloc[-1][1]
    days_delta_proj = today - first_day_proj + datetime.timedelta(days=1)
        
    # Udregn projektionen baseret på gennemsnittet denne måned.
    month_projection = sum_proj + (int(sum_proj) / int(days_delta_proj.days)) * int(remaining_days.days)

    if cmlative_km_plt.iloc[-1][0].month != current_month:
        return default_proj
    else: 
        return month_projection

########################## Weekly distance graph ########################
def week_block_graph(title: str, df: Optional[pd.DataFrame] = None, n_weeks: Optional[int] = None):
   
    if df is None: 
        df = uge_løb_df 
    if n_weeks is None: 
        n_weeks = 12
    
    df = df[-n_weeks:].sort_values(by='dato', ascending=False)

    fig, ax = plt.subplots(figsize=(7,5))

    y_max = 90

    x = df["dato"]
    y = df["km"]

    # Ugedistancer.
    ax.plot(x, y, color="mediumseagreen", drawstyle="steps-post")
    ax.fill_between(x, y, step="post", hatch="///", color="mediumseagreen", alpha=0.3)

    # y-værdier. 
    for i in range(len(x)):
        plt.text(x.iloc[i], y.iloc[i]+1.5, str(round(y.iloc[i], 1)), ha='right', va="baseline", fontsize=10.4)

    # Gennemsnit.
    mean_y = np.mean(y)
    ax.axhline(mean_y, color="deeppink", linewidth=2, linestyle=":", label=f"Gns. = {round(mean_y,1)}")

    # Total distance.
    sum_y = np.sum(y)
    plt.text(x.iloc[n_weeks-1], y_max - (y_max*0.06), f"Total distance = {round(sum_y,1)}", fontsize=10.4)

    plt.title(title)
    ax.legend()
    ax.set_ylim(0, y_max)
    ax.grid(True, axis="x", linestyle='--', linewidth=0.4, color='gray')
    date_format_løb = pltdates.DateFormatter('%d-%m')
    ax.xaxis.set_major_formatter(date_format_løb)
    ax.set_xticks(x)

############################## Yearly running distances ##########################
def yearly_distance(year: int):
    df_år_løb = df_løb[df_løb["dato"].dt.year == year]
    år_total = df_år_løb["km"].cumsum()
    print(f"Kilometer i {year}: ", round(år_total.values[-1], 1))

############################ Monthly running distances ##########################
def monthly_distance_graph():
    fig, ax = plt.subplots(figsize=(7,5))

    # Plot månedssummer. 
    ax.plot(cmlative_km_plt['dato'], cmlative_km_plt['km'], color='royalblue', drawstyle='steps-pre', label='Total distance')
    ax.fill_between(cmlative_km_plt['dato'], cmlative_km_plt['km'], color='royalblue', step="pre", alpha=0.3, hatch="...")

    # Tilføj km-værdier. 
    for x, y in zip(cmlative_km_plt['dato'], cmlative_km_plt['km']):
        ax.text(x, y, str(round(y,1)), horizontalalignment='right', verticalalignment="bottom", fontsize=10.2)

    # Linje til gennemsnit. 
    ax.axhline(avg_km_måned, color='deeppink', linestyle="-.", linewidth=2, label=f'gns. km/måned = {round(avg_km_måned, 1)}')

    # Plot fremskrivning af løbedistance for denne måned baseret på km løbet indtil nu.
    ax.hlines(month_distance_projection(), pltdates.date2num(first_day_proj), pltdates.date2num(today+remaining_days), 
        color='mediumseagreen', linestyle='-', linewidth=2)

    # Plotindstillinger.  
    ax.set_title("Running - 1 year monthly distances")
    ax.grid(True, axis='y', linestyle='--', linewidth=0.4, color='gray')
    ax.set_ylim(0,200)
    date_format_løb = pltdates.DateFormatter('%m')
    ax.xaxis.set_major_formatter(date_format_løb)
    ax.legend(loc='upper left')

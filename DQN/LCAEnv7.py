import io
import pstats
import cProfile
import datetime
from datetime import date
from numba import njit
import numba
import pprint
import csv
import time as mytime
import scipy.stats as ss
import matplotlib.pyplot as plt
from numpy import math
import pandas as pd
import numpy as np
import os


rundate = str(datetime.datetime.now().strftime("%Y-%m-%d %H'%M''"))
print("code run on", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# In[2]:


def profile(fnc):

    file = f"/code profiles/code profile {rundate}.txt"

    """A decorator that uses cProfile to profile a function"""

    def inner(*args, **kwargs):

        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        with open(file, 'w') as output:
            output.write(s.getvalue())
        return retval

    return inner


# # Constants

# In[3]:


MPH_TO_MPS = 0.44704  # Convert miles per hour to meters per second
MPERKM_TO_INPERMI = 63.36  # Convert meters per km to in per mile
INCH_TO_METER = 0.0254
FT_TO_METER = 0.3048
MILE_TO_METER = 1609.34
MILE_TO_KM = MILE_TO_METER/1000
kg_to_tons = 0.0011023113109244  # todo: what kind of ton is this?
SQMETRE_TO_SQYARD = 1.19599005

MEGA_TO_BASE = pow(10, 6)
GIGA_TO_BASE = pow(10, 9)
KILO_TO_GIGA = pow(10, -6)
GIGA_TO_MEGA = pow(10, 3)

NRE_TO_GWP_LDV = 3.942429712/58.07174308  # convert NRE in MJ to GWP in kgCO2e
NRE_TO_GWP_HDV = 3.751262042/55.26691605  # convert NRE in MJ to GWP in kgCO2e


ALPHA = 0.29  # for deflection function

GWP_GAL_GAS = 3.942429712  # GWP in kg CO2e per gallon of gasoline burned
GWP_GAL_DIESEL = 3.751262042  # GWP in kg CO2e per gallon of diesel burned

global ANALYSIS_PERIOD
ANALYSIS_PERIOD = 50  # years

global RUN_NAME  # a single variable so that all outputs are named with this
RUN_NAME = "Standard"

# Paths so user can have 1 folder for all input files
DEPENDANCY_PATH = 'LCA Dependancies'
OUTPUT_PATH = 'LCA Results'
IMAGE_OUTPUT_PATH = OUTPUT_PATH + '/Graphs'

# create folder if it doesn't exist
os.makedirs(os.path.dirname(OUTPUT_PATH+'/'), exist_ok=True)
os.makedirs(os.path.dirname(IMAGE_OUTPUT_PATH+'/'), exist_ok=True)


# # Vehicle & Fleet Characteristics

# ## Load Vehicle Characteristics

# In[4]:


### Load and clean mass and emissions from spreadsheet
file = DEPENDANCY_PATH + "/ME_data_Nov13.xlsx"
df_vehicles = pd.read_excel(file , sheet_name = "Arch", usecols = "A,B,N,O", nrows = 24, header = 2,  engine = "openpyxl")
#df_vehicles = pd.read_excel("ME_data_Nov13.xlsx" , usecols = "B:E", nrows = 24,  engine = "openpyxl")
column_names = ["vehicle","mass","fuel consumption","elec consumption"]
df_vehicles.columns = column_names
#note fuel is fuel/hydrogen todo: clarify that we need hydrogen  emission factor
kWh100km_to_MJ1mi = 3.6 / (100 * 0.62) # convert kwh/100km to MJ/mi
df_vehicles['fuel consumption'] *= kWh100km_to_MJ1mi
df_vehicles['elec consumption'] *= kWh100km_to_MJ1mi
#conver NaN for fuels to 0
df_vehicles.replace(np.NaN,0,inplace=True)


#add new columns for axle weights
df_vehicles['axle-1'] = 0
df_vehicles['axle-2'] = 0
df_vehicles['axle-3'] = 0
df_vehicles['axle-4'] = 0
df_vehicles['axle-5'] = 0
#assign axle weights from previous proportions for now
kg_to_N = 9.81

#assign to qingshi vehicle weights (all passenger vehicles with no more than 2 axles)
df_vehicles['axle-1'] = df_vehicles['mass']/2*kg_to_N
df_vehicles['axle-2'] = df_vehicles['mass']/2*kg_to_N

#add on dummy column to simplify gwp calcs
df_vehicles['class'] = ''
df_vehicles['technology'] = ''
classes = ['micro','PC','SUV','LT'] #this nomenclature is from qingshi paper. FHWA wise they are all passenger vehicles! todo
technologies = ['ICEV-g','ICEV-d','HEV','PHEV','BEV','HFCEV']
for veh_class in classes:
    df_vehicles['class'] = np.where(~df_vehicles['vehicle'].str.contains(veh_class),df_vehicles['class'],'pc')

for technology in technologies:
    df_vehicles['technology'] = np.where(~df_vehicles['vehicle'].str.contains(technology),df_vehicles['technology'],technology)

    

############ Axle Loads ###############
file = DEPENDANCY_PATH + "/vehicle weights.xlsx"
df_axles = pd.read_excel(file,engine = "openpyxl")
weight_cols = df_axles.columns.to_list()[1:]
df_axles = df_axles[weight_cols].mul(4448.2216) #kip to N
LT_axles = df_axles[['C4','C5']].mean(axis=1) #takes row wise average of selected rows, ignoring N/A
MT_axles = df_axles[['C6','C7','C8']].mean(axis=1)
HT_axles = df_axles[['C9','C10']].mean(axis=1) #have 11,12,13, we can include? #Todo
############ Axle Loads ###############



############ Fuel Economy ###############
#todo: this should probably become a function that can scale as these trucks become EV's?
MPG_LDV = 35    # MPG (highway) for an average car, gasoline
MPG_HDV = 1/0.139832384 # MPG for an average truck, diesel fuel
MJ_per_mile_LDV = 1/(MPG_LDV*0.03/3.6 * MILE_TO_KM) #MPG*0.03gallon/kwh then kwh*3.6MJ/KWH (in denominator)
MJ_per_mile_HDV = 1/(MPG_HDV*0.027/3.6 * MILE_TO_KM) #0.027gal/kwh for diesel, assume diesel for HDV
############ Fuel Economy ###############



FHWA_VEHICLES = [['ICEV-g FHWA LT',0,MJ_per_mile_LDV,0,0,0,0,0,0,'FHWA LT','ICEV-g'],
                 ['ICEV-d FHWA LT',0,MJ_per_mile_LDV,0,0,0,0,0,0,'FHWA LT','ICEV-d'],
                 ['HEV FHWA LT',0,MJ_per_mile_LDV,0,0,0,0,0,0,'FHWA LT','HEV'],
                 ['PHEV FHWA LT',0,MJ_per_mile_LDV,0,0,0,0,0,0,'FHWA LT','PHEV'],
                 ['BEV FHWA LT',0,0,0,0,0,0,0,0,'FHWA LT','BEV'],
                 ['HFCEV FHWA LT',0,0,0,0,0,0,0,0,'FHWA LT','HFCEV'],
                 
                 ['ICEV-g FHWA MT',0,MJ_per_mile_HDV,0,0,0,0,0,0,'FHWA MT','ICEV-g'],
                 ['ICEV-d FHWA MT',0,MJ_per_mile_HDV,0,0,0,0,0,0,'FHWA MT','ICEV-d'],
                 ['HEV FHWA MT',0,MJ_per_mile_HDV,0,0,0,0,0,0,'FHWA MT','HEV'],
                 ['PHEV FHWA MT',0,MJ_per_mile_HDV,0,0,0,0,0,0,'FHWA MT','PHEV'],
                 ['BEV FHWA MT',0,0,0,0,0,0,0,0,'FHWA MT','BEV'],
                 ['HFCEV FHWA MT',0,0,0,0,0,0,0,0,'FHWA MT','HFCEV'],
                 
                 ['ICEV-g FHWA HT',0,MJ_per_mile_HDV,0,0,0,0,0,0,'FHWA HT','ICEV-g'],
                 ['ICEV-d FHWA HT',0,MJ_per_mile_HDV,0,0,0,0,0,0,'FHWA HT','ICEV-d'],
                 ['HEV FHWA HT',0,MJ_per_mile_HDV,0,0,0,0,0,0,'FHWA HT','HEV'],
                 ['PHEV FHWA HT',0,MJ_per_mile_HDV,0,0,0,0,0,0,'FHWA HT','PHEV'],
                 ['BEV FHWA HT',0,0,0,0,0,0,0,0,'FHWA HT','BEV'],
                 ['HFCEV FHWA HT',0,0,0,0,0,0,0,0,'FHWA HT','HFCEV'],
                ]

#append the list of lists to the dataframe
for i in range(len(FHWA_VEHICLES)):
    df_vehicles.loc[len(df_vehicles)] = FHWA_VEHICLES[i]
    
print(df_vehicles)    
print(LT_axles)    
### Assign Weights
df_vehicles.loc[df_vehicles['class'] == 'FHWA LT','axle-1'] = LT_axles.iloc[0]
df_vehicles.loc[df_vehicles['class'] == 'FHWA LT','axle-2'] = LT_axles.iloc[1]
df_vehicles.loc[df_vehicles['class'] == 'FHWA LT','axle-3'] = LT_axles.iloc[2]

df_vehicles.loc[df_vehicles['class'] == 'FHWA MT','axle-1'] = MT_axles.iloc[0]
df_vehicles.loc[df_vehicles['class'] == 'FHWA MT','axle-2'] = MT_axles.iloc[1]
df_vehicles.loc[df_vehicles['class'] == 'FHWA MT','axle-3'] = MT_axles.iloc[2]
df_vehicles.loc[df_vehicles['class'] == 'FHWA MT','axle-4'] = MT_axles.iloc[3]


df_vehicles.loc[df_vehicles['class'] == 'FHWA HT','axle-1'] = HT_axles.iloc[0]
df_vehicles.loc[df_vehicles['class'] == 'FHWA HT','axle-2'] = HT_axles.iloc[1]
df_vehicles.loc[df_vehicles['class'] == 'FHWA HT','axle-3'] = HT_axles.iloc[2]
df_vehicles.loc[df_vehicles['class'] == 'FHWA HT','axle-4'] = HT_axles.iloc[3]

df_vehicles.sort_values(by='technology')


# ### Get_Truck_Weights

# In[5]:


#display the loaded information
#typical axle makeups from 47 WIM stations study
file = DEPENDANCY_PATH + "/vehicle weights.xlsx"
df_axle_makeup = pd.read_excel(file, engine='openpyxl')
# load fit params from R
file = DEPENDANCY_PATH + "/Rfit-LTPP2.csv"
df_fits = pd.read_csv(file)
df_lbs = pd.DataFrame()
#conversion factor
LB_TO_N = 4.4482216
df_fits = df_fits.transpose()
df_fits.columns = df_fits.iloc[2] #reassign correct row as headers
df_fits.drop(index = 'dataset',inplace=True) #remove the row to clean df
FIT_CLASSES = df_fits.columns.to_list()[2:9] #look only at esals for veh class 4-10

axle_split = pd.DataFrame()

def Get_Truck_Weights():

    for col in FIT_CLASSES:

        column = df_fits[col]

        param1 = column[0]
        param2 = column[1]

        best_fit = column[2] #this method is 3x faster than .loc


        if best_fit == 'LogNormal': #param1 = meanlog, param2 = sdlog
            predict = np.random.lognormal(param1,param2)

        elif best_fit == 'Weibull': #param1 = shape, param2 = scale
            predict = np.random.weibull(param1)*param2
            #note numpy does not take in scale, so just multiply
        else: 
            predict = np.random.gamma(param1,param2) #param1 = shape, param2 = rate or scale

        #add the mean to the dataframe to


        weight = 18000*predict #todo: once decided we are using this formula, convert entire DF to Newtons upon loading


        df_lbs.loc['weight',col] = weight

    for col in FIT_CLASSES:

        veh_class = col.split('_')[-1] #renaming the columns once will speedup
        index = 'C' + veh_class
        class_weight = df_lbs[col]
        class_weight = class_weight[0]

        for axle in range(sum(df_axle_makeup[index].notna())):

            total_ax = sum(df_axle_makeup[index].notna())

            axle_split.loc[axle,index]= class_weight
            
            
            
            
            
            
    #average the weights and assign to trucks
            
    weight_cols = axle_split.columns.to_list()[1:]
    df_axles = axle_split[weight_cols].mul(4.4482216282509) #lb to N
    LT_axles = axle_split[['C4','C5']].mean(axis=1) #takes row wise average of selected rows, ignoring N/A
    MT_axles = axle_split[['C6','C7','C8']].mean(axis=1)
    HT_axles = axle_split[['C9','C10']].mean(axis=1) #have 11,12,13, but excluded due to rarity of such large vehicles in aadtt


    ### Assign Weights
    df_vehicles.loc[df_vehicles['class'] == 'FHWA LT','axle-1'] = LT_axles.iloc[0]/3
    df_vehicles.loc[df_vehicles['class'] == 'FHWA LT','axle-2'] = LT_axles.iloc[1]/3
    df_vehicles.loc[df_vehicles['class'] == 'FHWA LT','axle-3'] = LT_axles.iloc[2]/3

    df_vehicles.loc[df_vehicles['class'] == 'FHWA MT','axle-1'] = MT_axles.iloc[0]/4
    df_vehicles.loc[df_vehicles['class'] == 'FHWA MT','axle-2'] = MT_axles.iloc[1]/4
    df_vehicles.loc[df_vehicles['class'] == 'FHWA MT','axle-3'] = MT_axles.iloc[2]/4
    df_vehicles.loc[df_vehicles['class'] == 'FHWA MT','axle-4'] = MT_axles.iloc[3]/4

    df_vehicles.loc[df_vehicles['class'] == 'FHWA HT','axle-1'] = HT_axles.iloc[0]/4
    df_vehicles.loc[df_vehicles['class'] == 'FHWA HT','axle-2'] = HT_axles.iloc[1]/4
    df_vehicles.loc[df_vehicles['class'] == 'FHWA HT','axle-3'] = HT_axles.iloc[2]/4
    df_vehicles.loc[df_vehicles['class'] == 'FHWA HT','axle-4'] = HT_axles.iloc[3]/4

    
    return 





Get_Truck_Weights()  
print("ESAL's broadcast to number of axles")
axle_split

#df_vehicles


# ## Load Fleet Composition

# In[6]:


#load fleet composition
file = DEPENDANCY_PATH + "/FleetChange.xlsx"
df_fleet = pd.read_excel(file , sheet_name = "Model_Results", usecols = "A,F:BA", nrows = 42, header = 0,  engine = "openpyxl",)

#clean the dataframe
drop_rows = np.arange(0,6)
df_fleet.drop(drop_rows,inplace=True)
df_fleet.index += 2


#TEMPORARY EXTEND DATAFRAME TO 2065
col_to_copy = df_fleet[2060]
for x in range(1,6):
    df_fleet[x+2060] = col_to_copy
    
df_fleet['scenario'] = None

#define scenarios
# 1 LED + Baseline
# 2 LED + RCP2.6
# 3 SSP1 + Baseline
# 4 SSP1 + RCP 2.6
# 5 SSP2 + Baseline
# 6 SSP2 + RCP2.6

df_fleet.loc[(df_fleet['SocEc scen'] == 'LED') & (df_fleet['ClimPol scen'] == 'Baseline(unmitigated)'),'scenario'] = 1
df_fleet.loc[(df_fleet['SocEc scen'] == 'LED') & (df_fleet['ClimPol scen'] == 'RCP2.6'),'scenario'] = 2
df_fleet.loc[(df_fleet['SocEc scen'] == 'SSP1') & (df_fleet['ClimPol scen'] == 'Baseline(unmitigated)'),'scenario'] = 3
df_fleet.loc[(df_fleet['SocEc scen'] == 'SSP1') & (df_fleet['ClimPol scen'] == 'RCP2.6'),'scenario'] = 4
df_fleet.loc[(df_fleet['SocEc scen'] == 'SSP2') & (df_fleet['ClimPol scen'] == 'Baseline(unmitigated)'),'scenario'] = 5
df_fleet.loc[(df_fleet['SocEc scen'] == 'SSP2') & (df_fleet['ClimPol scen'] == 'RCP2.6'),'scenario'] = 6


#calculate relative fleet composition
year_cols = df_fleet.columns[3:-1] # get the year columns
#todo: .loc scenario once, thens slice on year
for scenario in range(1,6+1):
    for year in year_cols:
        df_fleet.loc[df_fleet['scenario'].to_numpy() == scenario,year] = df_fleet.loc[df_fleet['scenario'].to_numpy() == scenario,year]/df_fleet.loc[df_fleet['scenario'].to_numpy() == scenario,year].sum()

        
#add a column for easy filtering
#df_fleet['technology'] = None
technologies = ['ICEV-g','ICEV-d','HEV','PHEV','BEV','HFCEV']
df_fleet.loc[(df_fleet['Indicator'] == 'In-use stock, Internal Combustion Engine, gasoline (ICEG)'),'technology'] = 'ICEV-g'
df_fleet.loc[(df_fleet['Indicator'] == 'In-use stock, Internal Combustion Engine, diesel (ICED)'),'technology'] = 'ICEV-d'
df_fleet.loc[(df_fleet['Indicator'] == 'In-use stock, Hybrid Electric Vehicles (HEV)'),'technology'] = 'HEV'
df_fleet.loc[(df_fleet['Indicator'] == 'In-use stock, Plugin Hybrid Electric Vehicles (PHEV)'),'technology'] = 'PHEV'
df_fleet.loc[(df_fleet['Indicator'] == 'In-use stock, Battery Electric Vehicles (BEV)'),'technology'] = 'BEV'
df_fleet.loc[(df_fleet['Indicator'] == 'In-use stock, Fuel Cell Vehicles (FCV)'),'technology'] = 'HFCEV'

        
#display results
#for scenario in range(1,6+1):
#    df_plot = df_fleet.loc[df_fleet['scenario'] == scenario]
#    transpose_cols = df_plot['Indicator'].tolist()
#    df_plot = df_plot.transpose().iloc[3:-2]
#    df_plot.columns = transpose_cols
#    df_plot.plot(title = f'scenario {scenario}')
    



#show result of scenario 1
df_fleet.loc[df_fleet['scenario'] == 1]
#df_fleet

#todo qingshi model still does not extend to 2072


# ![Screen%20Shot%202021-01-22%20at%203.46.42%20PM.png](attachment:Screen%20Shot%202021-01-22%20at%203.46.42%20PM.png)

# ### Technology Composition

# In[7]:


df_vehicles['technology composition'] = None
def Get_Technology_Composition(scenario, analysis_year):
    
    year = analysis_year + 2015
    
    fleet_composition = df_fleet.loc[df_fleet['scenario'] == scenario][year].to_numpy()
    #print(fleet_composition)
    
    #mask = df_fleet['scenario'].isin([scenario])
    #mask = df_fleet['scenario'] == scenario
    #fleet_composition = df_fleet.loc[mask][year]
    
    for i,technology in enumerate(technologies):
        composition = fleet_composition[i] #this index must be in the same order as technologies, otherwise lookup by value which slows down code alot
        df_vehicles.loc[df_vehicles['technology'] == technology,'technology composition'] = composition

    return 


Get_Technology_Composition(6,33)
df_vehicles.sort_values(by='technology')


# In[8]:


# %timeit Get_Technology_Composition(6,33) #1.31ms
# %timeit df_vehicles.loc[df_vehicles['technology'] == 'BEV','technology composition'] = 1
# %prun -s cumulative -l 0.4 Get_Technology_Composition(6,33)


# ### Traffic Composition

# In[9]:


def Get_Traffic_Composition(traffic_dict, rho):
    

    # unpack dict
    AADT = traffic_dict['AADT']
    mean_AADT = traffic_dict['AADT_mean']
    sd_AADT = traffic_dict['AADT_sd']
    AADTT = traffic_dict['AADTT']

    percent_pc = traffic_dict['percent_pc']
    percent_lt = traffic_dict['percent_lt']
    percent_mt = traffic_dict['percent_mt']
    percent_ht = traffic_dict['percent_ht']

    percent_trucks_impacted = traffic_dict['percent_trucks_impacted']
    percent_non_trucks_impacted = traffic_dict['percent_non_trucks_impacted']

    rho = rho  # todo: not sure what rho is

    # Step 1: Generate random variables x1 and x2 from the standard normal distribution
    x1 = np.random.standard_normal()
    x2 = np.random.standard_normal()

    # Step 2: Calculate a linear combination of x1 and x2
    x3 = rho*x1+((1-rho**2)**0.5)*x2

    # Step 3: Calculate correlated random variables
    y1 = mean_AADT+sd_AADT*x1  # non trucks growth
    y2 = mean_AADT+sd_AADT*x3  # trucks growth

    # Step 4: Update traffic variables
    AADT = AADT*(1+y1)
    AADTT = AADTT*(1+y2)

    percent_trucks = AADTT/AADT

    percent_lt = percent_trucks*(percent_lt) / (percent_lt + percent_mt + percent_ht)
    percent_mt = percent_trucks*(percent_mt) / (percent_lt + percent_mt + percent_ht)
    percent_ht = percent_trucks*(percent_ht) / (percent_lt + percent_mt + percent_ht)
    percent_pc = 1 - (percent_lt + percent_mt + percent_ht)

    percent_trucks_impacted *= (1+y2)
    percent_non_trucks_impacted *= (1+y1)  # should this also be a 1 or not?

    # Update dataframe
    # bundled 4 qingshi types into one: micro,pc,suv,lt
    df_vehicles.loc[df_vehicles['class'] ==
                    'pc', 'class composition'] = percent_pc/4
    df_vehicles.loc[df_vehicles['class'] ==
                    'FHWA LT', 'class composition'] = percent_lt
    df_vehicles.loc[df_vehicles['class'] ==
                    'FHWA MT', 'class composition'] = percent_mt
    df_vehicles.loc[df_vehicles['class'] ==
                    'FHWA HT', 'class composition'] = percent_ht

    # repack dict
    traffic_dict['AADT'] = AADT
    traffic_dict['AADTT'] = AADTT
    traffic_dict['AADT_mean'] = mean_AADT
    traffic_dict['AADT_sd'] = sd_AADT

    traffic_dict['percent_pc'] = percent_pc
    traffic_dict['percent_lt'] = percent_lt
    traffic_dict['percent_mt'] = percent_mt
    # six lines are HT, so the sum of 6 types of HT should be the total %HT of the fleet
    traffic_dict['percent_ht'] = percent_ht

    traffic_dict['percent_trucks_impacted'] = percent_trucks_impacted
    traffic_dict['percent_non_trucks_impacted'] = percent_non_trucks_impacted

    return traffic_dict


# In[10]:


#%timeit Get_Traffic_Composition(test_traffic_dict,rho=1)


# In[11]:


test_traffic_dict = {
    "AADT": 10000, "AADT_mean": 0, "AADT_sd": 0.01,'AADTT':1000,
    "percent_pc": 0.91, "percent_lt": 0.3, "percent_mt": 0.03, "percent_ht": 0.01,
    "percent_trucks_impacted": 0.001, "percent_non_trucks_impacted": 0.01}

# pprint.pprint(test_traffic_dict)
traffic_dict = Get_Traffic_Composition(test_traffic_dict, rho=1)
df_vehicles.sort_values('class composition')
#sum(df_vehicles['class composition'].mul(df_vehicles['technology composition']))


# Determine safe upper range for boundry of truck values. See how far truck values can go from year 0-60
#acc = []
#for xx in range(0,10):
#    traffic_dict = test_traffic_dict.copy()
    
#    for x in range(0,60):
#        traffic_dict = Get_Traffic_Composition(traffic_dict, rho=1)
        
#    acc.append(np.max(traffic_dict['AADTT']))
    
# print(np.max(acc))


# ## Load Fuel GWP

# In[12]:


def update_energycycle_gwp(ec_gwp_dict):
    
      
    for tech in technologies:
        
        df_vehicles.loc[df_vehicles['technology'] == tech,'fuel gwp'] = ec_gwp_dict[tech]["fuel gwp"]
        df_vehicles.loc[df_vehicles['technology'] == tech,'elec gwp'] = ec_gwp_dict[tech]["elec gwp"]


# In[13]:


# units kgco2/MJ
df_vehicles['fuel gwp'] = 0
df_vehicles['elec gwp'] = 0
ec_gwp_dict = {}
# temporary until characterization factors column
# https://afdc.energy.gov/files/u/publication/fuel_comparison_chart.pdf
# convert from kwh to gallon gas equivalent
gal_gas_to_MJ = 0.030 * 3.6
gal_diesel_to_MJ = 0.027 * 3.6
# GWP per gallon
GWP_GAL_GAS = 3.942429712  # GWP in kg CO2e per gallon of gasoline burned
GWP_GAL_DIESEL = 3.751262042  # GWP in kg CO2e per gallon of diesel burned
# Final GWP
GWP_hydro = 0.015989476201610076  # kgco2/kwh
GWP_hydro *= 3.6  # kgco2/MJ
GWP_gas = GWP_GAL_GAS*gal_gas_to_MJ
GWP_diesel = GWP_GAL_DIESEL*gal_gas_to_MJ

# TODO: GET HYDROGEN EMISSION FACTOR
GWP_hydrogen = GWP_gas

# Load and sort emission factors
file = DEPENDANCY_PATH + "/brightway_export.xlsx"
df_emissionfactors = pd.read_excel(file, engine='openpyxl')
df_emissionfactors.drop(['#'], axis=1, inplace=True)
# kgco2/kwh to kgco2/MJ
df_emissionfactors['LCA Score'] = df_emissionfactors['LCA Score']/3.6
df_emissionfactors = df_emissionfactors.sort_values(
    by='LCA Score', ascending=False)


def get_emission_factors(t, region, energy_type, ec_gwp_dict):

    df1 = df_emissionfactors.loc[(df_emissionfactors['Location'].values == region) & (
        df_emissionfactors['Type'].values == energy_type)]
    emit_factors = df1['LCA Score'].values

    # select factor from list. As t increases: higher chance of chosing lowest emitting variant
    best_score = math.ceil((len(emit_factors)-1)/ANALYSIS_PERIOD*(t) + 1)
    i = np.random.randint(0, best_score)
    GWP_elec = emit_factors[i]

    # pack the emission dictionary for this year
    # todo: gas and diesel should probably change too
    for tech in technologies:

        if tech == 'ICEV-d':
            ec_gwp_dict[tech] = {"fuel gwp": GWP_diesel, "elec gwp": GWP_elec}
        elif tech == 'HFCEV':
            ec_gwp_dict[tech] = {
                "fuel gwp": GWP_hydrogen, "elec gwp": GWP_elec}
        else:
            ec_gwp_dict[tech] = {"fuel gwp": GWP_gas, "elec gwp": GWP_elec}

    return ec_gwp_dict


ec_gwp_dict = get_emission_factors(1, "WECC", "Solar", ec_gwp_dict)
update_energycycle_gwp(ec_gwp_dict)
df_vehicles
df_emissionfactors


# ## Pavement Dimensions

# In[14]:


# SA used in Albedo, Lighting, and Excavation functions, these values are just for testing

# Case study is one lane mile for now
Lane_Width = 12  # ft
Highway_Length = 1  # mile
Number_Lanes = 6
Number_Shoulders = 2
Shoulder_Width = 8  # ft

# Calculate pavement surface area in square meters
SA = (Number_Lanes*Lane_Width + Number_Shoulders*Shoulder_Width) *     FT_TO_METER * Highway_Length*MILE_TO_METER


# # GWP Functions

# ## Embodied Impacts

# ### Embodied Impacts Setup

# In[15]:


# All the relevant data is assumed to be already assembled in an excel file.
# This makes it easier to import the large amount of data needed

# Unit GWP and quantity of each material/process is assumed to be distributed lognormally.

# The excel file should have a sheet called GWP.
# This sheet should have 34 rows representing 34 materials or processes, and at least 16 columns, containing:
# 1) The name of the material or process
# 2) The median environmental impact, in GWP, per unit of material or process
# 3) The uncertainty of the unit impact for each process
# 4) The median quantity of material or process ordered for each action
# 5) The uncertainty of the quanitity ordered for each material or process, for each action
def Embodied_GWP_Setup(Embodied_Input_File, GWP_Sheet_Name = 'GWP', Display_Results = 0):
    
    # Access data from excel file
    df = pd.ExcelFile(Embodied_Input_File)
    df1 = df.parse(GWP_Sheet_Name)
    
    
    # specify some basic parameters
    sim = 100000                        # number of simulations
    quant = 100                         # number of quantiles for QQ plot
    bins = 100                          # number of histogram bins
    num_mtrls = len(df1)                # number of materials/processes
    num_actions = 6                     # number of possible actions the decision maker can take, not including do nothing
    
    # initialize arrays
    sample_impacts = np.zeros((sim, num_actions)) # stores the total impact for each action, for each iteration
    
    # create arrays from df, this makes the program more efficient
    median_impact = df1.values[:, 1]        # stores the median impact for one unit of each material
    unc_impact = df1.values[:, 2]           # stores the uncertainty of the impact for one unit of each material
    median_quantity = df1.values[:, 3:9]    # stores the quantity of materials used for each action
    unc_quantity = df1.values[:, 10:16]     # stores the uncertainty of the quantity of materials used for each action
            
    
    # Loop over every material or process
    for j in range(num_mtrls):
            
        # generate a large number random impacts from lognormal distribution
        r = np.random.lognormal( np.log(median_impact[j]), unc_impact[j], sim)
            
        # weight by quantity of material produced, add to sample_impacts
        for k in range(num_actions):
            if median_quantity[j][k] != 0:
                sample_impacts[:, k] += r * np.random.lognormal( np.log(median_quantity[j][k]), unc_quantity[j][k], sim)
    
    
    
    # array to store the skewnormal distribution parameters of embodied GWP for each action
    skn_params = np.zeros((num_actions, 3))
    
    ## solve for the lognormal distribution parameters of embodied GWP and plot a histogram for each action
    for k in range(num_actions):
        skn_params[k] = ss.skewnorm.fit(np.log(sample_impacts[:, k]))
        
        
        #if(Display_Results == 1):
            
            # Print results
            #print()
            #print("Action: ", (k+1))
            #print("Median embodied GWP: %.0f kg CO2e" % np.exp(skn_params[k][1]))
            
            ## Plotting a histogram with an overlaid bell curve

            # Histogram
            #histogram = plt.hist(np.log(sample_impacts[:, k]), bins = bins, density = True, color = 'navy', label = 'Histogram entries')

            # Generate overlying curve
            #xdata = histogram[1]
            #curve = ss.skewnorm.pdf(xdata, skn_params[k][0], skn_params[k][1], skn_params[k][2])
            #plt.plot(xdata, curve, color='red', linewidth=2.5, label = 'Fitted distribution')

                    
            # Make the plot nicer
            #plt.xlabel(r'GWP (kg CO2e)')
            #plt.ylabel(r'Probability density')
            #plt.title((r'Log of embodied GWP for Action ' + str(k+1) ))
            #plt.legend(loc='best')
            
            #plt.clf()
            
            
            ## Ploting QQ plot
            
            # Find quantiles of random data
            #x = np.linspace(0, 1, quant)
            #qdata = np.quantile(np.log(sample_impacts[:,k]), x)
            
            # Find quantiles for fitted distribution
            #qdist = ss.skewnorm.ppf(x, skn_params[k][0], skn_params[k][1], skn_params[k][2])
            
            # Plot the observed quantiles vs theoretical quantiles
            #plt.scatter(qdist, qdata)
            
            # Plot a straight line through the origin for comparison
            #x = (qdata[0], qdata[quant-1])
            #plt.plot(x,x)
            
            # Make the plot nicer
            #plt.title('Log of QQ plot')
            #plt.xlabel('Fitted  quantiles')
            #plt.ylabel('Observed quantiles')
            
            #plt.clf()
            
    
    return skn_params


# ### Embodied GWP Setup test

# In[16]:


# Provide file name for excel sheet. File must be in same directory as this python code
file = DEPENDANCY_PATH + '/Input_Processes_CaseU1_UrbanInterstate.xls'

# Provide sheet name for GWP data, within the above excel file
# This sheet must have the right data in the right cells; see comment at top of Embodied_GWP_Setup definition
GWP_Sheet_Name = 'GWP'

# If equal to 1, the Embodied_GWP_Setup file will show the histogram and fitted curve of GWP for each action
Display_Results = 0


# Start timing
start = mytime.time()

# Call the function
Embodied_GWP_Parameters = Embodied_GWP_Setup(
    file, GWP_Sheet_Name, Display_Results)

# Stop timing
duration = mytime.time() - start

# Display results
# Embodied_GWP_Parameters contains the mean and variance of the Embodied GWP (assumed normal) for each action
print(Embodied_GWP_Parameters)

print('Time elapsed, in seconds:', duration)


# ### Get Embodied GWP function

# In[17]:


def Get_Embodied_GWP(Emb_Params, Action):

    return np.exp(ss.skewnorm.rvs(Emb_Params[Action-1, 0], Emb_Params[Action-1, 1], Emb_Params[Action-1, 2]))


# Testing
Action = 3
impact = Get_Embodied_GWP(Embodied_GWP_Parameters, Action)
print(impact)


# ### Excavation GWP function

# In[18]:


# only use when reconstructing the road
@njit
def Get_Excavation_GWP(SN, Lane_Width, Highway_Length, Number_Lanes, Volume_Mass_Density, EOL):

    # Conversion factor for square meters to lane-miles
    SqrMeters_to_LaneMiles = 10.7639/(5280*Lane_Width)
    Area = Number_Lanes * Highway_Length / SqrMeters_to_LaneMiles

    # Calculate thickness of pavement using SN
    # Note that this assumes we excavate (and replace, in the embodied impacts) the aggregate base
    Asphalt_thickness = SN / 0.45 * INCH_TO_METER
    Aggregate_thickness = 6 * INCH_TO_METER

    # Calculate volume that must be excavated (in m^3)
    Volume = (Asphalt_thickness + Aggregate_thickness) * Area

    # Calucalate energy for excavation
    Energy_for_excavation = 3.32  # MJ/m^3
    Energy_needed = Energy_for_excavation * Volume

    # GWP impacts and uncertainties from Input_Processes spreadsheet
    GWP_excavation = np.random.lognormal(np.log(
        0.090752558), 0.121655251) * np.random.lognormal(np.log(Energy_needed), 0.200374649)

    if EOL == False:

        return GWP_excavation

    # Randomly simulate the recycling rate
    recycling_rate = np.random.uniform(0.8, 1)

    # Calculate the mass in kg of landfilled material - assume 1% of recycled material still goes to the landfill

    landfilled_mass = (recycling_rate * 0.01 +
                       (1 - recycling_rate)) * Volume * Volume_Mass_Density

    # Calculate the EOL GWP impact (assume 94% of RAP is used for aggregate, and 6% is used for bitumen)

    GWP_landfilling = landfilled_mass *         np.random.lognormal(np.log(0.0119), 0.121655251)

    return GWP_excavation + GWP_landfilling

# Test the function


SN = 4.9
Volume_Mass_Density = 2243  # kg/cubic-meter

GWP = Get_Excavation_GWP(SN, Lane_Width, Highway_Length,
                         Number_Lanes, Volume_Mass_Density, True)

print(GWP)  # 9.87microseconds per run to 459nano seconds with njit


# In[19]:


#%timeit GWP = Get_Excavation_GWP(SN,Lane_Width, Highway_Length, Number_Lanes, Volume_Mass_Density, True)


# ## Albedo - Radiative Forcing

# In[20]:


# Create a dictionary (not a dataframe) for reduction in GWP
# Units: kgCO2/sqrmeter/50-years/0.01 increase in pavement albedo
Alb = {}

Alb['Miami'] = [1.23, 1.56]
Alb['Houston'] = [1.13, 1.54]
Alb['Phoenix'] = [1.47, 1.60]
Alb['Atlanta'] = [0.98, 1.38]
Alb['Los Angeles'] = [0.94, 1.52]
Alb['San Francisco'] = [1.16, 1.49]
Alb['Baltimore'] = [0.86, 1.26]
Alb['Nashville'] = [1.02, 1.36]
Alb['St. Louis'] = [0.97, 1.32]
Alb['Seattle'] = [1.04, 1.38]  # this is the average value
Alb['Boston'] = [0.96, 1.34]
Alb['Denver'] = [1.22, 1.44]
Alb['Minneapolis'] = [0.84, 1.17]
Alb['Duluth'] = [0.81, 1.13]


# Input units: pavement area (SA) in square meters
def Get_Albedo_GWP(Geographic_Location, Current_Albedo, SA):

    # Get GWP as a uniform random variable between lower and upper bounds for the given geographic location
    # Convert from kgCO2/sqrmeter/0.01-increase-in-albedo/50-years to kgCO2/year
    # Negative value because albedo should reduce GWP
    # Return GWP in kgCO2e per year

    return -np.random.uniform(Alb[Geographic_Location][0], Alb[Geographic_Location][1]) / 50 * (Current_Albedo/0.01) * SA


# Test the function
Geographic_Location = 'Seattle'
Current_Albedo = 0.3
print('Sampled Albedo GWP:', Get_Albedo_GWP(
    Geographic_Location, Current_Albedo, SA), 'kg CO2e')


# Average Albedo GWP function to initialize Q-values
def Get_Albedo_GWP_average(Geographic_Location, Current_Albedo, SA):

    return - ((Alb[Geographic_Location][0] + Alb[Geographic_Location][1]) / 2) / 50 * (Current_Albedo/0.01) * SA


# Test the function
Albedo_GWP = Get_Albedo_GWP_average(Geographic_Location, Current_Albedo, SA)
print('Average Albedo GWP', Albedo_GWP, 'kg CO2e')


# In[21]:


# %timeit Albedo_GWP = Get_Albedo_GWP_average(Geographic_Location, Current_Albedo, SA)


# ## Lighting

# In[22]:


# Source: Santero and Horvath 2009 supplementary information

Pavement_Lighting_Req = 9 # lumens per square meter (lux)
Baseline_Lighting_Req = 6 # lumens per square meter (lux)
Usage_time = 4380 # Hours of darkness in a year assuming 50% of all hours are dark on average
Emission_factor_electricity = 0.785 # Mg CO2e/kWh
Efficacy = 30000 # lumen/kW
Lighting_Uncertainty = 0.174068952 # sd of underlying normal distribution, obtained from pedigree matrix
@njit
def Get_Lighting_GWP(SA):
   
    Median_Lighting_GWP = (Pavement_Lighting_Req - Baseline_Lighting_Req) * SA * Usage_time * Emission_factor_electricity / Efficacy  
    
    return np.random.lognormal( np.log(Median_Lighting_GWP), Lighting_Uncertainty )

# Test the function
print('Sampled Lighting GWP (kg CO2e per year):', Get_Lighting_GWP(SA) ) #2.7us to 215ns with njit

# Average function to initialize Q-values
@njit
def Get_Lighting_GWP_average(SA):
    return (Pavement_Lighting_Req - Baseline_Lighting_Req) * SA * Usage_time * Emission_factor_electricity / Efficacy

# Test
print('Median Lighting GWP (kg CO2e per year):', Get_Lighting_GWP_average(SA) ) #155ns to 105ns


# In[23]:


#%timeit Get_Lighting_GWP(SA)
#%timeit Get_Lighting_GWP_average(SA)


# ## Get Fleet Composition

# ## Roughness - HB Update

# In[24]:


# Source: Ziyadi et al. 2018

# Input RSI model coefficients
k_a = [6.7e-1, 7.68e-1, 9.18e-1, 1.4]
k_c = [2.81e-4, 1.25e-4, 1.33e-4, 1.36e-4]
d_c = [2.1860e-1, 3.0769e-1, 9.7418e-1, 2.3900]
d_a = [2.1757e3, 7.0108e3, 9.2993e3, 1.9225e4]
b = [-1.6931e1, -7.3026e1, -1.3959e2, -2.6432e2]
p = [3.3753e4, 1.1788e5, 1.0938e5, 8.2782e4]

RSI_0 = [0, 0, 0, 0]
delta_minus_1 = [0, 0, 0, 0]
classes = ['pc', 'FHWA LT', 'FHWA MT', 'FHWA HT']
# @profile


def Get_Roughness_GWP(IRI, Speed_Limit, traffic_dict, Highway_Length):

    AADT = traffic_dict['AADT']

    # Convert IRI from m/km to in/mile
    IRI = IRI * MPERKM_TO_INPERMI

    for i in range(4):
        RSI_0[i] = p[i]/Speed_Limit + d_a[i] + b[i]*Speed_Limit +             d_c[i]*Speed_Limit**2  # Calculate the RSI when IRI=0 - eqn 13

    # Calculate excess fuel consumption factors using given roughness
    for i in range(4):
        delta_minus_1[i] = (k_a[i]*IRI + k_c[i]*IRI*Speed_Limit**2) / RSI_0[i]

    roughness_gwp = 0
    # later do:
    # (fuel * fuel gwp) + (elec* elec gwp), and combine with 'class' column, then filter micro,pc,suv,lt if possible

    for i, veh_class in enumerate(classes):

        # Tried 4 methods, this one is fastest
        veh_classes = df_vehicles[df_vehicles['class'].values == veh_class]

        # print(veh_classes)

        fuel_use = veh_classes['fuel consumption'].to_numpy()
        elec_use = veh_classes['elec consumption'].to_numpy()
        technology_comp = veh_classes['technology composition'].to_numpy()
        class_comp = veh_classes['class composition'].to_numpy()
        # print(technology_comp,class_comp)
        fuel_gwp = veh_classes['fuel gwp'].to_numpy()
        elec_gwp = veh_classes['elec gwp'].to_numpy()

        # calculate GWP for the vehicle class
        # todo: change to pandas mul for faster results and also update ec_array with fuel efficiencies and re-use where possible in version 8
        ec_array = fuel_use*fuel_gwp + elec_use*elec_gwp
        gwp_array = delta_minus_1[i]*technology_comp *             class_comp*ec_array*Highway_Length*365*AADT
        # print(gwp_array.sum())
        roughness_gwp += gwp_array.sum()

    # print(RSI_0,'\n',delta_minus_1)

    return roughness_gwp

# Test the function


# Select test values
Speed_Limit = 25
IRI = 1  # m/km
print('Roughness GWP (kgCO2/year)', Get_Roughness_GWP(IRI,
                                                      Speed_Limit, traffic_dict, Highway_Length,))

# Santero_Horvath_2009 probable range for roughness GWP (in kgCO2/year for 1 mile - functional unit gives no info on traffic volume): 0 - 16,100


# In[25]:


# %timeit Get_Roughness_GWP(IRI, Speed_Limit, traffic_dict, Highway_Length)


# ## Deflection

# ### Deflection data and constants

# In[26]:


Pij_DataFrame = pd.DataFrame([
    [-1.918, 4.487, -19.54, 59.58, -92.51, 56.23],
    [-0.4123, -1.802, 4.014, -4.628, 1.375, 0],
    [-0.06942, 0.2153, -0.8618, 0.7344, 0, 0],
    [-0.009575, 0.0203, 0.04669, 0, 0, 0], ],
    index=[0, 1, 2, 3],
    columns=[0, 1, 2, 3, 4, 5])


# Temperature Data from NOAA
# Still need to find reliable values for Minneapolis and Duluth
Avg_Temp_DataFrame = pd.DataFrame([
    [25.1], [21.5], [23.6], [17.4], [16.9], [13.6], [13.3], [15.5], [13.8], [11.9], [10.5], [9.9], ['Not Available'], ['Not Available'], ],
    index=['Miami', 'Houston', 'Phoenix', 'Atlanta', 'Los Angeles', 'San Francisco', 'Baltimore',
           'Nashville', 'St. Louis', 'Seattle', 'Boston', 'Denver', 'Minneapolis', 'Duluth'],
    columns=['Avg_Temp'])


# ### Deflection Excess Energy function

# In[27]:


# Function input units: stiffness in MPA, speed limit in mph, thickness in m, axle load in N, lane width in ft, highway length in miles
# Not sure what "volume mass density" is - assuming they are referring to a measuremnt of density, the input units should be kg/cubic meter
# Easy to change any of these units if needed

def Get_Deflection_Excess_Energy_DataFrame(Pavement_Stiffness, Speed_Limit, Volume_Mass_Density, Subgrade_Stiffness, Geographic_Location, sn_states):

    # Create a for loop to evaluate values for SN=3.43,4.17,4.9,5.64,6.37
    rows = []
    for SN in sn_states:

        # Step 1: Get values for structural properties to be used in GWP calculation - convert imperial input units to SI base units
        Top_Layer_Thickness = SN/0.45*INCH_TO_METER

        m = Volume_Mass_Density*Top_Layer_Thickness

        k = ALPHA*Subgrade_Stiffness*MEGA_TO_BASE

        l_s = pow(Pavement_Stiffness*MEGA_TO_BASE *
                  pow(Top_Layer_Thickness, 3)/(12*k), 0.25)

        c_cr = l_s*pow(k/m, 0.5)

        tau = 0.0083*pow(10, (-34*(Avg_Temp_DataFrame._get_value(Geographic_Location, 'Avg_Temp')-10))/(
            203+Avg_Temp_DataFrame._get_value(Geographic_Location, 'Avg_Temp')-10))  # eqn 15-17

        pi_1 = Speed_Limit*MPH_TO_MPS/c_cr

        pi_2 = tau*c_cr/l_s

        # Step 2: Evaluate equation 14 to calculate pi using a for loop

        log_pi = 0  # initialize for double sum table 1 p.99 to calculate pi
        for i in range(6):
            for j in range(4):
                # 'na' in Table 1 is zero in my dataframe
                if Pij_DataFrame._get_value(j, i) == 0:
                    pass
                else:
                    # eq 14 p.99
                    log_pi += Pij_DataFrame._get_value(j, i) * pow(
                        pi_1, i) * pow(math.log10(pi_2), j)
        pi = pow(10, log_pi)

        # deflection_parameters=(l_s,k,c_cr,pi)
        # print(deflection_parameters)

        # Step 3: Calculate excess energy per pavement length - units J/m (everything should be in base SI units at this point to avoid errors)

        # calculate column wise, add new column for this SN in the loop
        axle1 = df_vehicles['axle-1']
        axle2 = df_vehicles['axle-2']
        axle3 = df_vehicles['axle-3']
        axle4 = df_vehicles['axle-4']

        # print(axle1)
        # print(axle2)
        # ST_p1**2+ST_p2**2)/pow(l_s,2)/k*c_cr/(Speed_Limit*MPH_TO_MPS)*pi
        df_vehicles[SN] = (axle1**2 + axle2**2 + axle3**2+axle4**2) /             pow(l_s, 2)/k*c_cr/(Speed_Limit*MPH_TO_MPS)*pi
    # Combine all excess-energy columns into dict and drop extra cols
    #df_vehicles['excess-energy'] = df_vehicles[SN_States].to_dict(orient='records')
    # df_vehicles.drop(columns=SN_States,inplace=True)

    # Excess_Energy_DataFrame=pd.DataFrame(rows,index=SN_States,columns=['Small_Truck','Medium_Truck','Large_Truck','Passenger_Car'])

    return


df_vehicles


# ### Deflection GWP function

# In[28]:


# "To maintain a constant speed, the vehicle's engine must supply additional energy to compensate for the 
# energy that is dissipated in the pavement structure. This excess energy depends on structural and material
# properties of the pavement, temperature, and vehicle speed" Louhghalam et al. (2014) 

#So does it not depend on fuel efficiency of vehicle? not sure yet how to implement the FE improvements here

def Get_Deflection_GWP(traffic_dict, Highway_Length,excess_energy_factor,SN):
    
    Deflection_GWP = 0
    
    #Unpack Traffic Volumes
    AADT = traffic_dict['AADT']

    #excess energy column
    excess_energy = df_vehicles[SN].to_numpy()*df_vehicles['class composition'].to_numpy()*df_vehicles['technology composition'].to_numpy()
    
    #gwp per MJ column
    gwp_elec = df_vehicles['elec gwp'].to_numpy()
    gwp_gas = df_vehicles['fuel gwp'].to_numpy()
    
    #Excess energy x GWP/MJ
    ee_electricity = excess_energy*df_vehicles['elec consumption'].to_numpy() * gwp_elec
    ee_gasoline = excess_energy*df_vehicles['fuel consumption'].to_numpy() * gwp_gas
    
    #GWP * AADT * Functional Unit
    deflection_gwp = (ee_electricity + ee_gasoline)*AADT*365*(Highway_Length*MILE_TO_METER)*(1/MEGA_TO_BASE)


    #Sum the deflections
    Deflection_GWP = deflection_gwp*excess_energy_factor #multiply then sum? I dont know what this factor is
    Deflection_GWP = Deflection_GWP.sum()
    #Return GWP in kgCO2/year
    
    return Deflection_GWP


#Get_Deflection_GWP(traffic_dict,fleet_composition, Highway_Length,excess_energy_factor,SN)

#EE = df_vehicles[['vehicle','class','technology',SN]]


# ### Deflection Test

# In[29]:


# Select test values
sn_states = [3.65, 4.44, 5.22, 6.00, 6.79]
Highway_Length = 1
Pavement_Stiffness = 16000
AADT = 10000
Percent_Small_Trucks = 0.035
Percent_Medium_Trucks = 0.19
Percent_Large_Trucks = 0.02
Speed_Limit = 70  # mph
Volume_Mass_Density = 2603  # kg/cubic-meter
Gs = 267
s_Stiffness = 2*(1+0.4)*Gs
Geographic_Location = 'Denver'
excess_energy_factor = 0.98

# Get deflection excess energy

Excess_Energy_DataFrame = Get_Deflection_Excess_Energy_DataFrame(
    Pavement_Stiffness, Speed_Limit, Volume_Mass_Density, s_Stiffness, Geographic_Location, sn_states)

pprint.pprint(Excess_Energy_DataFrame)

# Enter test values into function
for SN in sn_states:

    Deflection_GWP = Get_Deflection_GWP(
        traffic_dict, Highway_Length, excess_energy_factor, SN)

    print('Deflection GWP (kgCO2/year):', Deflection_GWP)


# Santero_Horvath_2009 probable range for defelction GWP (in kgCO2/year for 1 mile and 11.81 ft lane width - functional unit gives no info on traffic volume): 0 - 3220

# df_vehicles


# In[30]:


# %timeit Deflection_GWP = Get_Deflection_GWP(traffic_dict, Highway_Length,excess_energy_factor,SN)


# ## WZ Congestion

# ### Get Delta (setup function)

# In[31]:


# Source: Table 2. Zhang et al. 2011.
# Assume the data is distributed normally, mean is the main value in the table
# Standard deviation is standard error divided by sqrt of number of samples

LDV_Freeflow_mean = 97
LDV_Freeflow_sd = 7/51**0.5
LDV_WZ_mean = 108
LDV_WZ_sd = 6/11**0.5

HDV_Freeflow_mean = 519
HDV_Freeflow_sd = 49/41**0.5
HDV_WZ_mean = 852
HDV_WZ_sd = 80/11**0.5


def WZ_Setup_Get_Delta():

    sim = 10000  # number of simulations

    # run simulation
    LDV_Delta_Samples = np.random.normal(
        LDV_WZ_mean, LDV_WZ_sd, sim) / np.random.normal(LDV_Freeflow_mean, LDV_Freeflow_sd, sim)
    HDV_Delta_Samples = np.random.normal(
        HDV_WZ_mean, HDV_WZ_sd, sim) / np.random.normal(HDV_Freeflow_mean, HDV_Freeflow_sd, sim)

    # fit normal distribution
    LDV_Delta = ss.norm.fit(LDV_Delta_Samples)
    HDV_Delta = ss.norm.fit(HDV_Delta_Samples)

    return LDV_Delta, HDV_Delta


# Test the function
LDV_Delta, HDV_Delta = WZ_Setup_Get_Delta()
print(LDV_Delta)
print(HDV_Delta)


# ### Congestion GWP functions - HB

# In[32]:


def Get_Congestion_GWP(LDV_Delta, HDV_Delta, traffic_dict, Highway_Length, Construction_Duration, Maintenance_Duration, action):

    AADT = traffic_dict['AADT']
    Percent_PC_Impacted = traffic_dict['percent_non_trucks_impacted']
    Percent_Trucks_Impacted = traffic_dict['percent_trucks_impacted']

    if (action < 6):
        WZ_duration = Construction_Duration
    elif(action < 7):
        WZ_duration = Maintenance_Duration
    else:
        return 0

    fuel_use = df_vehicles['fuel consumption']
    elec_use = df_vehicles['elec consumption']
    # powertrain composiiton ice-g,ice-d,bev,hev,phev,hfc
    technology_comp = df_vehicles['technology composition']
    # vehicle classes micro,pc,suv,lt
    class_comp = df_vehicles['class composition']
    class_array = df_vehicles['class']

    gwp_fuel = df_vehicles['fuel gwp']
    gwp_elec = df_vehicles['elec gwp']

    ec_array = fuel_use * GWP_GAL_GAS*gwp_fuel + elec_use * gwp_elec
    ec_array *= class_comp * technology_comp * AADT * WZ_duration * Highway_Length
    ec_array = pd.concat([class_array, ec_array], axis=1)

    # print(ec_array)

    ldv_array = ec_array.loc[(ec_array['class'] == 'pc') | (
        ec_array['class'] == 'FHWA LT'), 0]
    hdv_array = ec_array.loc[(ec_array['class'] == 'FHWA MT') | (
        ec_array['class'] == 'FHWA HT'), 0]

    # print(hdv_array)

    ldv_impacts = (LDV_Delta[0] - 1) * Percent_PC_Impacted
    hdv_impacts = (HDV_Delta[0] - 1) * Percent_Trucks_Impacted

    ldv_gwp = ldv_array*ldv_impacts
    hdv_gwp = hdv_array*hdv_impacts

    GWP = ldv_gwp.sum() + hdv_gwp.sum()

    return GWP


# Average WZ congestion function to initialize Q-values
def Get_Congestion_GWP_average(LDV_Delta, HDV_Delta, traffic_dict, Highway_Length, Construction_Duration, Maintenance_Duration, action):

    AADT = traffic_dict['AADT']
    Percent_PC_Impacted = traffic_dict['percent_non_trucks_impacted']
    Percent_Trucks_Impacted = traffic_dict['percent_trucks_impacted']

    if (action < 6):
        # incorporate uncertainty into WZ duration, standard deviation obtained from pedigree matrix
        WZ_duration = np.random.lognormal(
            np.log(Construction_Duration), 0.2053)
    elif(action < 7):
        # incorporate uncertainty into WZ duration, standard deviation obtained from pedigree matrix
        WZ_duration = np.random.lognormal(np.log(Maintenance_Duration), 0.2053)
    else:
        return 0

    # vehicle_names = df_vehicles.loc[df_vehicles['class'] == veh_class,['class','fuel consumption','elec consumption']] #for debugging only
    # print(vehicle_names)

    fuel_use = df_vehicles['fuel consumption']
    elec_use = df_vehicles['elec consumption']
    # powertrain composiiton ice-g,ice-d,bev,hev,phev,hfc
    technology_comp = df_vehicles['technology composition']
    # vehicle classes micro,pc,suv,lt
    class_comp = df_vehicles['class composition']
    class_array = df_vehicles['class']

    gwp_fuel = df_vehicles['fuel gwp']
    gwp_elec = df_vehicles['elec gwp']

    ec_array = fuel_use * GWP_GAL_GAS*gwp_fuel + elec_use * gwp_elec
    ec_array *= class_comp * technology_comp * AADT * WZ_duration * Highway_Length
    ec_array = pd.concat([class_array, ec_array], axis=1)

    # print(ec_array)

    ldv_array = ec_array.loc[(ec_array['class'] == 'pc') | (
        ec_array['class'] == 'FHWA LT'), 0]
    hdv_array = ec_array.loc[(ec_array['class'] == 'FHWA MT') | (
        ec_array['class'] == 'FHWA HT'), 0]

    # print(hdv_array)

    ldv_impacts = (LDV_Delta[0] - 1) * Percent_PC_Impacted
    hdv_impacts = (HDV_Delta[0] - 1) * Percent_Trucks_Impacted

    ldv_gwp = ldv_array*ldv_impacts
    hdv_gwp = hdv_array*hdv_impacts

    # note: could multily and add in one line for a dataframe column fine, but misbehaves on series.
    # note: had to multiply one line, then add on the next line for the series to work

    GWP = ldv_gwp.sum() + hdv_gwp.sum()

    #GWP_LDV = (LDV_Delta[0] - 1) * AADT * Percent_PC_Impacted
    #GWP_HDV = (HDV_Delta[0] - 1) * AADT * Percent_Trucks_Impacted * WZ_duration * hdv_fuel_use * Highway_Length * GWP_GAL_DIESEL*0.027

    return GWP


# Test the functions

traffic_dict['AADT'] = 10000  # Average Annual Daily Traffic
Highway_Length = 1  # in miles
action = 1  # for testing purposes

Construction_Duration = 120  # Work days
Maintenance_Duration = 40  # Work days
gal_per_mile_LDV = 1/35    # 1/MPG (highway) for an average car, gasoline
gal_per_mile_HDV = 0.139832384  # 1/MPG for an average truck, diesel fuel


Congestion_GWP = Get_Congestion_GWP(LDV_Delta, HDV_Delta, traffic_dict,
                                    Highway_Length, Construction_Duration, Maintenance_Duration, action)
print("Sampled congestion GWP in kg CO2e:", Congestion_GWP)

Congestion_GWP_average = Get_Congestion_GWP_average(
    LDV_Delta, HDV_Delta, traffic_dict, Highway_Length, Construction_Duration, Maintenance_Duration, action)
print("Average congestion GWP in kg CO2e:", Congestion_GWP_average)

# Santero_Horvath_2009 probable range for congestion GWP (in kgCO2/year for 1 mile - functional unit gives no info on traffic volume): 0 - 32,200


# # Cost Functions

# ### Price Index

# In[33]:


# Source: Yehia Wong Swei 2019, Table 2
@njit
def Get_Pt(prev_err, t):  # 2.69us to  248ns with njit
    current_err = 0.06 * prev_err + np.random.normal(0, 0.065)
    prev_err = current_err

    return np.exp(4.6 + 0.014*t + current_err), prev_err


# Test the function
P = np.zeros(51)
prev_err = 0

for t in range(51):
    P[t], prev_err = Get_Pt(prev_err, t)

#plt.plot(range(51), P)
#plt.title('Price Index over Analysis Period')



#test upper bounds of pt

#acc = []
#for x in range(0,100000):
    
#    for t in range(51):
#        P[t], prev_err = Get_Pt(prev_err, t)
    
#    acc.append(P[t])
    
# print(np.max(acc))


# In[34]:


# %timeit Get_Pt(prev_err, t)


# ### Maintenance/Construction Cost

# In[35]:


# Fitted model for roadway construction cost
asphalt_log_q, asphalt_const, asphalt_sd = -0.1490314, 5.475205, 0.3663
aggregate_log_q, aggregate_const, aggregate_sd = -0.1057854, 3.174054, 0.53427
milling_log_q, milling_const, milling_sd = -0.3230392, 4.217846, 0.48426
overlay_log_q, overlay_const, overlay_sd = -0.0813632, 4.750021, 0.17386

SN_States = [2.6, 3.15, 3.71, 4.27, 4.82]
Volume_Mass_Density = 2243  # kg/cubic-meter

# Pavement Cost Setup


def Pavement_Cost_Setup(SN_States, SA):

    # Asphalt costs
    Top_Layer_Thickness = np.array(SN_States) / 0.45 * INCH_TO_METER
    Pavement_Tons = Volume_Mass_Density * Top_Layer_Thickness * SA * kg_to_tons
    asph_cost_per_ton = np.exp(
        asphalt_const + asphalt_log_q*np.log(Pavement_Tons))
    Log_asph_cost_mean = np.log(asph_cost_per_ton * Pavement_Tons)

    # Aggregate costs
    SA_in_sq_yrds = SA * SQMETRE_TO_SQYARD
    agg_cost_per_sq_yrd = np.exp(
        aggregate_const + aggregate_log_q*np.log(SA_in_sq_yrds))
    Log_agg_cost_mean = np.log(agg_cost_per_sq_yrd * SA_in_sq_yrds)

    # Milling costs
    milling_cost_per_sq_yrd = np.exp(
        milling_const + milling_log_q*np.log(SA_in_sq_yrds))
    Log_milling_cost_mean = np.log(milling_cost_per_sq_yrd * SA_in_sq_yrds)

    # Overlay costs
    Overlay_Thickness = 2 * INCH_TO_METER
    Overlay_Tons = Volume_Mass_Density * Overlay_Thickness * SA * kg_to_tons
    overlay_cost_per_ton = np.exp(
        overlay_const + overlay_log_q*np.log(Overlay_Tons))
    Log_overlay_cost_mean = np.log(overlay_cost_per_ton * Overlay_Tons)

    return Log_asph_cost_mean, Log_agg_cost_mean, Log_milling_cost_mean, Log_overlay_cost_mean


@njit # 4.14us to 285ns
def Get_Cost(Action, Pt):  

    if Action < 6:

        asphalt_cost = np.random.lognormal(
            Log_asph_cost_mean[Action - 1], asphalt_sd)
        aggregate_cost = np.random.lognormal(Log_agg_cost_mean, aggregate_sd)
        return asphalt_cost + aggregate_cost

    elif Action < 7:

        milling_cost = np.random.lognormal(Log_milling_cost_mean, milling_sd)
        overlay_cost = np.random.lognormal(Log_overlay_cost_mean, overlay_sd)
        return milling_cost + overlay_cost

    else:
        return 0


Action = 3
Pt = 100
Cost = np.zeros(1000)

# Test the function
Log_asph_cost_mean, Log_agg_cost_mean, Log_milling_cost_mean, Log_overlay_cost_mean = Pavement_Cost_Setup(
    SN_States, SA)

for i in range(1000):
    Cost[i] = Get_Cost(Action, Pt)

#plt.hist(Cost, bins=50)
#plt.title('Histogram of costs for Action ' + str(Action))



# In[36]:


# %timeit Get_Cost(Action, Pt)


# ### Salvage Value

# In[37]:


@njit  # 105us to 1.77us
def Get_Salvage_Value(IRI, IRI_constructed, IRI_max, Age, T_last, C_last, SN, AADT, Percent_Small_Trucks, Percent_Medium_Trucks, Percent_Large_Trucks):

    years_left = 0
    AADTT = (Percent_Small_Trucks + Percent_Medium_Trucks +
             Percent_Large_Trucks) * AADT

    # Simulate the remaining lifespan of the pavement after the analysis period
    while IRI < IRI_max:
        years_left += 1
        Age += 1
        T_last += 1
        IRI += 0.08 * np.log(Age) * np.log(AADTT/2) *             SN**-2.5 + np.random.normal(0, 0.05)
        AADTT *= np.random.normal(1 + mean_AADTT, sd_AADTT)

    return -C_last * years_left / T_last


# Test the function
IRI = 1.5  # m/km
IRI_constructed = 1
IRI_max = 2.36

C_last = 60000
T_last = 10
Age = 50
SN = 4.9
AADT = 66667
Percent_Small_Trucks = 0.04
Percent_Medium_Trucks = 0.05
Percent_Large_Trucks = 0.03

# Traffic growth values from FHWA statistics -- VMT from 1980-2018
mean_AADTT = 0.020097337
sd_AADTT = 0.014689839

Salvage = np.zeros(1000)

for i in range(1000):
    Salvage[i] = Get_Salvage_Value(IRI, IRI_constructed, IRI_max, Age, T_last, C_last,
                                   SN, AADT, Percent_Small_Trucks, Percent_Medium_Trucks, Percent_Large_Trucks)

#plt.hist(Salvage, bins=50)
#plt.title('Histogram of salvage value for IRI ' + str(IRI))



# In[38]:


#%timeit Get_Salvage_Value(IRI, IRI_constructed, IRI_max, Age, T_last, C_last, SN, AADT, Percent_Small_Trucks, Percent_Medium_Trucks, Percent_Large_Trucks)


# # LCA Function

# In[39]:


# @profile
def Perform_LCA(IRI_max, traffic_dict, GWP_Q_Table, Cost_Q_Table, w_GWP, w_Cost, Traditional=False):

    # Arrays to hold simulated GWP
    Embodied_GWP = np.zeros((Number_Simulations_LCA, Num_Actions))
    Albedo_GWP = np.zeros((Number_Simulations_LCA, Num_Actions))
    Lighting_GWP = np.zeros((Number_Simulations_LCA, Num_Actions))
    Roughness_GWP = np.zeros((Number_Simulations_LCA, Num_Actions))
    Deflection_GWP = np.zeros((Number_Simulations_LCA, Num_Actions))
    Congestion_GWP = np.zeros((Number_Simulations_LCA, Num_Actions))
    EOL_GWP = np.zeros((Number_Simulations_LCA, Num_Actions))
    Total_GWP = np.zeros((Number_Simulations_LCA, Num_Actions))

    # Arrays to hold simulated costs
    Construction_Cost = np.zeros((Number_Simulations_LCA, Num_Actions))
    # Note that maintenance cost includes reconstructions after t = 0
    Maintenance_Cost = np.zeros((Number_Simulations_LCA, Num_Actions))
    Salvage_Value = np.zeros((Number_Simulations_LCA, Num_Actions))
    Total_Cost = np.zeros((Number_Simulations_LCA, Num_Actions))

    # Array to hold chosen actions during LCA
    Actions_Taken_LCA = np.zeros(
        (Num_Actions, Number_Simulations_LCA, ANALYSIS_PERIOD+1))

    # Loop through actions, simulations and time periods
    for a in range(Num_Actions):

        for i in range(Number_Simulations_LCA):

            # initialize values
            excess_energy_factor = excess_energy_factor_initial

            traffic_dict['AADT'] = AADT_INITIAL
            traffic_dict['AADT_mean'] = AADT_MEAN_INITIAL
            traffic_dict['AADT_std'] = AADT_SD_INITIAL
            # same as aadt can probably remove
            traffic_dict['AADTT_mean'] = AADT_MEAN_INITIAL
            # same as aadt can probably remove
            traffic_dict['AADTT_std'] = AADT_SD_INITIAL
            rho = 1
            traffic_dict['percent_pc'] = PERCENT_PC_INITIAL
            traffic_dict['percent_lt'] = PERCENT_LT_INITIAL
            traffic_dict['percent_mt'] = PERCENT_MT_INITIAL
            traffic_dict['percent_ht'] = PERCENT_HT_INITIAL
            traffic_dict['percent_trucks_impacted'] = PERCENT_TRUCKS_IMPACTED_INITIAL
            traffic_dict['percent_non_trucks_impacted'] = PERCENT_NON_TRUCKS_IMPACTED_INITIAL

            IRI = IRI_constructed
            Age = 0

            Pt = 100
            prev_err = 0

            Get_Truck_Weights()

            ec_gwp_dict = {}

            scen = 1  # np.random.randint(1,7)

            for t in range(ANALYSIS_PERIOD + 1):

                # seed the random number generator to get the same random numbers every time
                np.random.seed(100*i + t)

                # Step 1: Select Action

                if t == 0:
                    Action = a + 1  # We're looping through all initial actions

                elif IRI >= IRI_max:

                    if Traditional == True:
                        # Repair the pavement
                        Action = 6
                    else:
                        # Choose best action based on weighted Q-values, recognizing do nothing is not an option
                        GWP_scores = (
                            Worst_GWP - np.array(GWP_Q_Table[State][:6])) / (Worst_GWP - Best_GWP)
                        Cost_scores = (
                            Worst_Cost - np.array(Cost_Q_Table[State][:6])) / (Worst_Cost - Best_Cost)
                        Action = np.argmax(
                            GWP_scores*w_GWP + Cost_scores*w_Cost) + 1

                else:

                    if Traditional == True:
                        # Do nothing
                        Action = 7
                    else:
                        # Choose best action based on weighted Q-values
                        GWP_scores = (
                            Worst_GWP - np.array(GWP_Q_Table[State])) / (Worst_GWP - Best_GWP)
                        Cost_scores = (
                            Worst_Cost - np.array(Cost_Q_Table[State])) / (Worst_Cost - Best_Cost)
                        Action = np.argmax(
                            GWP_scores*w_GWP + Cost_scores*w_Cost) + 1

                # Record action
                Actions_Taken_LCA[a][i][t] = Action

                # Step 2a: Compute the GWP
                if t == ANALYSIS_PERIOD:
                    eol = Get_Excavation_GWP(
                        SN, Lane_Width, Highway_Length, Number_Lanes, Volume_Mass_Density, True)
                    EOL_GWP[i][a] = eol

                else:
                    eol = 0  # do we need this line to add zero?
                    EOL_GWP[i][a] = eol

                if t == 0:
                    albedo = lighting = roughness = deflection = congestion = 0

                    embodied = Get_Embodied_GWP(
                        Embodied_GWP_Parameters, Action)
                    Embodied_GWP[i][a] = embodied

                else:
                    albedo = Get_Albedo_GWP(
                        Geographic_Location, Current_Albedo, SA)
                    Albedo_GWP[i][a] = albedo

                    lighting = Get_Lighting_GWP(SA)
                    Lighting_GWP[i][a] = lighting

                    roughness = Get_Roughness_GWP(
                        IRI, Speed_Limit, traffic_dict, Highway_Length)
                    Roughness_GWP[i][a] = roughness

                    deflection = Get_Deflection_GWP(
                        traffic_dict, Highway_Length, excess_energy_factor, SN)
                    Deflection_GWP[i][a] = deflection

                    embodied = 0
                    congestion = 0

                    if Action < 7:
                        embodied = Get_Embodied_GWP(Embodied_GWP_Parameters, Action) + Get_Excavation_GWP(
                            SN, Lane_Width, Highway_Length, Number_Lanes, Volume_Mass_Density, False)
                        Embodied_GWP[i][a] = embodied

                        congestion = Get_Congestion_GWP(
                            LDV_Delta, HDV_Delta, traffic_dict, Highway_Length, Construction_Duration, Maintenance_Duration, Action)
                        Congestion_GWP[i][a] = congestion

                Total_GWP[i][a] = albedo + lighting + roughness +                     deflection + embodied + congestion + eol

                # Step 2b: Compute cost
                if Action < 7:
                    C_last = (1/(1+Discount_Rate)**t) *                         Get_Cost(Action, Pt)  # discounted cost
                    Total_Cost[i][a] += C_last

                    if t == 0:
                        Construction_Cost[i][a] += C_last
                    else:
                        Maintenance_Cost[i][a] += C_last

                # Step 3: Update exogenous information
                traffic_dict = Get_Traffic_Composition(traffic_dict, rho)

                # unpackdict temporary varaibles  usued below
                percent_lt = traffic_dict['percent_lt']
                percent_mt = traffic_dict['percent_mt']
                percent_ht = traffic_dict['percent_ht']

                # Update Environment
                Get_Technology_Composition(scenario=scen, analysis_year=t)
                ec_gwp_dict = get_emission_factors(
                    t, "WECC", "Solar", ec_gwp_dict)
                update_energycycle_gwp(ec_gwp_dict)

                ## THIS IS NOT BEING USED EXCEPT FOR EXCESS ENERGY FACTOR CAN MOVE INTO GET_EMISSIONS ##
                # Fuel_Economy_Improvement=np.random.normal(mean_FE_improvement,sd_FE_improvement) #Based on historical data provided by US EPA 1975-2019

                # gal_per_mile_LDV=gal_per_mile_LDV/(1+Fuel_Economy_Improvement)
                # gal_per_mile_HDV=gal_per_mile_LDV/(1+Fuel_Economy_Improvement)
                Fuel_Economy_Improvement = 0
                # We use this to incorporate fuel economy improvements into the deflection function - assume energy consumption scales the same as FE improvements
                excess_energy_factor = excess_energy_factor /                     (1+Fuel_Economy_Improvement)
                ## THIS IS NOT BEING USED CAN MOVE INTO GET_EMISSIONS ##

                # Uncertain cost growth over time
                Pt, prev_err = Get_Pt(prev_err, t)

                # Determine next pavement condition
                if Action < 6:
                    Age = 0
                    T_last = 0
                    SN = SN_States[Action - 1]
                    IRI = IRI_constructed
                elif Action < 7:
                    Age += 1
                    T_last = 0
                    IRI = IRI_constructed
                else:
                    Age += 1
                    T_last += 1
                    IRI += 0.08*math.log(Age)*math.log((percent_lt+percent_mt+percent_ht)
                                                       * AADT/2)*pow(SN, -2.5)+np.random.normal(0, 0.05)

                if IRI < 0.6*IRI_max:
                    IRI_discretized = '<0.6*IRI_max'
                elif IRI < 0.7*IRI_max:
                    IRI_discretized = '<0.7*IRI_max'
                elif IRI < 0.8*IRI_max:
                    IRI_discretized = '<0.8*IRI_max'
                elif IRI < 0.9*IRI_max:
                    IRI_discretized = '<0.9*IRI_max'
                elif IRI < IRI_max:
                    IRI_discretized = '<IRI_max'
                else:
                    IRI_discretized = '>=IRI_max'

                State_next = (t+1, math.floor(Age/10)*10, SN, IRI_discretized)
                State = State_next

            # At the end of the analysis period, determine Salvage Value
            Salvage_Value[i][a] = (1/(1+Discount_Rate)**t) * Get_Salvage_Value(IRI, IRI_constructed, IRI_max, Age, T_last,
                                                                               C_last, SN, AADT, Percent_Small_Trucks, Percent_Medium_Trucks, Percent_Large_Trucks)  # discounted cost
            Total_Cost[i][a] += Salvage_Value[i][a]

    # Data analysis and presentation

    # For plot of GWP vs. cost
    Total_GWP_quantiles = np.zeros((Num_Actions, len(quantiles)))
    Total_Cost_quantiles = np.zeros((Num_Actions, len(quantiles)))

    Average_GWP = np.zeros(Num_Actions)
    Average_Cost = np.zeros(Num_Actions)

    # For summary dataframe
    summary = np.zeros((12, 10))

    # Synthesize results
    for a in range(Num_Actions):

        # Quantiles - for plot of GWP vs. cost
        Total_GWP_quantiles[a, :] = np.quantile(Total_GWP[:, a], quantiles)
        Total_Cost_quantiles[a, :] = np.quantile(Total_Cost[:, a], quantiles)

        # Fill averages into summary dataframe
        summary[0][a*2] = np.average(Embodied_GWP[:, a])
        summary[1][a*2] = np.average(Albedo_GWP[:, a])
        summary[2][a*2] = np.average(Lighting_GWP[:, a])
        summary[3][a*2] = np.average(Roughness_GWP[:, a])
        summary[4][a*2] = np.average(Deflection_GWP[:, a])
        summary[5][a*2] = np.average(Congestion_GWP[:, a])
        summary[6][a*2] = np.average(EOL_GWP[:, a])
        summary[7][a*2] = Average_GWP[a] = np.average(Total_GWP[:, a])

        summary[8][a*2] = np.average(Construction_Cost[:, a])
        summary[9][a*2] = np.average(Maintenance_Cost[:, a])
        summary[10][a*2] = np.average(Salvage_Value[:, a])
        summary[11][a*2] = Average_Cost[a] = np.average(Total_Cost[:, a])

        # Fill percentages into summary dataframe
        summary[0][a*2 + 1] = round(summary[0][a*2] / summary[7][a*2] * 100, 2)
        summary[1][a*2 + 1] = round(summary[1][a*2] / summary[7][a*2] * 100, 2)
        summary[2][a*2 + 1] = round(summary[2][a*2] / summary[7][a*2] * 100, 2)
        summary[3][a*2 + 1] = round(summary[3][a*2] / summary[7][a*2] * 100, 2)
        summary[4][a*2 + 1] = round(summary[4][a*2] / summary[7][a*2] * 100, 2)
        summary[5][a*2 + 1] = round(summary[5][a*2] / summary[7][a*2] * 100, 2)
        summary[6][a*2 + 1] = round(summary[6][a*2] / summary[7][a*2] * 100, 2)
        summary[7][a*2 + 1] = 100.00

        summary[8][a*2 + 1] = round(summary[8]
                                    [a*2] / summary[11][a*2] * 100, 2)
        summary[9][a*2 + 1] = round(summary[9]
                                    [a*2] / summary[11][a*2] * 100, 2)
        summary[10][a*2 + 1] = round(summary[10]
                                     [a*2] / summary[11][a*2] * 100, 2)
        summary[11][a*2 + 1] = 100.00

    # Create summary dataframe
    index = ['Embodied GWP', 'Albedo GWP', 'Lighting_GWP', 'Roughness GWP', 'Deflection GWP', 'Congestion GWP',
             'EOL GWP', 'Total GWP', 'Construction Cost', 'Maintenance Cost', 'Salvage Value', 'Total Cost']
    columns = ['Avg Act. 1', '% Act. 1', 'Avg Act. 2', '% Act. 2', 'Avg Act. 3',
               '% Act. 3', 'Avg Act. 4', '% Act. 4', 'Avg Act. 5', '% Act. 5']
    Summary_Dataframe = pd.DataFrame(summary, index, columns)

    # Plot mean and quantiles of cost and GWP for the 5 initial actions
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    for a in range(Num_Actions):
        color = colors[a]
        plt.scatter(Total_Cost_quantiles[a, :], Total_GWP_quantiles[a,
                                                                    :], label='Quantiles A' + str(a + 1), c=color)
        plt.scatter(Average_Cost[a], Average_GWP[a],
                    label='Mean A' + str(a + 1), c=color, marker='*')

    plt.xlabel('Cost in dollars')
    plt.ylabel('GWP in kg CO2e')
    if Traditional == True:
        plt.title(
            f'Plot of simulated GWP vs. cost for {Num_Actions} actions \n using mean and quantiles {quantiles} \n with IRI max {IRI_max} Run: {RUN_NAME}')
    else:
        plt.title(
            f'Plot of simulated GWP vs. cost for {Num_Actions} actions \n using mean and quantiles {quantiles} \n with IRI max {IRI_max} and Q-table Run: {RUN_NAME}')
    plt.legend()
    file = IMAGE_OUTPUT_PATH + f"/{RUN_NAME}-{IRI_max}.jpg"
    plt.savefig(file)
    
    plt.clf()

    # Display results for best initial action(s)
    c = np.argmin(Average_Cost)
    g = np.argmin(Average_GWP)

    if c == g:
        print('Best action:', c + 1, 'GWP =',
              Average_GWP[c], 'Cost =', Average_Cost[c])
    else:
        print('Best cost action:', c + 1, 'GWP =',
              Average_GWP[c], 'Cost =', Average_Cost[c])
        print('Best GWP action:', g + 1, 'GWP =',
              Average_GWP[g], 'Cost =', Average_Cost[g])

    # Analysing actions taken
    avg_num_repairs = np.zeros(Num_Actions)
    avg_num_constructions = np.zeros(Num_Actions)

    for a in range(Num_Actions):
        avg_num_repairs[a] = np.sum(
            Actions_Taken_LCA[a, :, :] == 6) / Number_Simulations_LCA
        avg_num_constructions[a] = np.sum(
            Actions_Taken_LCA[a, :, :] < 6) / Number_Simulations_LCA

    print('Average number of repairs:', avg_num_repairs)
    print('Average number of constructions:', avg_num_constructions)

    # Writing to excel files

    details = np.zeros((Num_Actions, Number_Simulations_LCA, 14))
    columns = ['Embodied GWP', 'Albedo GWP', 'Lighting_GWP', 'Roughness GWP', 'Deflection GWP', 'Congestion GWP', 'EOL GWP', 'Total GWP',
               'Construction Cost', 'Maintenance Cost', 'Salvage Value', 'Total Cost', 'Number of Maintenances', 'Number of Constructions']

    for a in range(Num_Actions):
        details[a, :, 0] = Embodied_GWP[:, a]
        details[a, :, 1] = Albedo_GWP[:, a]
        details[a, :, 2] = Lighting_GWP[:, a]
        details[a, :, 3] = Roughness_GWP[:, a]
        details[a, :, 4] = Deflection_GWP[:, a]
        details[a, :, 5] = Congestion_GWP[:, a]
        details[a, :, 6] = EOL_GWP[:, a]
        details[a, :, 7] = Total_GWP[:, a]

        details[a, :, 8] = Construction_Cost[:, a]
        details[a, :, 9] = Maintenance_Cost[:, a]
        details[a, :, 10] = Salvage_Value[:, a]
        details[a, :, 11] = Total_Cost[:, a]

        for i in range(Number_Simulations_LCA):
            details[a, i, 12] = np.sum(Actions_Taken_LCA[a, i, :] == 6)
            details[a, i, 13] = np.sum(Actions_Taken_LCA[a, i, :] < 6)

    Action1_Dataframe = pd.DataFrame(details[0, :, :], columns=columns)
    Action2_Dataframe = pd.DataFrame(details[1, :, :], columns=columns)
    Action3_Dataframe = pd.DataFrame(details[2, :, :], columns=columns)
    Action4_Dataframe = pd.DataFrame(details[3, :, :], columns=columns)
    Action5_Dataframe = pd.DataFrame(details[4, :, :], columns=columns)

    if Traditional == True:
        
        file = OUTPUT_PATH + f'/LCA (salvage, 50yr) with IRI_max = {IRI_max}.xlsx'

        with pd.ExcelWriter(file) as writer:
            Summary_Dataframe.to_excel(writer, sheet_name='Summary')
            Action1_Dataframe.to_excel(
                writer, sheet_name='Action 1', index=False)
            Action2_Dataframe.to_excel(
                writer, sheet_name='Action 2', index=False)
            Action3_Dataframe.to_excel(
                writer, sheet_name='Action 3', index=False)
            Action4_Dataframe.to_excel(
                writer, sheet_name='Action 4', index=False)
            Action5_Dataframe.to_excel(
                writer, sheet_name='Action 5', index=False)

    else:
        
        file = OUTPUT_PATH + f'/LCA with Qtable {w_GWP}-{w_Cost}.xlsx'

        with pd.ExcelWriter(file) as writer:
            Summary_Dataframe.to_excel(writer, sheet_name='Summary')
            Action1_Dataframe.to_excel(
                writer, sheet_name='Action 1', index=False)
            Action2_Dataframe.to_excel(
                writer, sheet_name='Action 2', index=False)
            Action3_Dataframe.to_excel(
                writer, sheet_name='Action 3', index=False)
            Action4_Dataframe.to_excel(
                writer, sheet_name='Action 4', index=False)
            Action5_Dataframe.to_excel(
                writer, sheet_name='Action 5', index=False)

    return Summary_Dataframe, Total_GWP_quantiles, Total_Cost_quantiles, Total_GWP, Total_Cost


# # Case U1: Urban Interstate

# In[40]:


ANALYSIS_PERIOD = 50  # years
Number_Simulations_LCA = 1

Discount_Rate = 0.007  # Source: OMB Circular A-94 Appendix C (2016)

Num_Actions = 5
Age_States = [0, 10, 20, 30, 40]

IRI_constructed = 1

IRI_max = 2.36  # m/km
SN_initial = 3.65
SN_States = [3.65, 4.44, 5.22, 6.00, 6.79]

IRI_States = ['<0.6*IRI_max', '<0.7*IRI_max',
              '<0.8*IRI_max', '<0.9*IRI_max', '<IRI_max', '>=IRI_max']
# Pt_States = ['<0.9*E[Pt]', '<1.1*E[Pt]', '>=1.1*E[Pt]'] # removed because it didn't have much of an effect on price optimization

# Input values for GWP Function
Geographic_Location = 'Seattle'
Current_Albedo = 0.12

# Input pavement dimensions
Lane_Width = 12  # ft
Highway_Length = 1  # mile
Number_Lanes = 6
Number_Shoulders = 2
Shoulder_Width = 8  # ft

# Calculate pavement surface area in square meters
SA = (Number_Lanes*Lane_Width + Number_Shoulders*Shoulder_Width) *     FT_TO_METER * Highway_Length*MILE_TO_METER

Speed_Limit = 65  # mph

# Based on FHWA VMT Stats 2018
AADT_INITIAL = 81289
AADTT_INITIAL = 800
PERCENT_LT_INITIAL = 0.19
PERCENT_MT_INITIAL = 0.04
PERCENT_HT_INITIAL = 0.09
PERCENT_PC_INITIAL = 1 -     sum([PERCENT_LT_INITIAL, PERCENT_MT_INITIAL, PERCENT_HT_INITIAL])

# Axle loads all in Newtons
Pavement_Stiffness = 8000  # MPa
Volume_Mass_Density = 2243  # kg/cubic-meter
Subgrade_Stiffness = 112.8  # MPa
excess_energy_factor_initial = 1

# Traffic growth values from FHWA statistics -- VMT from 1980-2018
AADT_MEAN_INITIAL = 0.020097337
AADT_SD_INITIAL = 0.014689839
rho = 1
mean_FE_improvement = 0.01146134
sd_FE_improvement = 0.0377051

# Input values for congestion function
Construction_Duration = 120  # Work days
Maintenance_Duration = 40  # Work days
# Percent of AADT that is passenger cars impacted by construction/maintenace - value from Hallenbeck, et al., Vehicle Volume Distributions by Classification, 1997.  - used in FHWA TMG
PERCENT_NON_TRUCKS_IMPACTED_INITIAL = 0.154
# Percent of AADT that is trucks impacted by construction/maintenance - value from FHWA data traffic pocket guide
PERCENT_TRUCKS_IMPACTED_INITIAL = 0.097 *     (PERCENT_LT_INITIAL+PERCENT_MT_INITIAL+PERCENT_HT_INITIAL)/AADT_INITIAL


# Fitted model for roadway construction cost
asphalt_log_q, asphalt_const, asphalt_sd = -0.1490314, 5.475205, 0.3663
aggregate_log_q, aggregate_const, aggregate_sd = -0.1057854, 3.174054, 0.53427
milling_log_q, milling_const, milling_sd = -0.3230392, 4.217846, 0.48426
overlay_log_q, overlay_const, overlay_sd = -0.0813632, 4.750021, 0.17386
# Cost setup function
Log_asph_cost_mean, Log_agg_cost_mean, Log_milling_cost_mean, Log_overlay_cost_mean = Pavement_Cost_Setup(
    SN_States, SA)

# Embodied GWP Setup Function
file = DEPENDANCY_PATH + '/Input_Processes_CaseU1_UrbanInterstate.xls'
GWP_Sheet_Name = 'GWP'
Embodied_GWP_Parameters = Embodied_GWP_Setup(
    file, GWP_Sheet_Name)

# Calculate deflection function parameters
Excess_Energy_DataFrame = Get_Deflection_Excess_Energy_DataFrame(
    Pavement_Stiffness, Speed_Limit, Volume_Mass_Density, Subgrade_Stiffness, Geographic_Location, sn_states=SN_States)
# Create lists to hold the LCA summary dataframes, expected costs/GWP, and lists of all iterations of cost/GWP
Summary_df_list = []
Total_GWP_list = []
Total_Cost_list = []

# Create lists to hold quantiles
quantiles = [0.01, 0.25, 0.5, 0.75, 0.99]
GWP_quantiles_list = []
Cost_quantiles_list = []

# df_vehicles


# In[ ]:





# # Discritizers and Approximator Helpers

# ### Approximate

# In[41]:


T_STATES = np.arange(0,50+1)
AGE_STATES = np.arange(0,50+1)
SN_STATES = np.array([3.65,4.44,5.22,6.00,6.79])
IRI_STATES = np.array([1.36, 1.56, 1.76, 1.96, 2.16, 2.36, 2.56, 2.76, 2.96, 3.16, 3.36, 3.56, 3.76])

@njit
def approximate_state(state):
        
        #note, the numpy implementation with all it's ugly syntax is over 20x faster than pure python lists and lambda
        
        t,age,sn,iri = state.T
        
        #approximate time
        idx = (np.abs(T_STATES - t)).argmin()
        t = T_STATES[idx]
        #approximate age             
        idx = (np.abs(AGE_STATES - age)).argmin()
        age = AGE_STATES[idx]     
        #approximate sn        
        idx = (np.abs(SN_STATES - sn)).argmin()
        sn = SN_STATES[idx]
        #approximate iri               
        idx = (np.abs(IRI_STATES - iri)).argmin()
        iri = IRI_STATES[idx]
        
        state = np.array([t,age,sn,iri])
        
        return state
        
    
approximate_state(np.array([20.8,8.4,0.32,3.5]))


# In[42]:


#%timeit approximate_state(np.array([20.8,8.4,0.32,3.5]))


# ### Discretize

# In[43]:


IRI_MAX = 2.36
SN_STATES = np.array([3.65,4.44,5.22,6.00,6.79])


def discritize_state(state):
    
    t,age,sn,iri,scen,aadtt,pt = state.T


    if iri < 0.6*IRI_MAX:
        iri_discretized = 0
    elif iri < 0.7*IRI_MAX:
        iri_discretized = 1
    elif iri < 0.8*IRI_MAX:
        iri_discretized = 2
    elif iri < 0.9*IRI_MAX:
        iri_discretized = 3
    elif iri < IRI_MAX:
        iri_discretized = 4
    else:
        iri_discretized = 5
        
    
    if sn < 3.65:
        sn_discritized = 0
    elif sn < 4.44:
        sn_discritized = 1
    elif sn < 5.22:
        sn_discritized = 2
    elif sn < 6.00:
        sn_discritized = 3
    else:
        sn_discritized = 4
        
        
    if aadtt < 900:
        aadtt_discritized = 0
    elif aadtt < 100:
        aadtt_discritized = 1
    elif aadtt < 1100:
        aadtt_discritized = 2
    elif aadtt < 1200:
        aadtt_discritized = 3
    else:
        aadtt_discritized = 4
        
    if pt < 105:
        pt_discritized = 0
    elif pt < 140:
        pt_discritized = 1
    elif pt < 175:
        pt_discritized = 2
    elif pt < 210:
        pt_discritized = 3
    else:
        pt_discritized = 4
    
    
    
    
    
    
    state = np.array([t,age,sn,iri_discretized,scen,aadtt_discritized,pt_discritized],dtype = 'int8')
    
    

    return state


# In[44]:


#timeit discritize_state(np.array([20.8,8.4,0.32,3.5,3,884.455,264.568]))

discritize_state(np.array([20.8,8.4,0.32,3.5,3,884.455,264.568]))


# # Reinforcement Learning

# ## Environment

# In[45]:


import gym
from gym import Env
from gym.spaces import Discrete, Box,MultiDiscrete


# In[46]:


class LCAEnv(Env):
    
    
    
    def __init__(self):
        # Actions we can take
        self.action_space = Discrete(7) #0-6
        # States array
        # time, age, sn, iri, scenario, aadtt, pt
        self.observation_space = Box(low=np.array([0,0,3.43,1.36,0,800,100]), high=np.array([50,50,6.37,3.76,6,1200,270]))
        
        # For discrete space use:
        # MultiDiscrete
    
        
    
        #Variables for entire instance
        self.traffic_dict = {}
        self.ec_gwp_dict = {}
        self.scen = 1 + np.random.randint(0,6)
        self.excess_energy_factor = excess_energy_factor_initial
        
        
        
        #State variables
        self.t = 0
        self.age = 0
        self.sn = SN_initial
        self.iri = IRI_constructed
        self.aadtt = AADTT_INITIAL
        
        #Other variables
        self.pt = 100
        self.prev_err = 0
        self.c_last = 60000

        Get_Truck_Weights()
        
        
        #Set State Array
        self.state = np.array([self.t,self.age,self.sn,self.iri,self.scen,self.aadtt,self.pt])
        
        #information accumulator
        self.info = []
            

        
               
    def step(self, action):

    
        
        #Update Exogenous Information
        self.traffic_dict = Get_Traffic_Composition(self.traffic_dict, self.rho)
        percent_lt = self.traffic_dict['percent_lt']
        percent_mt = self.traffic_dict['percent_mt']
        percent_ht = self.traffic_dict['percent_ht']
        AADT = self.traffic_dict['AADT']
        self.aadtt = self.traffic_dict['AADTT']
        Get_Technology_Composition(scenario=self.scen,analysis_year=int(self.t))
        #print("getting emissiom factors self.t, self.ec_gwp_dict",self.t,self.ec_gwp_dict)
        self.ec_gwp_dict = get_emission_factors(int(self.t),"WECC","Solar",self.ec_gwp_dict)
        update_energycycle_gwp(self.ec_gwp_dict)
        Fuel_Economy_Improvement = 0
        self.excess_energy_factor= self.excess_energy_factor/(1+Fuel_Economy_Improvement) #We use this to incorporate fuel economy improvements into the deflection function - assume energy consumption scales the same as FE improvements
        self.pt, self.prev_err = Get_Pt(self.prev_err, int(self.t))
        #Update State Variables

        if action < 5:
            self.age = 0
            self.T_last = 0 #todo: what is T_last?
            self.t += 1
            self.sn = SN_States[action]
            self.iri = IRI_constructed
        elif action < 6:
            self.age += 1
            self.T_last = 0
            self.t += 1
            self.iri = IRI_constructed
        else:
            self.age += 1
            self.T_last += 1
            self.t += 1
            self.iri += 0.08*math.log(self.age)*math.log((percent_lt+percent_mt+percent_ht)*AADT/2)*pow(self.sn,-2.5)+np.random.normal(0,0.05)


        #store state    
        self.state = np.array([self.t,self.age,self.sn,self.iri,self.scen,self.aadtt,self.pt])
        
        
        # Calculate reward
        #print("before reward, state is:",self.state,action)
        reward,total_cost = self.reward_calc(self.state,action)
        #print(f"internal reward {reward}")
        
        #check if done
        
        if self.t>= 50: #>= or >?
            done = True
        else:
            done = False
            
        #Can impose a heavy penalty into reward for iri exceeding limit
        #print(type(self.state.tolist()))
        info = {'state':self.state,'action':action,'total_cost':total_cost,'total_gwp':reward} #todo: modify this to return episode-wise by making cost, reward methods and appending during rest.
        #adding state to this info dict breaks the whole thing
        self.info.append(info)

        # Return step information
        return self.state, reward, done, info
    
    def reward_calc(self, state, action):

        
        #Global Warming Potential Calculations
        lighting,albedo,roughness,deflection,embodied,eol,congestion = (0,0,0,0,0,0,0)
        
        lighting = Get_Lighting_GWP(SA)
        albedo = Get_Albedo_GWP(Geographic_Location, Current_Albedo, SA)
        #print("speed limit,"Speed_Limit)
        roughness = Get_Roughness_GWP(self.iri, Speed_Limit, self.traffic_dict, Highway_Length)
        deflection = Get_Deflection_GWP(self.traffic_dict, Highway_Length, self.excess_energy_factor, self.sn)
        
        if self.t == 0:            
            embodied = Get_Embodied_GWP(Embodied_GWP_Parameters, action+1)
        elif self.t == ANALYSIS_PERIOD:
            eol = Get_Excavation_GWP(self.sn, Lane_Width, Highway_Length, Number_Lanes, Volume_Mass_Density, True)
            congestion = Get_Congestion_GWP(LDV_Delta, HDV_Delta, self.traffic_dict, Highway_Length, Construction_Duration, Maintenance_Duration, action+1)
        
        if action <6: #actions 1-6
            embodied = Get_Excavation_GWP(self.sn, Lane_Width, Highway_Length, Number_Lanes, Volume_Mass_Density, False)
            embodied += Get_Embodied_GWP(Embodied_GWP_Parameters, action+1)

        
        
        #Dollar Cost Calculations, for now bundle all costs together        
        ## total cost = sum(construction or maintenance,salvage)
        
        #Must pull these local varibles in reward calc or make them class methods
        percent_lt = self.traffic_dict['percent_lt']
        percent_mt = self.traffic_dict['percent_mt']
        percent_ht = self.traffic_dict['percent_ht']
        
        
        if action < 6: #todo: this if statement will be combined with above later, seperate for now.
            self.c_last = (1/(1+Discount_Rate)**self.t) * Get_Cost(action, self.pt) # discounted cost
            #todo: if statement to divide maintenance and construction
        
        salvage_value = (1/(1+Discount_Rate)**self.t) * Get_Salvage_Value(self.iri, IRI_constructed, IRI_max, self.age, self.T_last, self.c_last, self.sn, AADT, percent_lt, percent_mt, percent_ht) # discounted cost
        
        #todo: does total cost always include self.c_last?
        
        total_cost = self.c_last + salvage_value
        

          

        #calculate gwp
        reward = lighting+albedo+roughness+deflection+embodied+eol+congestion
        #print("lighting,",lighting)
        #print("albedo,",albedo)      
        #print("roughness,",roughness)
        #print("deflection,",deflection)
        reward = -reward #minimize gwp
        
        
        #GLOBAL WARMING POTENTIAL CALCULATIONS

        return reward,total_cost #minimize gwp!

    def render(self):
        
        if self.t %10 ==0:
            print("the state right now is",self.state)
            print(f"internal reward is",self.reward)
        else:
            pass
       

    
    def reset(self):
        
        #print("Environment Has Been Reset")
            
        self.excess_energy_factor = excess_energy_factor_initial

        self.traffic_dict['AADT'] = AADT_INITIAL
        self.traffic_dict['AADTT'] = AADTT_INITIAL
        self.traffic_dict['AADT_mean'] = AADT_MEAN_INITIAL
        self.traffic_dict['AADT_sd'] = AADT_SD_INITIAL
        self.rho = 1
        self.traffic_dict['percent_pc'] = PERCENT_PC_INITIAL
        self.traffic_dict['percent_lt'] = PERCENT_LT_INITIAL
        self.traffic_dict['percent_mt'] = PERCENT_MT_INITIAL
        self.traffic_dict['percent_ht'] = PERCENT_HT_INITIAL
        self.traffic_dict['percent_trucks_impacted'] = PERCENT_TRUCKS_IMPACTED_INITIAL
        self.traffic_dict['percent_non_trucks_impacted'] = PERCENT_NON_TRUCKS_IMPACTED_INITIAL
        
    

        Get_Truck_Weights()
        

        
        #RESET TO SAME AS INITIALIZATION
        
        #High level params
        self.scen = 1 + np.random.randint(0,6) #can draw 0-5 + 1 so scen = 1-6
        self.excess_energy_factor = excess_energy_factor_initial
        
        
        
        #State variables
        self.t = 0
        self.age = 0
        self.sn = SN_initial
        self.iri = IRI_constructed
        
        #Other variables
        self.pt = 100
        self.prev_err = 0
        self.c_last = 60000

        
        #Super other variables
        self.T_last = 0
        
        
        #Set State Array
        self.state = np.array([self.t,self.age,self.sn,self.iri,self.scen,self.aadtt,self.pt])
        
        
        
        return self.state


# In[ ]:





# In[47]:


env = LCAEnv()

# In[48]:



print("RANDOM SAMPLES OF ACTION AND ENVIRONMENT ARE INDEPENDENT.\nJust sampling to make sure we can call every action and space")
for x in range(0,50):
    print("action:", env.action_space.sample(),"Environment:",env.observation_space.sample())


# In[49]:


episodes = 15
full_gwp_list = []
full_cost_list = []

full_info = []
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    episode_gwp_list = []
    episode_cost_list = []
    
    while not done:
        #env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        #print(action,n_state)
        episode_gwp_list.append(reward)
        episode_cost_list.append(info['total_cost'])
        full_info.append(info)
        
    gwp = np.average(episode_gwp_list)
    cost = np.average(episode_cost_list)
    print('Episode:{} Score:{:.2f} Cost: {:.2f}'.format(episode, gwp,cost))

    full_gwp_list.append(gwp)
    full_cost_list.append(cost)

print('average gwp reward of {:.2f} episodes conducting random actions:{:.2f}'.format(episodes,(np.average(full_gwp_list))))
print('average cost reward of {:.2f} episodes conducting random actions:{:.2f}'.format(episodes,(np.average(full_cost_list))))
#plt.plot(full_cost_list,label = 'total cost per episode /50',marker='.',linestyle='none')
#plt.plot(full_gwp_list,label = 'total gwp per episode /50',marker='.',linestyle='none')
#plt.legend()
#


# In[50]:


import gym




# In[52]:


states = env.observation_space.shape[0] #numpy array with real bounds
print("States Space",states)
actions = env.action_space.n #discrete
print("Actions Space",actions)


# In[53]:

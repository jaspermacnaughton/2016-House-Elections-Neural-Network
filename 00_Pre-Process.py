import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

allData = pd.read_csv("./data/CandidateSummaryAction1.csv")
currency_fields = ("ind_ite_con", "ind_uni_con", "ind_con", "par_com_con", "oth_com_con", "can_con", "tot_con", "tra_fro_oth_aut_com", "can_loa", "oth_loa", "tot_loa", "off_to_ope_exp", "off_to_fun", "off_to_leg_acc", "oth_rec", "tot_rec", "ope_exp", "exe_leg_acc_dis", "fun_dis", "tra_to_oth_aut_com", "can_loa_rep", "oth_loa_rep", "tot_loa_rep", "ind_ref", "par_com_ref", "oth_com_ref", "tot_con_ref", "oth_dis", "tot_dis", "cas_on_han_beg_of_per", "cas_on_han_clo_of_per", "net_con", "net_ope_exp", "deb_owe_by_com", "deb_owe_to_com")

# Other Unworked Way: .str.extract(r'(\d+\.?\d+)').astype(np.float)
for field in currency_fields:
    allData[field] = allData[field].str.replace(r'[\',$()]', '').astype(np.float)
    allData[field] = allData[field].fillna(0)

allData["votes"] = allData["votes"].fillna(0)
allData["winner"] = allData["winner"].map(lambda x: x == 'Y')

# Just use house elections, trim columns, and normalize numeric data
full_house = pd.DataFrame(allData[allData.can_off == 'H'])
numeric_cols = list(currency_fields)
house = full_house[numeric_cols]
house = normalize(house, norm='l2')
house = pd.DataFrame(house, columns = numeric_cols)

# Add Dummy Variables - Need to wrap numpy array so row nums not saved, causing misalignment
# CAREFUL to avoid dummy trap: add n-1 cols
house["incumbent"] = np.array(full_house['can_inc_cha_ope_sea'].map(lambda x: int(x == "INCUMBENT")))
house["challenger"] = np.array(full_house['can_inc_cha_ope_sea'].map(lambda x: int(x == "CHALLENGER")))
house["open_seat"] = np.array(full_house['can_inc_cha_ope_sea'].map(lambda x: int(x == "OPEN")))

# 93% either republican or democrat - chalk rest of them
full_house["can_par_aff"].value_counts()
house["democrat"] = np.array(full_house['can_par_aff'].map(lambda x: int(x == "DEM")))
house["republican"] = np.array(full_house['can_par_aff'].map(lambda x: int(x == "REP")))

# And finally, add Winner column
house["winner"] = np.array(full_house["winner"])


house.to_pickle("./data/preprocessed_house.pkl")
from __future__ import print_function
import glob
import json
import numpy as np

pattern =  "*-spectros/*-JPLUS13-magnitude.json"

file_list = glob.glob(pattern)

def write_table(columns, col_names):
    """
    Write an ascii table of columns (sequence of sequences), using col_names as the header
    """
    table = "# " + "\t".join(col_names) + "\n"
    for row in zip(*columns):
        table += "\t".join(row) + "\n"
    return table
        

# Initialize a list for each column in output table
col_names = [ "F348", "F378", "F395", 
              "F410", "F430", "g_sdss", 
               "F515", "r_sdss", "F660", 
               "i_sdss", "F861", "z_sdss", "type"]
table = {cn: [] for cn in col_names}


for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
    table["F348"].append(str(data["F348"]))
    table["F378"].append(str(data["F378"]))
    table["F395"].append(str(data["F395"]))
    table["F410"].append(str(data["F410"]))
    table["F430"].append(str(data["F430"]))
    table["g_sdss"].append(str(data["F480_g_sdss"]))
    table["F515"].append(str(data["F515"]))
    table["r_sdss"].append(str(data["F610_r_sdss"]))
    table["F660"].append(str(data["F660"]))
    table["i_sdss"].append(str(data["F760_i_sdss"]))
    table["F861"].append(str(data["F861"]))
    table["z_sdss"].append(str(data["F910_z_sdss"]))
    if data["id"].endswith("HPNe"):
        table["type"].append(str(0))  
    else:
        table["type"].append(str(1)) 
    

  
  
    
# Write output table to a file
with open("summary-jplus.tab", "w") as f:
    f.write(write_table([table[cn] for cn in col_names], col_names))

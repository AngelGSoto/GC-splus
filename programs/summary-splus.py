from __future__ import print_function
import glob
import json
import numpy as np

pattern =  "*-spectros/*-SPLUS-magnitude.json"

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
col_names = [ "Object", "uJAVA", "F378", "F395", 
              "F410", "F430", "gSDSS", 
              "F515", "F625", "F660", 
              "iSDSS", "F861", "zSDSS"]
table = {cn: [] for cn in col_names}


for file_name in file_list:
    with open(file_name) as f:
        data = json.load(f)
    table["Object"].append(str(data["id"]))
    table["uJAVA"].append(str(data["F348_uJAVA"]))
    table["F378"].append(str(data["F378"]))
    table["F395"].append(str(data["F395"]))
    table["F410"].append(str(data["F410"]))
    table["F430"].append(str(data["F430"]))
    table["gSDSS"].append(str(data["F480_gSDSS"]))
    table["F515"].append(str(data["F515"]))
    table["F625"].append(str(data["F625"]))
    table["F660"].append(str(data["F660"]))
    table["iSDSS"].append(str(data["F770_iSDSS"]))
    table["F861"].append(str(data["F861"]))
    table["zSDSS"].append(str(data["F910_zSDSS"]))
    

    
# Write output table to a file
with open("summary-splus.tab", "w") as f:
    f.write(write_table([table[cn] for cn in col_names], col_names))
    

        

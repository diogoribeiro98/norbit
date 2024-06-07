import numpy as np

def metalic_mean(n):
    return (n + np.sqrt(n**2+4))/2

#Ratios
golden_ratio = metalic_mean(1)
silver_ratio = metalic_mean(2)
bronze_ratio = metalic_mean(3)

#Figure sizes for papers
single_column_width = 3 + 3/8
double_column_width = 2*single_column_width
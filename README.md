# Stress-testing Calvi (JPE, 2020)

The main goal of the program is to disentangle the cohort effect from the hypothesized age effect in Calvi (2020). This will move forward in two stages: 

1. The Consumption surveys done before 2011-12, the one used by Calvi (2020), do not contain data on assignable clothing items. 
This means that the SAP+SAT assumption used to estimate the Engel curve equations are no longer applicable. So I must first write up the NLSUR estimation with SAT. 
That essentially means writing up an MLE program with Normal errors that runs until the var-covar matrix of the errors settles. 
2. Once that is completed, I will run the same program for previous years datasets and see the age profile of women's resource shares. 

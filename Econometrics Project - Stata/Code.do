destring Year GDPgrowth GDP Unemployment , replace force
tsset Year
tsline GDPgrowth
tsfilter hp c_gdp=GDP
tsfilter hp c_unem=Unemployment

asdoc regress c_gdp c_unem 

twoway (lfitci c_gdp c_unem) (scatter c_gdp c_unem)
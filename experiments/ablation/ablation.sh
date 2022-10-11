
# remove one data inpuit
python3.9 run.py experiment=conv3d_sat_nwp model.include_nwp=False
python3.9 run.py experiment=conv3d_sat_nwp model.include_pv_or_gsp_yield_history=False
python3.9 run.py experiment=conv3d_sat_nwp model.include_pv_yield_history=False
python3.9 run.py experiment=conv3d_sat_nwp model.include_sun=False
python3.9 run.py experiment=conv3d_sat_nwp model.include_satellite=False

## remove all data inputs apart from one
# satellite
python3.9 run.py experiment=conv3d_sat_nwp\
 model.include_satellite=True\
 model.include_pv_yield_history=False\
 model.include_pv_or_gsp_yield_history=False\
 model.include_nwp=False\
 model.include_sun=False
# nwp
python3.9 run.py experiment=conv3d_sat_nwp\
 model.include_satellite=False\
 model.include_pv_yield_history=False\
 model.include_pv_or_gsp_yield_history=False\
 model.include_nwp=True\
 model.include_sun=False
# gsp
python3.9 run.py experiment=conv3d_sat_nwp\
 model.include_satellite=False\
 model.include_pv_yield_history=False\
 model.include_pv_or_gsp_yield_history=True\
 model.include_nwp=False\
 model.include_sun=False
# pv
python3.9 run.py experiment=conv3d_sat_nwp\
 model.include_satellite=False\
 model.include_pv_or_gsp_yield_history=False\
 model.include_pv_yield_history=True\
 model.include_nwp=False\
 model.include_sun=False
# sun
python3.9 run.py experiment=conv3d_sat_nwp\
 model.include_satellite=False\
 model.include_pv_yield_history=False\
 model.include_pv_or_gsp_yield_history=False\
 model.include_nwp=False\
 model.include_sun=True
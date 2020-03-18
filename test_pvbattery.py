import unittest
from pvbattery import *


class TestPowerFlowsMethod(unittest.TestCase):
    # PV GIS tool: https://re.jrc.ec.europa.eu/pvg_tools/en/tools.html#PVP
    REL_DEV = 0.1      # maximum acceptable relative deviation for our results compared to PV GIS tool

    def setUp(self) -> None:
        self.ghi = get_irradiance_data()
        self.P_l = get_load_data()

    def power_flows_ideal(self, mod_irrads):
        """ Calculate power flows assuming no system losses and for a install capacity of 1 kWp
        (equivalent to 5m^2 and 20% efficiency) """

        return power_flows(mod_irrads=mod_irrads, P_l=self.P_l, A_pv=5, P_max_inv=2, E_batt_max=10,
                           P_max_batt_charge=1, P_max_batt_discharge=1, eta_batt_charge=1,
                           eta_batt_discharge=1, eta_pv=0.2, eta_inv=1)

    def test_power_flows(self):
        # variation in tilt for south facing panel
        E_GIS = [985.05, 1164.72, 1115.73]  # yearly pv production from PV GIS tool [kWh]
        var_GIS = [32.15, 48.38, 54.86]     # yearly variability [kWh]
        tilts = [0, 30, 60]
        for tilt, E_bm, var_bm in zip(tilts, E_GIS, var_GIS):
            mod_irrads = module_irradiance(ghi_series=self.ghi, mod_tilt=tilt, mod_azi=180, geo_latitude=LATITUDE,
                                           geo_longitude=LONGITUDE)
            df_flows = self.power_flows_ideal(mod_irrads)
            E_pv = yearly_pv_production(df_flows)
            print(E_pv)
            self.assertTrue((1 - self.REL_DEV) * (E_bm - var_bm) < E_pv < (1 + self.REL_DEV) * (E_bm + var_bm))

    def test_power_flows2(self):
        # variation in tilt for east facing panel
        E_GIS = [949.73, 823.82]  # yearly pv production from PV GIS tool [kWh]
        var_GIS = [35.54, 36.72]  # yearly variability [kWh]
        tilts = [30, 60]
        for tilt, E_bm, var_bm in zip(tilts, E_GIS, var_GIS):
            mod_irrads = module_irradiance(ghi_series=self.ghi, mod_tilt=tilt, mod_azi=90, geo_latitude=LATITUDE,
                                           geo_longitude=LONGITUDE)
            df_flows = self.power_flows_ideal(mod_irrads)
            E_pv = yearly_pv_production(df_flows)
            print(E_pv)
            self.assertTrue((1 - self.REL_DEV) * (E_bm - var_bm) < E_pv < (1 + self.REL_DEV) * (E_bm + var_bm))

    def test_power_flows3(self):
        # variation in tilt for west facing panel
        E_GIS = [938.04, 809.51]  # yearly pv production from PV GIS tool [kWh]
        var_GIS = [33.08, 34.38]  # yearly variability [kWh]
        tilts = [30, 60]
        for tilt, E_bm, var_bm in zip(tilts, E_GIS, var_GIS):
            mod_irrads = module_irradiance(ghi_series=self.ghi, mod_tilt=tilt, mod_azi=270, geo_latitude=LATITUDE,
                                           geo_longitude=LONGITUDE)
            df_flows = self.power_flows_ideal(mod_irrads)
            E_pv = yearly_pv_production(df_flows)
            print(E_pv)
            self.assertTrue((1 - self.REL_DEV) * (E_bm - var_bm) < E_pv < (1 + self.REL_DEV) * (E_bm + var_bm))

    def test_power_flows4(self):
        # monthly comparison for specific case, west facing module at 60° tilt
        E_GIS = [18.85, 30.65, 63.85, 94.73, 107.31, 116.76, 116.22, 97.41, 76.71, 50.1, 22.74, 14.46]
        mod_irrads = module_irradiance(ghi_series=self.ghi, mod_tilt=60, mod_azi=270, geo_latitude=LATITUDE,
                                       geo_longitude=LONGITUDE)
        df_flows = self.power_flows_ideal(mod_irrads)
        #TODO
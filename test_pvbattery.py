import unittest
from pvbattery import *
import matplotlib.pyplot as plt
import itertools


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
            self.assertTrue((1 - self.REL_DEV) * (E_bm - var_bm) < E_pv < (1 + self.REL_DEV) * (E_bm + var_bm))

    def test_power_flows4(self):
        # monthly comparison for specific case, west facing module at 60° tilt
        E_GIS = np.array([18.85, 30.65, 63.85, 94.73, 107.31, 116.76, 116.22, 97.41, 76.71, 50.1, 22.74, 14.46])
        mod_irrads = module_irradiance(ghi_series=self.ghi, mod_tilt=60, mod_azi=270, geo_latitude=LATITUDE,
                                       geo_longitude=LONGITUDE)
        df_flows = self.power_flows_ideal(mod_irrads)
        monthly = monthly_pv_production(df_flows)

        plt.plot(E_GIS)
        plt.plot(monthly)
        plt.legend(['Monthly production GIS [kWh]', 'Monthly production ours [kWh]'])
        plt.show()


class NetPresentValueTest(unittest.TestCase):
    def setUp(self) -> None:
        ghi = get_irradiance_data()
        self.grid_flow = pd.Series(data=0, index=ghi.index)
        self.price_parameters = {
            'electricity price': 0,
            'yearly electricity price increase': 0,
            'prosumer tariff': 0,
            'investment': 0,
            'O&M': lambda t: 0,
            'distribution tariff': 0,
            'transmission tariff': 0,
            'taxes & levies': 0,
            'salvage value': 0,
            'capacity tariff': 0
        }

    def test_net_present_value1(self):
        npv = net_present_value(power_flow_grid=self.grid_flow, price_parameters=self.price_parameters,
                                inverter_size=0, study_period=20, discount_rate=0.00, cf_other=lambda t: 0)
        self.assertEqual(npv, 0)

    def test_net_present_value2(self):
        self.price_parameters['investment'] = 1000
        npv = net_present_value(power_flow_grid=self.grid_flow, price_parameters=self.price_parameters,
                                inverter_size=0, study_period=20, discount_rate=0.05, cf_other=lambda t: 0)
        self.assertEqual(npv, -1000)

    def test_net_present_value3(self):
        prices = [1, 2]
        powers = [10, 3]
        periods = [10, 20]
        rates = [0.01, 0.08]
        price_increases = [0.01, 0.03]

        for price, power, period, rate, price_increase in itertools.product(prices, powers, periods, rates, price_increases):
            self.price_parameters['electricity price'] = price
            self.price_parameters['yearly electricity price increase'] = price_increase
            self.grid_flow = self.grid_flow.apply(lambda x: power)
            npv = net_present_value(power_flow_grid=self.grid_flow, price_parameters=self.price_parameters,
                                    inverter_size=0, study_period=period, discount_rate=rate, cf_other=lambda t: 0)

            self.assertAlmostEqual(npv, float(np.sum([(365 * 24 * (-power) * price * (1 + price_increase)**(t-1)
                                                       / (1 + rate)**t) for t in range(1, period + 1)])), places=3)
import pvbattery
import numpy as np
import matplotlib.pyplot as plt


class OptimizationCase(object):
    def __init__(self):
        self._ghi = pvbattery.get_irradiance_data()
        self._load = pvbattery.get_load_data()

        # TODO: fill in default values
        self.price_params = {
            'electricity price': 0.05,
            'yearly electricity price increase': 0,
            'price remuneration': 0.05,
            'prosumer tariff': 0,
            'investment': 0,
            'O&M': lambda t: 0,
            'distribution tariff': 0,
            'transmission tariff': 0,
            'taxes & levies': 30,
            'salvage value': 0,
        }

        self.system_params = {
            'A_pv': 0,                  # total surface area of ALL pv modules, distributed evenly over orientations
            'P_max_inv': 0,
            'E_batt_max': 0,
            'P_max_batt_charge': 0,
            'P_max_batt_discharge': 0,
            'eta_batt_charge': 0.9,
            'eta_batt_discharge': 0.9,
            'eta_pv': 0.20,
            'eta_inv': 0.96,
        }

        self.misc_params = {
            'study_period': 20,
            'discount_rate': 0.05,
            'cf_other': lambda t: 0     # other cash flows
        }

        # the tilts and azimuths of the different pv surface
        # e.g. gable roof with east-west orientation mod_azis = [90, 270] and mod_tilt= [x, x]
        self.mod_tilts = []
        self.mod_azis = []

    def net_present_value(self):
        return pvbattery.net_present_value(self.power_flows()['P grid'],
                                           price_parameters=self.price_params,
                                           inverter_size=self.system_params['P_max_inv'],
                                           **self.misc_params)

    def power_flows(self):
        """ Calculate power flows in the system accounting for multiple orientations. """
        mod_irrads = []
        for i, (azi, tilt) in enumerate(zip(self.mod_azis, self.mod_tilts)):
            mod_irrads.append(pvbattery.module_irradiance(ghi_series=self._ghi,
                                                          mod_tilt=tilt,
                                                          mod_azi=azi,
                                                          geo_longitude=pvbattery.LONGITUDE,
                                                          geo_latitude=pvbattery.LATITUDE))

        return pvbattery.power_flows(mod_irrads=mod_irrads,
                                     P_l=self._load,
                                     **self.system_params)

    def _set_parameters(self, A):
        """ Set all the parameters that depend on the surface area. """
        self._set_Apv(A)
        self._set_investment(A)
        self._set_Pmax_inv(A)

    def _set_Apv(self, A):
        self.system_params['A_pv'] = A

    def _set_investment(self, A):
        # around 300 euro for a 2m^2 panel. Around 250 euro for fixed cost inverter, around 100 euro per 0.5 kW for inverter.
        self._set_Pmax_inv(A)
        self.price_params['investment'] = 300 * A / 2 + 250 + 100 * self.system_params['P_max_inv'] / 0.5

    def _set_Pmax_inv(self, A):
        # Inverter size in steps of 0.5 kW. 300W panel around 2m^2. Take inverter size equal to peak power, rounded above to nearest 0.5 kW
        self.system_params['P_max_inv'] = 0.5 * np.ceil(A/2 * 0.3 / 0.5)

    def _set_prosumer_tariff(self, A):
        self._set_Pmax_inv(A)
        self.price_params['prosumer tariff'] = 90 * self.system_params['P_max_inv']

    def optimize(self):
        # TODO: write optimization algorithm here
        """
        *   To calculate net present values use the function defined in this class, not the one in pvbattery.py.
            This function keeps into account that we can have multiple orientations (e.g. gable roof east-west)

        *   You can set parameters using e.g. self.system_params['A_pv'] = ...
            To calculate the NPV using update values simply call self.net_present_value() again.

        *   This function should return nothing, it sets the parameters of the object to their optimal values found.

        """
        n = 5
        As = np.linspace(1, 20, n)
        npvs = np.zeros((n,))
        for i, A in enumerate(As):
            self._set_parameters(A)
            npvs[i] = self.net_present_value()

        plt.plot(As, npvs)
        plt.show()


if __name__ == '__main__':
    # example
    case = OptimizationCase()
    case.mod_tilts = [0]
    case.mod_azis = [180]
    case.optimize()

    print(case.system_params)
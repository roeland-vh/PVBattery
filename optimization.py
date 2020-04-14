import pvbattery
import numpy as np
import matplotlib.pyplot as plt


class OptimizationCase(object):
    def __init__(self):
        self._ghi = pvbattery.get_irradiance_data()
        self._load = pvbattery.get_load_data()

        self.price_params = {
            'electricity price': 0.0753,                # fixed
            'yearly electricity price increase': 0,     # fixed
            'price remuneration': 0.0753,               # fixed, same as electricity price for net metering
            'prosumer tariff': 85.49,                   # fixed
            'investment': 0,                            # variable, correlation with A_pv in _set_investment
            'O&M': lambda t: 0,                         # variable, replace inverter after 10 years, see _set_OM
            'distribution tariff': 0.1085,              # fixed
            'transmission tariff': 0.0217,              # fixed
            'taxes & levies': 19.02,                    # fixed
            'salvage value': 0,                         # fixed
        }

        self.system_params = {
            'A_pv': 0,                                  # variable, total surface area of ALL pv modules, distributed evenly over orientations
            'P_max_inv': 0,                             # variable, see _set_Pmax_inv
            'E_batt_max': 0,
            'P_max_batt_charge': 0,
            'P_max_batt_discharge': 0,
            'eta_batt_charge': 0.9,
            'eta_batt_discharge': 0.9,
            'eta_pv': 0.16,                             # fixed
            'eta_inv': 0.972,                           # fixed
        }

        self.misc_params = {
            'study_period': 20,                         # fixed
            'discount_rate': 0.06,                      # fixed
            'cf_other': lambda t: 0                     # fixed, other cash flows
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
        self.system_params['A_pv'] = A
        self.price_params['investment'] = self._get_investment(A)
        self.system_params['P_max_inv'] = self._get_Pmax_inv(A)
        self.price_params['O&M'] = self._get_OM(A)

    def _get_investment(self, A):
        # TODO: mounting and cabling cost, installation cost
        # around €136.5 per m^2 of physical solar cell area
        # assume installation cost of €1000
        # cabling, mounting cost of around € 78 per m^2
        return 136.5 * A + self._get_inv_cost(A) + 1000 + 78 * A

    @staticmethod
    def _get_Pmax_inv(A):
        # Inverter size in steps of 0.5 kW. On average 0.16 kW per m^2.
        # Take inverter size equal to sizing_factor * peak power, rounded above to nearest 0.5 kW
        sizing_factor = 1.0
        return 0.5 * np.ceil(sizing_factor * 0.16 * A / 0.5)

    def _get_inv_cost(self, A):
        # Around 287 euro for fixed cost inverter, around 180 euro per kW for inverter.
        return 287 + 180 * self._get_Pmax_inv(A)

    def _get_OM(self, A):
        """ Calculate O&M cost. Replace inverter after 10 years. """
        return lambda t: (t == 10) * self._get_inv_cost(A)

    def optimize(self):
        """
        *   To calculate net present values use the function defined in this class, not the one in pvbattery.py.
            This function keeps into account that we can have multiple orientations (e.g. gable roof east-west)

        *   You can set parameters using e.g. self.system_params['A_pv'] = ...
            To calculate the NPV using update values simply call self.net_present_value() again.

        *   This function should return nothing, it sets the parameters of the object to their optimal values found.

        """
        n = 10
        As = np.linspace(1, 40, n)
        npvs = np.zeros((n,))
        for i, A in enumerate(As):
            self._set_parameters(A)
            npvs[i] = self.net_present_value()

        i = np.argmax(npvs)
        self._set_parameters(As[i])

        return As, npvs


def run_optim():
    case = OptimizationCase()
    case.mod_azis = [90, 270]
    case.mod_tilts = [10, 10]
    As, npvs = case.optimize()
    baseline = -5672 * np.ones_like(As)

    f, ax = plt.subplots()
    ax.plot(As, npvs, '+')
    ax.plot(As, baseline)
    ax.set_xlabel('PV surface area [m^2]')
    ax.set_ylabel('NPV [€]')
    plt.legend(['PV System', 'Baseline'])

    plt.show()


def run_params():
    case = OptimizationCase()
    case.mod_azis = [180]
    case.mod_tilts = [35]
    case._set_parameters(20)
    print('PRICE PARAMETERS: {}'.format(case.price_params))
    print()
    print('SYSTEM PARAMETERS: {}'.format(case.system_params))


if __name__ == '__main__':
    run_optim()

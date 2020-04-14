import pvbattery
import numpy as np
import matplotlib.pyplot as plt
import itertools


class OptimizationCase(object):
    def __init__(self):
        self._ghi = pvbattery.get_irradiance_data()
        self._load = pvbattery.get_load_data()

        self.price_params = {
            'electricity price': 0.05460 + 0.02472 + 0.00339,   # fixed, elektriciteitsprijs, kosten groene stroom,
                                                                # kosten WKK (zie Engie Direct Fixed 1 jaar prijzenfiche enkelvoudig)
            'yearly electricity price increase': 0,             # fixed
            'prosumer tariff': 85.49,                           # 85.49 for prosumer tariff, 0 for capacity tariff
            'investment': 0,                                    # variable, correlation with A_pv in _set_investment
            'O&M': lambda t: 0,                                 # variable, replace inverter after 10 years, see _set_OM
            'distribution tariff': 0.5*0.1085 + 0.0039251,      # fixed, when using capacity tariff, multiply first number with 0.5 !!! (zie Engie Direct Fixed 1 jaar prijzenfiche)
            'transmission tariff': 0.0210,                      # fixed
            'taxes & levies': 19.02,                            # fixed
            'salvage value': 0,                                 # fixed
            'capacity tariff': 0                               # 0 for prosumer tariff, 33 for capacity tariff
        }

        self.system_params = {
            'A_pv': 0,                                          # variable, total surface area of ALL pv modules, distributed evenly over orientations
            'P_max_inv': 0,                                     # variable, see _set_Pmax_inv
            'E_batt_max': 0,
            'P_max_batt_charge': 0,
            'P_max_batt_discharge': 0,
            'eta_batt_charge': 0.9,
            'eta_batt_discharge': 0.9,
            'eta_pv': 0.16,                                     # fixed, average over a number of panels
            'eta_inv': 0.972,                                   # fixed, representative number
        }

        self.misc_params = {
            'study_period': 20,                                 # fixed
            'discount_rate': 0.06,                              # fixed
            'cf_other': lambda t: 0                             # fixed, other cash flows
        }

        # the tilts and azimuths of the different pv surface
        # e.g. gable roof with east-west orientation mod_azis = [90, 270] and mod_tilt= [x, x]
        self.mod_tilts = []
        self.mod_azis = []

    def set_prosumer_tariff(self):
        self.price_params['prosumer tariff'] = 85.49
        self.price_params['capacity tariff'] = 0

    def set_capacity_tariff(self):
        self.price_params['prosumer tariff'] = 0
        self.price_params['capacity tariff'] = 33

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

    def _set_parameters(self, A, E_batt):
        """ Set all the parameters that depend on the surface area. """
        self.system_params['A_pv'] = A
        self.price_params['investment'] = self._get_investment(A)
        self.system_params['P_max_inv'] = self._get_Pmax_inv(A)
        self.price_params['O&M'] = self._get_OM(A)

        assert E_batt % 3 == 0.0, "Battery size not available!"
        self.system_params['E_batt_max'] = E_batt
        self.system_params['P_max_batt_charge'] = (E_batt / 3) * 1.5            # 1.5 kW per module of 3 kWh
        self.system_params['P_max_batt_discharge'] = (E_batt / 3) * 1.5         # 1.5 kW per module of 3 kWh
        self.price_params['investment'] += 802 * E_batt                         # battery price €802 per kWh
        self.misc_params['cf_other'] = lambda t: -(t == 10) * 802 * E_batt      # replace battery after 10 years

    def _get_investment(self, A):
        # around €136.5 per m^2 of physical solar cell area
        # cabling, mounting and installation cost of around € 78 per m^2
        return 136.5 * A + self._get_inverter_cost(A) + 78 * A

    @staticmethod
    def _get_Pmax_inv(A):
        # Inverter size in steps of 0.5 kW. On average 0.16 kW per m^2.
        # Take inverter size equal to sizing_factor * peak power, rounded above to nearest 0.5 kW
        sizing_factor = 0.8
        return 0.5 * np.ceil(sizing_factor * 0.16 * A / 0.5)

    def _get_inverter_cost(self, A):
        # Around 287 euro for fixed cost inverter, around 180 euro per kW variable cost.
        return 287 + 180 * self._get_Pmax_inv(A)

    def _get_OM(self, A):
        """ Calculate O&M cost. Replace inverter after 10 years. """
        return lambda t: (t == 10) * self._get_inverter_cost(A)

    def optimize(self, optimize_battery=False):
        """ To calculate net present values use the function defined in this class, not the one in pvbattery.py.
        This function keeps into account that we can have multiple orientations (e.g. gable roof east-west)
        This function returns the PV surface areas for which the NPV was calculated and the
        net present values in a dictionary with as keys the battery size and as value an array of the npv's for each
        surface area. """

        if optimize_battery:
            n = 5
            E_batts = np.array([0, 3, 6, 9])
        else:
            n = 10
            E_batts = np.array([0])
        As = np.linspace(1, 40, n)
        npvs = {E_batt: np.zeros((n, )) for E_batt in E_batts}

        npv_max = -1e16
        A_optim = None
        E_batt_optim = None
        for E_batt in E_batts:
            for i, A in enumerate(As):
                self._set_parameters(A, E_batt)
                npv = self.net_present_value()
                npvs[E_batt][i] = npv

                if npv > npv_max:
                    npv_max = npv
                    A_optim = A
                    E_batt_optim = E_batt

        self._set_parameters(A_optim, E_batt_optim)

        return As, npvs


def run_optim():
    """ Runs the optimization and plots the result. """
    case = OptimizationCase()

    ######################################################################
    #################### CHANGE THESE PARAMETERS #########################
    optimize_battery = True
    case.mod_azis = [180]
    case.mod_tilts = [35]
    case.set_capacity_tariff()
    baseline = -6448                # fill in baseline NPV value here for the plot, can be obtained from run_params() with your case.
    #####################################################################
    #####################################################################

    As, npvs_dict = case.optimize(optimize_battery=optimize_battery)

    f, ax = plt.subplots()
    legends = []
    for E_batt, npvs in npvs_dict.items():
        ax.plot(As, npvs, '+')
        legends.append(str(E_batt) + ' kWh')

    ax.plot(As, baseline * np.ones_like(As))
    ax.set_xlabel('PV surface area [m^2]')
    ax.set_ylabel('NPV [€]')
    if optimize_battery:
        plt.legend(legends + ['Baseline'])
    else:
        plt.legend(['PV system', 'Baseline'])

    plt.show()


def run_params():
    """ Run this function to print some parameters of a case you define. """
    
    case = OptimizationCase()
    case.mod_azis = [180]
    case.mod_tilts = [35]
    # case.set_capacity_tariff()
    print('PRICE PARAMETERS: {}'.format(case.price_params))
    print()
    print('SYSTEM PARAMETERS: {}'.format(case.system_params))
    print('NPV: {}'.format(case.net_present_value()))


if __name__ == '__main__':
    run_optim()

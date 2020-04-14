import pysolar
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

# geographical latitude and altidude of our solar installation location in degrees
LATITUDE, LONGITUDE = 50.99461, 5.53972

# solar constant [W / m^2]
SOLAR_CONSTANT = 1361.5

# solar altitude cut-off angle [degrees]
SUN_ALT_CUTOFF = 10

# day tariff hours, varies from supplier to supplier. Usually night tariff is also applicable in the weekends.
DAY_TARIFF_START = 7
DAY_TARIFF_STOP = 22


# helper function
def _angle_between(angle, lower, upper):
    for k in [-1, 0, 1]:
        if lower < angle + 2 * np.pi * k < upper:
            return True
    return False


def calculate_solar_data(ghi):
    df = pd.DataFrame(index=ghi.index, data=np.zeros((len(ghi), 2)), columns=['solar altitude', 'solar azimuth'])
    for i, (time, _) in enumerate(ghi.iteritems()):
        # solar altitude between -90° and 90°, measured from the horizon
        df['solar altitude'][i] = pysolar.solar.get_altitude_fast(latitude_deg=LATITUDE, longitude_deg=LONGITUDE, when=time)
        # solar azimuth between 0° and 360°, measured from the north clockwise
        df['solar azimuth'][i] = pysolar.solar.get_azimuth_fast(latitude_deg=LATITUDE, longitude_deg=LONGITUDE, when=time)

    df.to_pickle('data/solar_angles_data.pkl')


def module_irradiance(ghi_series, mod_tilt, mod_azi, geo_latitude, geo_longitude):
    """ Calculates solar module irradiance given a time series of horizontal irradiance data,
    the tilt and azimuth of the module and the geographical latitude and longitude of the installation.

    :param ghi_series: global horizontal irradiance
    :type ghi_series: pd.Series with datetime objects as indices
    :param mod_tilt: module tilt in degrees
    :param mod_azi: azimuth/orientation in degrees, 0 degrees is north
    :param geo_latitude: latitude of location in degrees, northern hemisphere is positive
    :param geo_longitude: longitude of location in degrees
    :returns: module irradiance for each time specified in param times
    :rtype: numpy array
    """

    solar_angle_data = pd.read_pickle('data/solar_angles_data.pkl')

    mod_tilt = mod_tilt * np.pi / 180
    mod_azi = mod_azi * np.pi / 180

    mod_irrad = np.zeros(shape=(len(ghi_series, )))  # module irradiance
    for i, (time, hor_irrad) in tqdm(enumerate(ghi_series.iteritems())):
        '''
        # solar altitude between -90° and 90°, measured from the horizon
        sun_alt = pysolar.solar.get_altitude_fast(latitude_deg=geo_latitude, longitude_deg=geo_longitude, when=time)
        # solar azimuth between 0° and 360°, measured from the north clockwise
        sun_azi = pysolar.solar.get_azimuth_fast(latitude_deg=geo_latitude, longitude_deg=geo_longitude, when=time)
        '''
        sun_alt = solar_angle_data['solar altitude'][i]
        sun_azi = solar_angle_data['solar azimuth'][i]

        if sun_alt < SUN_ALT_CUTOFF:
            # to account for shading
            mod_irrad[i] = 0
        else:
            sun_alt = sun_alt * np.pi / 180
            sun_azi = sun_azi * np.pi / 180

            # clearness index
            kt = hor_irrad / (SOLAR_CONSTANT * np.sin(sun_alt))

            # diffuse fraction
            kd = np.piecewise(kt, condlist=[lambda x: x < 0.35, lambda x: 0.35 <= x <= 0.75, lambda x: x > 0.75],
                              funclist=[lambda x: 1.0 - 0.249 * x, lambda x: 1.577 - 1.84 * x, lambda x: 0.177])

            # diffuse irradiance
            dif_irrad = hor_irrad * kd * (1 + np.cos(mod_tilt)) / 2

            # direct normal irradiance
            dni = hor_irrad * (1 - kd) / np.sin(sun_alt)

            # geometrical factor
            if not _angle_between(sun_azi, mod_azi - np.pi / 2, mod_azi + np.pi / 2) and (0 < sun_alt < mod_tilt):
                geom_factor = 0
            else:
                geom_factor = np.cos(sun_alt) * np.sin(mod_tilt) * np.cos(mod_azi - sun_azi) + np.sin(sun_alt) * np.cos(
                    mod_tilt)

            # module irradiance
            mod_irrad[i] = dif_irrad + dni * geom_factor

    return pd.Series(data=mod_irrad, index=ghi_series.index)
    # return mod_irrad


def process_irradiance_data():
    data = pd.read_csv('data/irradiance_data.csv', names=['time', 'ghi'], header=0)
    # Watt / m^2 to kiloWatt / m^2
    data['ghi'] = data['ghi'] / 1000

    # TODO: account for winter/summer hour
    tz = datetime.timezone(offset=datetime.timedelta(hours=+1))

    # convert times from string to datetime objects and add time zone
    unaware_dt_objects = data['time'].apply(func=lambda x: datetime.datetime.strptime(x, '%m/%d/%Y %H:%M'))
    aware_dt_objects = unaware_dt_objects.apply(func=lambda x: x.replace(tzinfo=tz))
    data['time'] = aware_dt_objects

    # reduce to 15 minute intervals, each data point gives the average power over the next (!) 15 minutes
    temp = data['time'][0]
    intervals = data['time'].apply(func=lambda x: int(np.floor((x - temp).total_seconds() / (60 * 15))))
    data['interval'] = intervals

    grouped_interval = data[['time', 'interval']].groupby('interval')
    grouped_ghi = data[['ghi', 'interval']].groupby('interval')
    starting_times_interval = grouped_interval.aggregate(min)
    mean_ghi = grouped_ghi.aggregate(np.mean)

    new_data = pd.DataFrame(data=mean_ghi)
    new_data['time'] = starting_times_interval

    new_data.reset_index(inplace=True)
    new_data.drop('interval', axis=1, inplace=True)

    # put in Series object
    new_series = pd.Series(data=new_data['ghi'].to_list(), index=new_data['time'])

    return new_series


def get_irradiance_data():
    data = pd.read_pickle('data/irradiance_data_15min.pkl')
    return data


def process_load_data():
    data = pd.read_csv('data/load_profile.csv', names=['time', 'power'], header=0)
    data = data.dropna()
    tz = datetime.timezone(offset=datetime.timedelta(hours=+1))
    days = 0
    previous_dt = datetime.datetime.strptime(data['time'][0], '%H:%M')
    for i, time in tqdm(enumerate(data['time'])):
        dt = datetime.datetime.strptime(time, '%H:%M')
        if dt < previous_dt:
            days += 1
        previous_dt = dt
        dt = dt.replace(year=2018, month=1, day=1, tzinfo=tz)
        dt = dt + datetime.timedelta(days=days)
        data['time'][i] = dt

    # put in Series object
    series = pd.Series(data=data['power'].tolist(), index=data['time'])

    return series


def get_load_data():
    data = pd.read_pickle('data/load_profile_15min.pkl')
    return data


def power_flows(mod_irrads, P_l, A_pv, P_max_inv, E_batt_max, P_max_batt_charge, P_max_batt_discharge,
                eta_pv, eta_inv, eta_batt_charge, eta_batt_discharge, delta_t=0.25):
    """ Calculates the power flows in the system.

    :param mod_irrads: module irradiance [kW / m^2] !!
    :rtype mod_irrads: pandas Series
    :param P_l: load [kW]
    :rtype P_l: pandas Series
    :param A_pv: total surface area of PV modules, divided equally between different orientations
    :param P_max_inv: maximum power inverter
    :param P_max_batt_charge: maximum battery charging power (before converter) [kW]
    :param P_max_batt_discharge: maximum battery charging power (after converter) [kW]
    :param E_batt_max: capacity of the battery [kWh]
    :param eta_pv: efficiency PV modules
    :param eta_inv: efficiency inverter
    :param eta_batt_charge: charging efficiency battery
    :param eta_batt_discharge: discharging efficiency battery
    :param delta_t: time step [hrs]
    :returns: pandas DataFrame with load power, PV power after inverter, power taken from the grid,
    charging power (before converter), battery level (as a fraction of capacity)
    """

    if not isinstance(mod_irrads, list):
        mod_irrads = [mod_irrads]

    df = pd.DataFrame(data=P_l.data.tolist(), index=P_l.index, columns=['load'])
    for i, mi in enumerate(mod_irrads):
        df['mod irrad_{}'.format(i)] = mi
    # only keep datetimes for which both load and irradiance data is available
    df = df.dropna()

    P_pv = np.zeros(shape=(len(df),))
    P_g = np.zeros(shape=(len(df),))  # power taken from the grid [kW]
    P_b = np.zeros(shape=(len(df),))  # power to battery (before converter) [kW]
    E_b = np.zeros(shape=(len(df),))  # battery level [kWh]
    for i in tqdm(range(len(df))):
        # power generated by PV installation, after inverter
        p_unlimited = np.sum(eta_pv * eta_inv * (A_pv / len(mod_irrads)) * np.sum([df['mod irrad_{}'.format(k)][i] for k, _ in enumerate(mod_irrads)]))
        P_pv[i] = min(p_unlimited, P_max_inv)

        # determine power to/from battery
        if P_pv[i] - df['load'][i] > 0:
            # surplus generation
            if E_b[max(i - 1, 0)] < E_batt_max:
                # battery not full, charge it
                P_b[i] = min(P_pv[i] - df['load'][i], P_max_batt_charge)
            else:
                # battery full
                P_b[i] = 0
        else:
            # pv generation cannot cover load
            if E_b[max(i - 1, 0)] > 0:
                # battery not empty, use some energy from battery
                P_b[i] = max(P_pv[i] - df['load'][i], -P_max_batt_discharge)
            else:
                # battery empty
                P_b[i] = 0

        # power balanced -> power drawn from/supplied to the grid
        P_g[i] = df['load'][i] + P_b[i] - P_pv[i]

        # energy in battery
        E_b[i] = E_b[max(i - 1, 0)] + P_b[i] * (eta_batt_charge if P_b[i] > 0 else 1 / eta_batt_discharge) * delta_t

    df['P PV'] = P_pv
    df['P grid'] = P_g
    df['P battery'] = P_b
    df['charge fraction'] = E_b / E_batt_max

    return df


def plot_mean_load():
    """ Plot the mean load over a year for every hour of the day. """
    data = pd.read_csv('data/load_profile.csv', names=['time', 'power'], header=0)
    data = data.dropna()
    dt_objects = data['time'].apply(func=lambda x: datetime.datetime.strptime(x, '%H:%M'))
    data['time'] = dt_objects
    grouped = data.groupby('time')
    mean = grouped.aggregate(np.mean)
    mean.plot()
    plt.show()


# TODO: validate
# TODO: remuneration mechanism?
def net_present_value(power_flow_grid, price_parameters, inverter_size, study_period, discount_rate,
                      cf_other=lambda _: 0):
    """
    Calculates the net present value (NPV) of the given power flows under the given parameters.

    :param power_flow_grid: power flow [kW] to and from the grid in 15 minute intervals for one year.
    From the grid is positive, to the grid is negative.
    :type power_flow_grid: pandas.Series
    :param price_parameters: dictionary with the following keys:
        * electricity price: price you pay for taking electricity from the grid [€/kWh] (single tariff)
        * yearl electricity price increase: [%]
        * price remuneration: price you get for putting electricity on the grid [€/kWh]
        * prosumer tariff: yearly tariff to pay per kW of your inverter [€/kW]
        * investment: investment done in year 0 [€]
        * O&M: function giving the operation and maintenance cost in year t [€]
        * distribution tariff: tariff to pay each year to the DS0 [€/kWh]
        * transmission tariff: tariff to pay each year to the TSO [€/kWh]
        * taxes & levies: taxes and levies to pay each year [€]
        * salvage value: salvage value at the end of the study period [€]
    :param inverter_size: size of the inverter [kW]
    :param study_period: number of years over which to conduct the study [years]
    :param discount_rate: discount rate
    :param cf_other: function returning other cash flows in year t
    :return: calculated net present value
    """

    # energy put on the grid [kWh]
    E_2grid = -15 / 60 * np.sum(power_flow_grid[power_flow_grid < 0])

    # energy drawn from the grid [kWh]
    E_from_grid = 15/60 * np.sum(power_flow_grid[power_flow_grid > 0])

    # correct for incomplete data. we don't have 365 days of data.
    E_2grid *= 365 / (len(power_flow_grid) / (24 * 4))
    E_from_grid *= 365 / (len(power_flow_grid) / (24 * 4))
    E_net_from_grid = E_from_grid - E_2grid

    """
    E_night = 0  # energy drawn from grid during night tariff
    E_day = 0  # energy drawn from grid during day tariff
    for dt, power in power_flow_grid.iteritems():
        if power > 0:
            if dt.weekday() < 5 and DAY_TARIFF_START <= dt.hour < DAY_TARIFF_STOP:
                E_day += 15 / 60 * power
            else:
                E_night += 15 / 60 * power
    """

    # net present value calculation
    npv = -price_parameters['investment']
    for t in range(1, study_period + 1):
        ncf_t = 0  # net cash flow in year t

        # salvage value
        if t == study_period:
            ncf_t += price_parameters['salvage value']

        # energy component, taken from grid
        ncf_t += -E_from_grid * price_parameters['electricity price'] * (1 + price_parameters['yearly electricity price increase']) ** (t - 1)

        # energy component, put on the grid
        ncf_t += E_2grid * price_parameters['price remuneration']

        # prosumer tariff
        ncf_t += -price_parameters['prosumer tariff'] * inverter_size

        # operation and maintenance cost
        ncf_t += -price_parameters['O&M'](t)

        # distribution and transmission tariffs
        ncf_t += -E_net_from_grid * price_parameters['distribution tariff'] - E_net_from_grid * price_parameters['transmission tariff']

        # taxes and levies
        ncf_t += -price_parameters['taxes & levies']

        # other
        ncf_t += cf_other(t)

        npv += ncf_t / (1 + discount_rate) ** t

    return npv


def plot_avg_irradiance_vs_tilt():
    ghi = get_irradiance_data()
    tilts = np.linspace(start=0, stop=80, num=10)
    azimuths = [0, 90, 180, 270]
    avg_irrads = np.zeros((10,))

    f, ax = plt.subplots()
    for azimuth in azimuths:
        for i, tilt in enumerate(tilts):
            mod_irrads = module_irradiance(ghi_series=ghi, mod_tilt=tilt, mod_azi=azimuth, geo_latitude=LATITUDE, geo_longitude=LONGITUDE)
            avg_irrads[i] = np.mean(mod_irrads) * 365 / (len(mod_irrads) / (24 * 4))    # correct for missing data
        ax.plot(tilts, avg_irrads)

    ax.set_xlabel('Module tilt [°]')
    ax.set_ylabel('Average yearly module irradiance [kW/m^2]')
    plt.legend(['N', 'E', 'S', 'W'])
    plt.show()


def plot_E_grid_vs_tilt(direction='from grid'):
    """ Plot energy taken from or put on the grid in a year [kWh].

    :param direction:
    :type direction: str, either 'from grid' or 'to grid'
    :return:
    """
    ghi = get_irradiance_data()
    P_l = get_load_data()
    tilts = np.linspace(start=0, stop=80, num=10)
    azimuths = [0, 90, 180, 270]
    E_grids = np.zeros((10,))

    f, ax = plt.subplots()
    for azimuth in azimuths:
        for i, tilt in enumerate(tilts):
            mod_irrads = module_irradiance(ghi_series=ghi, mod_tilt=tilt, mod_azi=azimuth, geo_latitude=LATITUDE,
                                           geo_longitude=LONGITUDE)
            df_flows = power_flows(mod_irrads=mod_irrads, P_l=P_l, A_pv=10, P_max_inv=3, E_batt_max=1e-6,
                                   P_max_batt_charge=0, P_max_batt_discharge=0, eta_batt_charge=1,
                                   eta_batt_discharge=1, eta_pv=0.2, eta_inv=0.96)
            power_flow_grid = df_flows['P grid']
            if direction == 'from grid':
                E_grids[i] = 15/60 * np.sum(power_flow_grid[power_flow_grid > 0]) * 365 / (len(power_flow_grid) / (24 * 4))
            else:
                E_grids[i] = -15 / 60 * np.sum(power_flow_grid[power_flow_grid < 0]) * 365 / (
                            len(power_flow_grid) / (24 * 4))
        ax.plot(tilts, E_grids)

    ax.set_xlabel('Module tilt [°]')
    ax.set_ylabel('Yearly kWh drawn from grid')
    plt.legend(['N', 'E', 'S', 'W'])
    plt.show()


def yearly_pv_production(df_flows):
    """ Calculate yearly pv production [kWh] """
    P_pv = df_flows['P PV']
    return 15/60 * np.sum(P_pv) * 365 / (len(P_pv) / (24 * 4))


def monthly_pv_production(df_flows):
    """ Calculates monthly pv production in kWh. """
    P_pv = df_flows['P PV']
    df = pd.DataFrame(data=P_pv.iteritems(), columns=['time', 'P PV'])
    grouped_month = df.groupby(by=lambda i: df['time'][i].month)
    agg_month = grouped_month.aggregate(np.mean)
    mean_power_month = np.array(agg_month['P PV'])

    return 24 * mean_power_month * np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


if __name__ == '__main__':
    # load the irradiance and load data
    ghi = get_irradiance_data()
    P_l = get_load_data()
    # calculate the module irradiance
    mod_irrads = module_irradiance(ghi_series=ghi, mod_tilt=30, mod_azi=180, geo_latitude=LATITUDE,
                                   geo_longitude=LONGITUDE)
    # calculate all power flows in the system
    df_flows = power_flows(mod_irrads=mod_irrads, P_l=P_l, A_pv=10, P_max_inv=3, E_batt_max=4,
                           P_max_batt_charge=0.9, P_max_batt_discharge=0.9, eta_batt_charge=0.9,
                           eta_batt_discharge=0.9, eta_pv=0.2, eta_inv=0.96)

    # visualize power flows
    df_flows.plot()
    plt.legend(['Load [kW]',
                'MI [kW/m2]',
                'PV [kW]',
                'Grid [kW]',
                'Battery [kW]',
                'Charge []'])
    plt.show()


    """
    # net present value calculation
    price_parameters = {
        'electricity price': 0.0614+0.0247+0.0034,
        'yearly electricity price increase': 0.012,
        'price remuneration': 0.0,
        'prosumer tariff': 0,
        'investment': 0,
        'O&M': lambda t: 12 * 2.6,
        'distribution tariff': 278.02,
        'transmission tariff': 54.49,
        'taxes & levies': 19.02,
        'salvage value': 0,
    }

    npv = net_present_value(power_flow_grid=df_flows['P grid'], price_parameters=price_parameters, inverter_size=0,
                            study_period=20, discount_rate=0.07, cf_other=lambda t: -27.35)

    print('\nNPV: €{}\n'.format(npv))
    """
from scipy.stats import kstest, norm, gamma

def Kalmogorov_Smirnov_Grid_Cell(tabsd_wet, mu, sigma, rhiresd_wet, alpha, beta, city_name="City"):
    """
    Prints KS statistics and p-values for temperature (assuming normal distribution) and precipitation 
    (assuming gamma distribution) for the grid cell specified considering their parameters are already available for that grid cell
    """
    # KS test (normal)
    ks_stat_tabsd, p_value_tabsd = kstest(tabsd_wet, "norm", args=(mu, sigma))
    print(f"KS statistic for average temperature on wet days in {city_name} is {ks_stat_tabsd:.3f} with a p-value of {p_value_tabsd:.3f}")

    # KS test (gamma)
    ks_stat_precip, p_value_precip = kstest(rhiresd_wet, "gamma", args=(alpha, 0, beta))  # gamma uses shape, loc, scale
    print(f"KS statistic for total daily precipitation on wet days in {city_name} is {ks_stat_precip:.3f} with a p-value of {p_value_precip:.3f}")

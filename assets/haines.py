import numpy as np

def calculate_dew_point(specific_humidity, pressure):
    """Calculate dew point temperature from specific humidity, temperature, and pressure."""
    mixing_ratio = specific_humidity / (1 - specific_humidity)
    vapor_pressure = (mixing_ratio * pressure) / (0.622 + mixing_ratio)
    dew_point = (243.5 * np.log(vapor_pressure / 6.112)) / (17.67 - np.log(vapor_pressure / 6.112))
    return dew_point

def calculate_haines_index(T500, T700, sh700):
    """Calculate the Haines Index given temperatures and specific humidity."""
    # Calculate dew point at 700 hPa
    Td700 = calculate_dew_point(sh700, T700)
    
    #Stability term (700 - 500)
    S = T700 - T500
    if S <= 18:
        stability_points = 1
    elif 18 < S <= 22:
        stability_points = 2
    else:
        stability_points = 3

    # Moisture term (T700 - Td700)
    M = T700 - Td700
    if M < 15:
        moisture_points = 1
    elif 15 <= M <= 21:
        moisture_points = 2
    else:
        moisture_points = 3

    haines_index = stability_points + moisture_points
    return haines_index
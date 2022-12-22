

#This functions calcualtes the efficiency EEF for the air-source heat pump depending on the outsideTemperature (unit: [°C]) and the modulationDegree (between 0 and 1)
def calculateEfficiency_EEF(outsideTemperature, modulationDegreeForCalculatingTheEfficiency):
    import numpy as np
    maxEEF = 6

    #Technical values for the heat pump  AERO SLM 3-11 HGL when using a cooling temperature (supply temperature) of 18 °C
    y = np.array([3, 3.75, 4.26, 3.45, 4.32, 4.73, 4.11, 4.79, 5.21, 4.76, 5.27, 5.68])
    X = np.array(
        [
            [1, 0.55, 0.22, 1, 0.57, 0.23, 1, 0.64, 0.26, 1, 0.71, 0.28],
            [40, 40, 40, 35, 35, 35, 30, 30, 30, 25, 25, 25],
        ]
    )
    X = X.T  # transpose so input vectors are along the rows
    X = np.c_[X, np.ones(X.shape[0])]  # add bias term
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]

    X_test = np.array([[modulationDegreeForCalculatingTheEfficiency], [outsideTemperature]])
    X_test = X_test.T
    X_test = np.c_[X_test, np.ones(X_test.shape[0])]

    resultingEEF = np.dot(X_test, beta_hat) [0]

    if resultingEEF > maxEEF:
        resultingEEF = maxEEF

    return resultingEEF

#This function calculated the maximum electricalPower of the heat pump (unit: [W]) depending on the outsideTemperature (unit [°C])
def calculateMaximumElectricalPowerOfTheHeatPump(outsideTemperature):
    import numpy as np

    #Technical values for the heat pump  AERO SLM 3-11 HGL when using a cooling temperature (supply temperature) of 18 °C
    maxPower = 3630
    y = np.array([3630, 3180, 2670, 2290, 2050, 1850])
    X = np.array([40, 35, 30, 25, 20, 15])
    X = X.T  # transpose so input vectors are along the rows
    X = np.c_[X, np.ones(X.shape[0])]  # add bias term
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]

    X_test = np.array( [outsideTemperature])
    X_test = X_test.T
    X_test = np.c_[X_test, np.ones(X_test.shape[0])]

    resultingPower = np.dot(X_test, beta_hat) [0]

    if resultingPower > maxPower:
        resultingPower = maxPower

    resultingPower = round(resultingPower, 1)

    return resultingPower


#This method calculated the optimal comfort temperature in a room depending on the outside temperature. Calculation is based on "de Dear, Richard; Brager, G. S.: Developing an adaptive model of thermal comfort and preference "
def calculateOptimalComfortTemperature(outsideTemperature):
    optimalComfortTemperature = round(18.9 + 0.255 * outsideTemperature, 1)

    return optimalComfortTemperature


#This method calculated the optimal comfort temperature in a room depending on the outside temperature. Calculation is based on "the second comfort model (see paper)
def calculateOptimalComfortTemperature_ComfortModel2(szenario, roomNumber, timeslotOfTheDay, timeResolution, optimalComfortTemperatureOffTimes):
    if szenario == 1:
        if timeslotOfTheDay >= (32/(timeResolution/15)) and timeslotOfTheDay < (68/(timeResolution/15)):
            optimalComfortTemperature = (27.4 + 24.8) /2
        else:
            optimalComfortTemperature = optimalComfortTemperatureOffTimes


    elif szenario == 2:
        if roomNumber ==1:
            if timeslotOfTheDay >= (32 / (timeResolution / 15)) and timeslotOfTheDay < (48 / (timeResolution / 15)):
                optimalComfortTemperature = (27.4 + 24.8) /2
            if timeslotOfTheDay >= (48 / (timeResolution / 15)) and timeslotOfTheDay <= (52 / (timeResolution / 15)):
                optimalComfortTemperature = (29.4 + 22.8) /2
            if timeslotOfTheDay >= (52 / (timeResolution / 15)) and timeslotOfTheDay <= (68 / (timeResolution / 15)):
                optimalComfortTemperature = (27.4 + 24.8) /2
            else:
                optimalComfortTemperature =optimalComfortTemperatureOffTimes
        if roomNumber ==2:
            if timeslotOfTheDay >= (32 / (timeResolution / 15)) and timeslotOfTheDay < (48 / (timeResolution / 15)):
                optimalComfortTemperature = (27.4 + 24.8) /2
            if timeslotOfTheDay >= (48 / (timeResolution / 15)) and timeslotOfTheDay <= (52 / (timeResolution / 15)):
                optimalComfortTemperature = (29.4 + 22.8) /2
            if timeslotOfTheDay >= (52 / (timeResolution / 15)) and timeslotOfTheDay <= (68 / (timeResolution / 15)):
                optimalComfortTemperature = (29.4 + 22.8) /2
            else:
                optimalComfortTemperature = optimalComfortTemperatureOffTimes
        if roomNumber ==3:
            if timeslotOfTheDay >= (32 / (timeResolution / 15)) and timeslotOfTheDay < (48 / (timeResolution / 15)):
                optimalComfortTemperature = (29.4 + 22.8) /2
            if timeslotOfTheDay >= (48 / (timeResolution / 15)) and timeslotOfTheDay <= (52 / (timeResolution / 15)):
                optimalComfortTemperature = (29.4 + 22.8) /2
            if timeslotOfTheDay >= (52 / (timeResolution / 15)) and timeslotOfTheDay <= (68 / (timeResolution / 15)):
                optimalComfortTemperature = (29.4 + 22.8) /2
            else:
                optimalComfortTemperature = optimalComfortTemperatureOffTimes
        if roomNumber ==4:
            if timeslotOfTheDay >= (32 / (timeResolution / 15)) and timeslotOfTheDay < (48 / (timeResolution / 15)):
                optimalComfortTemperature = (29.4 + 22.8) /2
            if timeslotOfTheDay >= (48 / (timeResolution / 15)) and timeslotOfTheDay <= (52 / (timeResolution / 15)):
                optimalComfortTemperature = (27.4 + 24.8) /2
            if timeslotOfTheDay >= (52 / (timeResolution / 15)) and timeslotOfTheDay <= (68 / (timeResolution / 15)):
                optimalComfortTemperature = (29.4 + 22.8) /2
            else:
                optimalComfortTemperature = optimalComfortTemperatureOffTimes
        if roomNumber ==5:
            if timeslotOfTheDay >= (32 / (timeResolution / 15)) and timeslotOfTheDay < (48 / (timeResolution / 15)):
                optimalComfortTemperature = (29.4 + 22.8) /2
            if timeslotOfTheDay >= (48 / (timeResolution / 15)) and timeslotOfTheDay <= (52 / (timeResolution / 15)):
                optimalComfortTemperature = (29.4 + 22.8) /2
            if timeslotOfTheDay >= (52 / (timeResolution / 15)) and timeslotOfTheDay <= (68 / (timeResolution / 15)):
                optimalComfortTemperature = (29.4 + 22.8) /2
            else:
                optimalComfortTemperature = optimalComfortTemperatureOffTimes

            #ToDo: Continue (09.11.22): Komfortmodel: aufrufende Funktionen anpassen



    else:
        optimalComfortTemperature = 0
        print("Warning! Wrong szenario number passed into the function calculateOptimalComfortTemperature_ComfortModel2. Should be '1' or '2' as integers")

    return optimalComfortTemperature



#Function for assigning a score for the thermal discomcomfort in a room based on the optimalComfortTemperature and a buffer around the ideal temperature for optimal comfort. Deviations to the limits are linearly penalized
def calculateThermalDisComfort(currentTemperature, optimalComfortTemperature, bufferAroundIdealTemperature ):
    discomfortValue = 0
    if currentTemperature > optimalComfortTemperature + bufferAroundIdealTemperature:
        discomfortValue = currentTemperature - (optimalComfortTemperature + bufferAroundIdealTemperature)
    if currentTemperature < optimalComfortTemperature - bufferAroundIdealTemperature:
        discomfortValue = (optimalComfortTemperature - bufferAroundIdealTemperature) - currentTemperature

    return discomfortValue




#Function for assigning a score for the thermal discomcomfort in a room based on the temperature boundaries and a buffer around the ideal temperature for optimal comfort. Deviations to the limits are linearly penalized
def calculateThermalDisComfort_ComfortModel2(currentRoomTemperature,szenario, roomNumber, timeslotOfTheDay, timeResolution):
    discomfortValue = 0

    if szenario==1:
        if timeslotOfTheDay >= (32/(timeResolution/15)) and timeslotOfTheDay < (68/(timeResolution/15)):
            if currentRoomTemperature > 27.4:
                discomfortValue = currentRoomTemperature - 27.4
            if currentRoomTemperature < 24.8:
                discomfortValue = 24.8 - currentRoomTemperature
        else:
            if currentRoomTemperature > 30:
                discomfortValue = currentRoomTemperature - 30
            if currentRoomTemperature < 16:
                discomfortValue = 16 - currentRoomTemperature

    if szenario ==2:
        if roomNumber ==1:
            if timeslotOfTheDay >= (32 / (timeResolution / 15)) and timeslotOfTheDay < (48 / (timeResolution / 15)):
                if currentRoomTemperature > 27.4:
                    discomfortValue = currentRoomTemperature - 27.4
                if currentRoomTemperature < 24.8:
                    discomfortValue = 24.8 - currentRoomTemperature
            if timeslotOfTheDay >= (48 / (timeResolution / 15)) and timeslotOfTheDay <= (52 / (timeResolution / 15)):
                if currentRoomTemperature > 29.4:
                    discomfortValue = currentRoomTemperature - 29.4
                if currentRoomTemperature < 22.8:
                    discomfortValue = 22.8 - currentRoomTemperature
            if timeslotOfTheDay >= (52 / (timeResolution / 15)) and timeslotOfTheDay <= (68 / (timeResolution / 15)):
                if currentRoomTemperature > 27.4:
                    discomfortValue = currentRoomTemperature - 27.4
                if currentRoomTemperature < 24.8:
                    discomfortValue = 24.8 - currentRoomTemperature
            else:
                if currentRoomTemperature > 30:
                    discomfortValue = currentRoomTemperature - 30
                if currentRoomTemperature < 16:
                    discomfortValue = 16 - currentRoomTemperature
        if roomNumber ==2:
            if timeslotOfTheDay >= (32 / (timeResolution / 15)) and timeslotOfTheDay < (48 / (timeResolution / 15)):
                if currentRoomTemperature > 27.4:
                    discomfortValue = currentRoomTemperature - 27.4
                if currentRoomTemperature < 24.8:
                    discomfortValue = 24.8 - currentRoomTemperature
            if timeslotOfTheDay >= (48 / (timeResolution / 15)) and timeslotOfTheDay <= (52 / (timeResolution / 15)):
                if currentRoomTemperature > 29.4:
                    discomfortValue = currentRoomTemperature - 29.4
                if currentRoomTemperature < 22.8:
                    discomfortValue = 22.8 - currentRoomTemperature
            if timeslotOfTheDay >= (52 / (timeResolution / 15)) and timeslotOfTheDay <= (68 / (timeResolution / 15)):
                if currentRoomTemperature > 29.4:
                    discomfortValue = currentRoomTemperature - 29.4
                if currentRoomTemperature < 22.8:
                    discomfortValue = 22.8 - currentRoomTemperature
            else:
                if currentRoomTemperature > 30:
                    discomfortValue = currentRoomTemperature - 30
                if currentRoomTemperature < 16:
                    discomfortValue = 16 - currentRoomTemperature
        if roomNumber ==3:
            if timeslotOfTheDay >= (32 / (timeResolution / 15)) and timeslotOfTheDay < (48 / (timeResolution / 15)):
                if currentRoomTemperature > 29.4:
                    discomfortValue = currentRoomTemperature - 29.4
                if currentRoomTemperature < 22.8:
                    discomfortValue = 22.8 - currentRoomTemperature
            if timeslotOfTheDay >= (48 / (timeResolution / 15)) and timeslotOfTheDay <= (52 / (timeResolution / 15)):
                if currentRoomTemperature > 29.4:
                    discomfortValue = currentRoomTemperature - 29.4
                if currentRoomTemperature < 22.8:
                    discomfortValue = 22.8 - currentRoomTemperature
            if timeslotOfTheDay >= (52 / (timeResolution / 15)) and timeslotOfTheDay <= (68 / (timeResolution / 15)):
                if currentRoomTemperature > 29.4:
                    discomfortValue = currentRoomTemperature - 29.4
                if currentRoomTemperature < 22.8:
                    discomfortValue = 22.8 - currentRoomTemperature
            else:
                if currentRoomTemperature > 30:
                    discomfortValue = currentRoomTemperature - 30
                if currentRoomTemperature < 16:
                    discomfortValue = 16 - currentRoomTemperature
        if roomNumber ==4:
            if timeslotOfTheDay >= (32 / (timeResolution / 15)) and timeslotOfTheDay < (48 / (timeResolution / 15)):
                if currentRoomTemperature > 29.4:
                    discomfortValue = currentRoomTemperature - 29.4
                if currentRoomTemperature < 22.8:
                    discomfortValue = 22.8 - currentRoomTemperature
            if timeslotOfTheDay >= (48 / (timeResolution / 15)) and timeslotOfTheDay <= (52 / (timeResolution / 15)):
                if currentRoomTemperature > 27.4:
                    discomfortValue = currentRoomTemperature - 27.4
                if currentRoomTemperature < 24.8:
                    discomfortValue = 24.8 - currentRoomTemperature
            if timeslotOfTheDay >= (52 / (timeResolution / 15)) and timeslotOfTheDay <= (68 / (timeResolution / 15)):
                if currentRoomTemperature > 29.4:
                    discomfortValue = currentRoomTemperature - 29.4
                if currentRoomTemperature < 22.8:
                    discomfortValue = 22.8 - currentRoomTemperature
            else:
                if currentRoomTemperature > 30:
                    discomfortValue = currentRoomTemperature - 30
                if currentRoomTemperature < 16:
                    discomfortValue = 16 - currentRoomTemperature
        if roomNumber ==5:
            if timeslotOfTheDay >= (32 / (timeResolution / 15)) and timeslotOfTheDay < (48 / (timeResolution / 15)):
                if currentRoomTemperature > 29.4:
                    discomfortValue = currentRoomTemperature - 29.4
                if currentRoomTemperature < 22.8:
                    discomfortValue = 22.8 - currentRoomTemperature
            if timeslotOfTheDay >= (48 / (timeResolution / 15)) and timeslotOfTheDay <= (52 / (timeResolution / 15)):
                if currentRoomTemperature > 29.4:
                    discomfortValue = currentRoomTemperature - 29.4
                if currentRoomTemperature < 22.8:
                    discomfortValue = 22.8 - currentRoomTemperature
            if timeslotOfTheDay >= (52 / (timeResolution / 15)) and timeslotOfTheDay <= (68 / (timeResolution / 15)):
                if currentRoomTemperature > 29.4:
                    discomfortValue = currentRoomTemperature - 29.4
                if currentRoomTemperature < 22.8:
                    discomfortValue = 22.8 - currentRoomTemperature
            else:
                if currentRoomTemperature > 30:
                    discomfortValue = currentRoomTemperature - 30
                if currentRoomTemperature < 16:
                    discomfortValue = 16 - currentRoomTemperature

    return discomfortValue



if __name__ == "__main__":
    #Test
    maxPower = calculateMaximumElectricalPowerOfTheHeatPump (35)
    print(f"maxPower: {maxPower}")
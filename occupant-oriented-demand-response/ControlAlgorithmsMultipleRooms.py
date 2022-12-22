#This python code contains 2 control algorithms for demand response with a heat pump, desribed in the paper "Occupant-Oriented Demand Response with Room-Individual Building Control":
# - Price Storage Control
# - Hysteresis based two point controller
# Author: Dr.-Ing. Thomas Dengiz
#
# Karlsruhe Institute of Technology (KIT) - IIP (Chair of Energy Economics)
# Tel.: +49-721-608-44678 ‖ E-Mail: thomas.dengiz@kit.edu
# Webpage: https://www.iip.kit.edu/english/86_3459.php

# Import modules
import numpy as np
import pandas as pd
import scipy
import src.examples.control_loop
from random import randrange


# Parameters for the simulation and the control algorithms
timeStepsForUpdatingEmpiricalDistributionFunction = 96 # Unit: [Time steps]
timeResolutionInMinutes = 15 # Unit: [Minutes]
minimalTimeStepsHP_Running = 4 # Unit: [Time steps]
minimalTimeStepsHP_StandBy = 4 # Unit: [Time steps]

#Control parameters of the cooling system
minimumModulationDegree = 0.25 # Unit: [100 %]
minimalBufferStorageSystemForCooling = 0.1  # Unit: [100 %]
maximumElectricalPowerHeatPump = 1000 # Unit: [W]


#Parameters of the water tank (for storage)
maxVolumeWaterTank = 300 # Unit: [l]
initialVolumeWaterTank = 150 # Unit: [l]
temperatureOfTheCoolingWater_Supply = 18  # Unit: [°C]
temperatureOfTheCoolingWater_Return = 24  # Unit: [°C]
specificHeatCapacityOfWater = 4190 # Unit [J/Kg*K]
densitiyOfWater = 1 # Unit: [kg/l]
standingLossesWaterTank = 45  # Unit: [W]
maximumCoolingEnergyContentWaterTank = maxVolumeWaterTank * densitiyOfWater * specificHeatCapacityOfWater * (temperatureOfTheCoolingWater_Return - temperatureOfTheCoolingWater_Supply)




#Changable parameters for the case study
allowedDeviationHelpValue = 1.5  # Range [1.0, 1.5, 2.0]
optimalComfortTemperatureOffTimes = 27 # Range [25, 26, 27]
comfortSzearioNumber = 2 # Range [1, 2]

# Temperatures for the different rooms
allowedDeviationTemperature_Room1 = allowedDeviationHelpValue # Unit: [°C]
allowedDeviationTemperature_Room2 = allowedDeviationHelpValue# Unit: [°C]
allowedDeviationTemperature_Room3 = allowedDeviationHelpValue # Unit: [°C]
allowedDeviationTemperature_Room4 = allowedDeviationHelpValue # Unit: [°C]
allowedDeviationTemperature_Room5 = allowedDeviationHelpValue # Unit: [°C]

temperatureBufferForGettingPerfectScoreForThermalDiscomfort = allowedDeviationHelpValue # Unit: [°C]





#Control Algorithm PSC (Price-Storage-Control) for multiple rooms in one building using a price and a storage factor.
def controlAlgorithm_PriceStorageControl (considerTemperatureRangesInsteadOfSetPoint, sumUpIndividualStoragePercentagesInsteadOfAverage, multiplyFactorsInsteadOfSum, onlyUsePriceFactor, startDate, endDate, pathForTheRun, useAdjustedPriceSignal, considerDiscomfortForTheDistributionOfCoolingEnergy, adjustedModulationDegree):
    #Read input data from pkl file
    import pandas as pd
    from src.model.simulation import loadFMU
    import examples.control_loop as control_loop
    pathToFMU = "./building-model/MultiZone.fmu"


    #Read pkl file with the weather data
    #filename_pkl_data = "./data/data.pkl"
    filename_pkl_data_raw = "./data/data.pkl"
    #resources / data / influx / data.pkl
    df_data_raw = pd.read_pickle(filename_pkl_data_raw)

    df_data_raw.rename(columns={'time': 'timestamp'}, inplace=True)

    #df_data_raw['timestamp'] = df_data_raw.index
    df_data_raw.index = np.arange(0, len(df_data_raw))
    df_data_raw['timestamp'] = pd.to_datetime(df_data_raw['timestamp']).dt.strftime('%d.%m.%Y %H:%M')

    # Choose the relevant weather data for the simulation
    indexStartDate = df_data_raw.index[df_data_raw['timestamp'] == startDate].tolist() [0]
    indexEndDate = df_data_raw.index[df_data_raw['timestamp'] == endDate].tolist()[0]
    df_weatherDataForSimulation = df_data_raw  [:] [indexStartDate: indexEndDate+1]
    df_weatherDataForSimulation.reset_index(drop=True, inplace=True)


    #Read price data from csv file
    df_priceDayAhead = pd.read_csv("./Daten/DSM/Stromtarif_FlexKälte_DayAheadMarkt_2021.csv", sep=';')
    #df_priceTarif1 = pd.read_csv("./Daten/DSM/Stromtarif_FlexKälte_DynamischerTarif1_2021.csv", sep=';')

    #Duplicate the values of the price to have a timestamp for every 15 minutes
    df_priceDayAhead.columns = ['timestamp', 'Price']
    df_priceDayAhead['timestamp'] = pd.to_datetime(df_priceDayAhead['timestamp'], dayfirst=True)
    df_priceDayAhead = df_priceDayAhead.set_index('timestamp').asfreq('15T', method='ffill').reset_index()
    df_priceDayAhead['timestamp'] = pd.to_datetime(df_priceDayAhead['timestamp']).dt.strftime('%d.%m.%Y %H:%M')

    #Choose the relevant price data for the simulation
    indexStartDatePrice = df_priceDayAhead.index[df_priceDayAhead['timestamp'] == startDate].tolist()[0]
    indexEndDatePrice = df_priceDayAhead.index[df_priceDayAhead['timestamp'] == endDate].tolist()[0]
    df_priceDataForSimulation  = df_priceDayAhead [indexStartDatePrice: indexEndDatePrice+1]
    df_priceDataForSimulation.reset_index(drop=True, inplace=True)

    optimizationHorizon = len(df_weatherDataForSimulation)

    #Initialize the output arrays of the results
    output_timeStamp = ["" for x in range(optimizationHorizon)]
    output_electricityCostsPerTimeSlot = np.zeros(optimizationHorizon)
    output_electricityTarifPerTimeSlot = np.zeros(optimizationHorizon)
    output_electricalLoadHeatPumpPerTimeSlot = np.zeros(optimizationHorizon)
    output_temperatureRoom1PerTimeSlot = np.zeros(optimizationHorizon)
    output_temperatureRoom2PerTimeSlot = np.zeros(optimizationHorizon)
    output_temperatureRoom3PerTimeSlot = np.zeros(optimizationHorizon)
    output_temperatureRoom4PerTimeSlot = np.zeros(optimizationHorizon)
    output_temperatureRoom5PerTimeSlot = np.zeros(optimizationHorizon)

    output_outsideTemperaturePerTimeSlot = np.zeros(optimizationHorizon)
    output_maxElectricalPowerPerTimeSlot = np.zeros(optimizationHorizon)
    output_optimalComfortTemperaturePerTimeSlot = np.zeros(optimizationHorizon)

    output_storagePercentageRoom1PerTimeSlot = np.zeros(optimizationHorizon)
    output_storagePercentageRoom2PerTimeSlot = np.zeros(optimizationHorizon)
    output_storagePercentageRoom3PerTimeSlot = np.zeros(optimizationHorizon)
    output_storagePercentageRoom4PerTimeSlot = np.zeros(optimizationHorizon)
    output_storagePercentageRoom5PerTimeSlot = np.zeros(optimizationHorizon)
    output_storagePercentagCombinedPerTimeSlot = np.zeros(optimizationHorizon)
    output_storagePercentagCombinedPerTimeSlotAverage = np.zeros(optimizationHorizon)

    output_coolingPowerRoom1PerTimeSlot = np.zeros(optimizationHorizon)
    output_coolingPowerRoom2PerTimeSlot = np.zeros(optimizationHorizon)
    output_coolingPowerRoom3PerTimeSlot = np.zeros(optimizationHorizon)
    output_coolingPowerRoom4PerTimeSlot = np.zeros(optimizationHorizon)
    output_coolingPowerRoom5PerTimeSlot = np.zeros(optimizationHorizon)

    output_coolingEfficiencyEERPerTimeSlot = np.zeros(optimizationHorizon)
    output_priceFactorPerTimeSlot = np.zeros(optimizationHorizon)
    output_storageFactorPerTimeSlot = np.zeros(optimizationHorizon)
    output_modulationDegreeHeatPumpPerTimeSlot = np.zeros(optimizationHorizon)
    output_scoreThermalDiscomfortPerTimeSlot = np.zeros(optimizationHorizon)

    output_idealTemperature = np.zeros(optimizationHorizon)
    output_maximumPowerHeatPump = np.zeros(optimizationHorizon)


    currentNumberTimeStepsHP_Running = 0
    currentNumberTimeStepsHP_StandBy = 0
    helpCounterTimeSlotsForUpdatingEDF = 0
    thermalDiscomfort_HighTemperature_lastTimeSlot =0

    with src.model.fmu.yieldFMU(pathToFMU) as fmu:
        #Initialize room temperatures at the beginning
        initalOutsideTemperature = df_weatherDataForSimulation.loc[0, 'Ta']
        initialTemperature_Room1 = 26.1
        initialTemperature_Room2 = 26.1
        initialTemperature_Room3 = 26.1
        initialTemperature_Room4 = 26.1
        initialTemperature_Room5 = 26.1

        control_loop.setStartingTemperatures(fmu, initialTemperature_Room1, initialTemperature_Room2, initialTemperature_Room3 ,initialTemperature_Room4, initialTemperature_Room5)

        #Loop over all timeslots
        for index_timeslot in range (0, optimizationHorizon):

            outsideTemperature = df_weatherDataForSimulation.loc[index_timeslot, 'Ta']
            solarRadiation = df_weatherDataForSimulation.loc[index_timeslot, 'phis']
            timeStamp  = df_weatherDataForSimulation.loc [index_timeslot, 'timestamp']
            output_timeStamp[index_timeslot] = str(timeStamp)
            helpCounterTimeSlotsForUpdatingEDF +=1


            #Function call for optimal comfort temeprature and maximum power of the heat pump
            idealTemperature_Room1 = HelpFunctions.calculateOptimalComfortTemperature_ComfortModel2(comfortSzearioNumber, 1, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes, optimalComfortTemperatureOffTimes)
            idealTemperature_Room2 = HelpFunctions.calculateOptimalComfortTemperature_ComfortModel2(comfortSzearioNumber, 2, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes, optimalComfortTemperatureOffTimes)
            idealTemperature_Room3 = HelpFunctions.calculateOptimalComfortTemperature_ComfortModel2(comfortSzearioNumber, 3, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes, optimalComfortTemperatureOffTimes)
            idealTemperature_Room4 = HelpFunctions.calculateOptimalComfortTemperature_ComfortModel2(comfortSzearioNumber, 4, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes, optimalComfortTemperatureOffTimes)
            idealTemperature_Room5 = HelpFunctions.calculateOptimalComfortTemperature_ComfortModel2(comfortSzearioNumber, 5, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes, optimalComfortTemperatureOffTimes)
            maximumElectricalPowerHeatPump = HelpFunctions.calculateMaximumElectricalPowerOfTheHeatPump(outsideTemperature)
            output_idealTemperature [index_timeslot] = idealTemperature_Room1
            output_maximumPowerHeatPump[index_timeslot] = maximumElectricalPowerHeatPump

            # Update the electricity prices for the next day and the corresponding empirical distribution function at the beginning of each day
            if helpCounterTimeSlotsForUpdatingEDF ==timeStepsForUpdatingEmpiricalDistributionFunction or index_timeslot==0:
                # Calculate empirial cumulative distribution function (ECDF) for the future prices
                from statsmodels.distributions.empirical_distribution import ECDF
                electricityTarifCurrentDay = df_priceDataForSimulation.loc [index_timeslot: index_timeslot + 96 - 1, 'Price'].values
                adjustedPriceSignal = np.zeros(timeStepsForUpdatingEmpiricalDistributionFunction)
                #Calculate the EEF values for the next day in advance
                future_eef_values = np.zeros(96)
                for i in range(0, len(electricityTarifCurrentDay)):
                    future_eef_values [i] = HelpFunctions.calculateEfficiency_EEF(df_data_raw.loc[index_timeslot + i, 'Ta'], 0.5)
                    adjustedPriceSignal [i] = electricityTarifCurrentDay [i] / future_eef_values [i]


                if useAdjustedPriceSignal==True:
                    inputSignalPrice = adjustedPriceSignal
                else:
                    inputSignalPrice = electricityTarifCurrentDay
                ecdf_prices = ECDF(inputSignalPrice)
                helpCounterTimeSlotsForUpdatingEDF =0




            #Get room temperatures from the previous time slot
            if index_timeslot ==0:
                temperature_lastTimeslot_Room1 = idealTemperature_Room1
                temperature_lastTimeslot_Room2 = idealTemperature_Room2
                temperature_lastTimeslot_Room3 = idealTemperature_Room3
                temperature_lastTimeslot_Room4 = idealTemperature_Room4
                temperature_lastTimeslot_Room5 = idealTemperature_Room5


            else:
                temperature_lastTimeslot_Room1 = output_temperatureRoom1PerTimeSlot [index_timeslot - 1]
                temperature_lastTimeslot_Room2 = output_temperatureRoom2PerTimeSlot [index_timeslot - 1]
                temperature_lastTimeslot_Room3 = output_temperatureRoom3PerTimeSlot [index_timeslot - 1]
                temperature_lastTimeslot_Room4 = output_temperatureRoom4PerTimeSlot [index_timeslot - 1]
                temperature_lastTimeslot_Room5 = output_temperatureRoom5PerTimeSlot [index_timeslot - 1]





            #Calculate deviation to temperature limits by considering the ideal temperature setpoint

            #Room1
            temperature_distanceToUpperLimit_Room1 = idealTemperature_Room1 + allowedDeviationTemperature_Room1 - temperature_lastTimeslot_Room1

            if temperature_lastTimeslot_Room1 <= idealTemperature_Room1:
                storagePercentage_Room1 = 1.0
            else:
                storagePercentage_Room1 = temperature_distanceToUpperLimit_Room1 / allowedDeviationTemperature_Room1
                if temperature_distanceToUpperLimit_Room1 <= 0:
                    storagePercentage_Room1 = 0


            # Room2
            temperature_distanceToUpperLimit_Room2 = idealTemperature_Room2 + allowedDeviationTemperature_Room2 - temperature_lastTimeslot_Room2

            if temperature_lastTimeslot_Room2 <= idealTemperature_Room2:
                storagePercentage_Room2 = 1.0
            else:
                storagePercentage_Room2 = temperature_distanceToUpperLimit_Room2 / allowedDeviationTemperature_Room2
                if temperature_distanceToUpperLimit_Room2 <= 0:
                    storagePercentage_Room2 = 0


            # Room3
            temperature_distanceToUpperLimit_Room3 = idealTemperature_Room3 + allowedDeviationTemperature_Room3 - temperature_lastTimeslot_Room3

            if temperature_lastTimeslot_Room3 <= idealTemperature_Room3:
                storagePercentage_Room3 = 1.0
            else:
                storagePercentage_Room3 = temperature_distanceToUpperLimit_Room3 / allowedDeviationTemperature_Room3
                if temperature_distanceToUpperLimit_Room3 <= 0:
                    storagePercentage_Room3 = 0



            # Room4
            temperature_distanceToUpperLimit_Room4 = idealTemperature_Room4 + allowedDeviationTemperature_Room4 - temperature_lastTimeslot_Room4

            if temperature_lastTimeslot_Room4 <= idealTemperature_Room4:
                storagePercentage_Room4 = 1.0
            else:
                storagePercentage_Room4 = temperature_distanceToUpperLimit_Room4 / allowedDeviationTemperature_Room4
                if temperature_distanceToUpperLimit_Room4 <= 0:
                    storagePercentage_Room4 = 0



            # Room5
            temperature_distanceToUpperLimit_Room5 = idealTemperature_Room5 + allowedDeviationTemperature_Room5 - temperature_lastTimeslot_Room5

            if temperature_lastTimeslot_Room5 <= idealTemperature_Room5:
                storagePercentage_Room5 = 1.0
            else:
                storagePercentage_Room5 = temperature_distanceToUpperLimit_Room5 / allowedDeviationTemperature_Room5
                if temperature_distanceToUpperLimit_Room5 <= 0:
                    storagePercentage_Room5 = 0



            #Calculate deviation to temperature limits by considering the full range of the temperature limits (storage_Percentage_fullRange=1 if temperature at lower limit and 0 at upper limit)

            # Room1
            storagePercentage_fullRange_Room1 =1 -( temperature_lastTimeslot_Room1 - (idealTemperature_Room1 - allowedDeviationTemperature_Room1)) / ((idealTemperature_Room1 + allowedDeviationTemperature_Room1)- (idealTemperature_Room1 - allowedDeviationTemperature_Room1))
            if storagePercentage_fullRange_Room1 < 0:
                storagePercentage_fullRange_Room1 = 0
            if storagePercentage_fullRange_Room1 > 1:
                storagePercentage_fullRange_Room1 = 1


            # Room2
            storagePercentage_fullRange_Room2 =1 -( temperature_lastTimeslot_Room2 - (idealTemperature_Room2 - allowedDeviationTemperature_Room2)) / ((idealTemperature_Room2 + allowedDeviationTemperature_Room2)- (idealTemperature_Room2 - allowedDeviationTemperature_Room2))
            if storagePercentage_fullRange_Room2 < 0:
                storagePercentage_fullRange_Room2 = 0
            if storagePercentage_fullRange_Room2 > 1:
                storagePercentage_fullRange_Room2 = 1


            # Room3
            storagePercentage_fullRange_Room3 =1 -( temperature_lastTimeslot_Room3 - (idealTemperature_Room3 - allowedDeviationTemperature_Room3)) / ((idealTemperature_Room3 + allowedDeviationTemperature_Room3)- (idealTemperature_Room3 - allowedDeviationTemperature_Room3))
            if storagePercentage_fullRange_Room3 < 0:
                storagePercentage_fullRange_Room3 = 0
            if storagePercentage_fullRange_Room3 > 1:
                storagePercentage_fullRange_Room3 = 1


             # Room4
            storagePercentage_fullRange_Room4 =1 -( temperature_lastTimeslot_Room4 - (idealTemperature_Room4 - allowedDeviationTemperature_Room4)) / ((idealTemperature_Room4 + allowedDeviationTemperature_Room4)- (idealTemperature_Room4 - allowedDeviationTemperature_Room4))
            if storagePercentage_fullRange_Room4 < 0:
                storagePercentage_fullRange_Room4 = 0
            if storagePercentage_fullRange_Room4 > 1:
                storagePercentage_fullRange_Room4 = 1


            # Room5
            storagePercentage_fullRange_Room5 =1 - ( temperature_lastTimeslot_Room5 - (idealTemperature_Room5 - allowedDeviationTemperature_Room5)) / ((idealTemperature_Room5 + allowedDeviationTemperature_Room5)- (idealTemperature_Room5 - allowedDeviationTemperature_Room5))
            if storagePercentage_fullRange_Room5 < 0:
                storagePercentage_fullRange_Room5 = 0
            if storagePercentage_fullRange_Room5 > 1:
                storagePercentage_fullRange_Room5 = 1

            if considerTemperatureRangesInsteadOfSetPoint == True:
                storagePercentage_Room1 = storagePercentage_fullRange_Room1
                storagePercentage_Room2 = storagePercentage_fullRange_Room2
                storagePercentage_Room3 = storagePercentage_fullRange_Room3
                storagePercentage_Room4 = storagePercentage_fullRange_Room4
                storagePercentage_Room5 = storagePercentage_fullRange_Room5


            #Calculate the Price Factor
            priceFactor = round(1 - ecdf_prices (inputSignalPrice [helpCounterTimeSlotsForUpdatingEDF]- 1e-3),2)

            # Calculate the combined Storage Factor (storageFactor = 1 if temperaures are at upper limit and 0 at lower limit)
            if sumUpIndividualStoragePercentagesInsteadOfAverage == True:
                storageFactor_Combined = 1 - (storagePercentage_Room1 + storagePercentage_Room2 + storagePercentage_Room3 + storagePercentage_Room4 + storagePercentage_Room5)
            else:
                storageFactor_Combined = 1- ((storagePercentage_Room1 + storagePercentage_Room2 + storagePercentage_Room3 + storagePercentage_Room4 + storagePercentage_Room5)/ 5)


            #Calculate combined storage percentage (storagePercentage_fullRange_combined =1 if temperatures at lower limit and 0 at upper limit)
            storagePercentage_fullRange_combined = (storagePercentage_fullRange_Room1 + storagePercentage_fullRange_Room2 + storagePercentage_fullRange_Room3 + storagePercentage_fullRange_Room4 + storagePercentage_fullRange_Room5)

            storagePercentage_fullRange_combined_average = storagePercentage_fullRange_combined/5


            if storageFactor_Combined >5:
                storageFactor_Combined = 5.0
            if storageFactor_Combined <0:
                storageFactor_Combined = 0

            output_storagePercentageRoom1PerTimeSlot [index_timeslot] = round(storagePercentage_Room1, 2)
            output_storagePercentageRoom2PerTimeSlot[index_timeslot] = round(storagePercentage_Room2, 2)
            output_storagePercentageRoom3PerTimeSlot[index_timeslot] = round(storagePercentage_Room3, 2)
            output_storagePercentageRoom4PerTimeSlot[index_timeslot] = round(storagePercentage_Room4, 2)
            output_storagePercentageRoom5PerTimeSlot[index_timeslot] = round(storagePercentage_Room5, 2)





            #Derive the desired modulation degree of the heat pump
            if multiplyFactorsInsteadOfSum == True:
                desiredModulationDegreeHeatPump = storageFactor_Combined * priceFactor
            else:
                weight_storageFactor = 0.5
                weight_priceFactor = 0.5
                desiredModulationDegreeHeatPump = storageFactor_Combined * weight_storageFactor + priceFactor * weight_priceFactor


            if onlyUsePriceFactor==True:
                desiredModulationDegreeHeatPump = priceFactor

            #Adust modulation degree if average storage percentage is small
            if adjustedModulationDegree ==True:
                #Adjust desiredModulaitonDegree if average storage is too low
                if storagePercentage_fullRange_combined_average < 0.3 and desiredModulationDegreeHeatPump < minimumModulationDegree:
                    desiredModulationDegreeHeatPump = minimumModulationDegree

                #Adjust desiredModulationDegree if there is too high discomfort because of too high temperatures
                if thermalDiscomfort_HighTemperature_lastTimeSlot > 1 and desiredModulationDegreeHeatPump < minimumModulationDegree:
                    desiredModulationDegreeHeatPump = minimumModulationDegree
                if thermalDiscomfort_HighTemperature_lastTimeSlot > 1 and desiredModulationDegreeHeatPump >= minimumModulationDegree:
                    desiredModulationDegreeHeatPump = desiredModulationDegreeHeatPump + 0.1

                if storagePercentage_fullRange_combined_average <0.05:
                    desiredModulationDegreeHeatPump = desiredModulationDegreeHeatPump + 0.1

            #Check for minimum and maximum modulation degree
            if desiredModulationDegreeHeatPump < minimumModulationDegree:
                desiredModulationDegreeHeatPump = 0

            if desiredModulationDegreeHeatPump > 1:
                desiredModulationDegreeHeatPump = 1



            #Distribute the cooling energy among the rooms
            freeCapacity_storageP_fullRange_combined = (1-storagePercentage_fullRange_Room1) + (1-storagePercentage_fullRange_Room2) + (1-storagePercentage_fullRange_Room3) + (1- storagePercentage_fullRange_Room4) + (1-storagePercentage_fullRange_Room5)

            #Stop cooling if the storage is full (temperatur too cold)
            if storagePercentage_fullRange_combined >5 - 5* minimalBufferStorageSystemForCooling and coolingEnergyOfTheHeatPump > 0.01:
                desiredModulationDegreeHeatPump =0

            #Force cooling with full power if the storage is empty (temperatur too warm)
            if storagePercentage_fullRange_combined < minimalBufferStorageSystemForCooling and coolingEnergyOfTheHeatPump < 0.01:
                desiredModulationDegreeHeatPump = 1


            #Consider minimal runtime of HP
            if desiredModulationDegreeHeatPump < 0.01 and currentNumberTimeStepsHP_Running >=1 and currentNumberTimeStepsHP_Running < minimalTimeStepsHP_Running:
                desiredModulationDegreeHeatPump = minimumModulationDegree

            #Consider minimal standby time of HP
            if desiredModulationDegreeHeatPump > 0.01 and currentNumberTimeStepsHP_StandBy >=1 and currentNumberTimeStepsHP_StandBy < minimalTimeStepsHP_StandBy:
                desiredModulationDegreeHeatPump =0


            # Calculate the cooling energy of the heat pump if the desired modulation degree is used and
            EEF_currentTimeSlot = HelpFunctions.calculateEfficiency_EEF(outsideTemperature, desiredModulationDegreeHeatPump)
            coolingEnergyOfTheHeatPump = desiredModulationDegreeHeatPump * timeResolutionInMinutes * EEF_currentTimeSlot * maximumElectricalPowerHeatPump * 60

            #Assign the cooling load to the different rooms if the combined storage percentage is below a threshold value (e.g. 90%)
            if storagePercentage_fullRange_combined !=0:
                storageShareRoom1 = (storagePercentage_fullRange_Room1)/ (storagePercentage_fullRange_combined )
                storageShareRoom2 = (storagePercentage_fullRange_Room2)/ (storagePercentage_fullRange_combined )
                storageShareRoom3 = (storagePercentage_fullRange_Room3)/ (storagePercentage_fullRange_combined )
                storageShareRoom4 = (storagePercentage_fullRange_Room4)/ (storagePercentage_fullRange_combined )
                storageShareRoom5 = (storagePercentage_fullRange_Room5)/ (storagePercentage_fullRange_combined )

            else:
                storageShareRoom1 = 1/5
                storageShareRoom2 = 1/5
                storageShareRoom3 = 1/5
                storageShareRoom4 = 1/5
                storageShareRoom5 = 1/5



            storageShare_OppositeValue_Room1 = 1 - storageShareRoom1
            storageShare_OppositeValue_Room2 = 1 - storageShareRoom2
            storageShare_OppositeValue_Room3 = 1 - storageShareRoom3
            storageShare_OppositeValue_Room4 = 1 - storageShareRoom4
            storageShare_OppositeValue_Room5 = 1 - storageShareRoom5
            sumStorageShares_OppositeValue = storageShare_OppositeValue_Room1 + storageShare_OppositeValue_Room2 + storageShare_OppositeValue_Room3 + storageShare_OppositeValue_Room4 + storageShare_OppositeValue_Room5

            coolingEnergy_Room1 = (storageShare_OppositeValue_Room1/sumStorageShares_OppositeValue) * coolingEnergyOfTheHeatPump
            coolingEnergy_Room2 = (storageShare_OppositeValue_Room2/sumStorageShares_OppositeValue) * coolingEnergyOfTheHeatPump
            coolingEnergy_Room3 = (storageShare_OppositeValue_Room3/sumStorageShares_OppositeValue) * coolingEnergyOfTheHeatPump
            coolingEnergy_Room4 = (storageShare_OppositeValue_Room4/sumStorageShares_OppositeValue) * coolingEnergyOfTheHeatPump
            coolingEnergy_Room5 = (storageShare_OppositeValue_Room5/sumStorageShares_OppositeValue) * coolingEnergyOfTheHeatPump


            #Assign the share of cooling energy based on the discomfort score
            if index_timeslot >0 and considerDiscomfortForTheDistributionOfCoolingEnergy == True:
                if output_scoreThermalDiscomfortPerTimeSlot [index_timeslot -1] > 0:
                    #Don't cool rooms that are already too cold
                    if discomfort_Room1_lastTimeslot_LowTemperature >0:
                        coolingEnergy_Room1 =0
                    if discomfort_Room2_lastTimeslot_LowTemperature > 0:
                        coolingEnergy_Room2 = 0
                    if discomfort_Room3_lastTimeslot_LowTemperature > 0:
                        coolingEnergy_Room3 = 0
                    if discomfort_Room4_lastTimeslot_LowTemperature > 0:
                        coolingEnergy_Room4 = 0
                    if discomfort_Room5_lastTimeslot_LowTemperature > 0:
                        coolingEnergy_Room5 = 0
                    if coolingEnergy_Room1 ==0 and coolingEnergy_Room2 ==0 and coolingEnergy_Room3 ==0 and coolingEnergy_Room4 ==0 and coolingEnergy_Room5 ==0:
                        desiredModulationDegreeHeatPump = 0
                        coolingEnergyOfTheHeatPump =0

                    #Calculate total discomfort because of too high temperatures
                    totalDiscomfort_lastTimeSlot_HighTemperature = discomfort_Room1_lastTimeslot_HighTemperature + discomfort_Room2_lastTimeslot_HighTemperature + discomfort_Room3_lastTimeslot_HighTemperature + discomfort_Room4_lastTimeslot_HighTemperature + discomfort_Room5_lastTimeslot_HighTemperature

                    #Calculate the share of each room for causing thermal discomfort because of too high temperatures
                    if totalDiscomfort_lastTimeSlot_HighTemperature >0:
                        shareOfRoom1ForThermalDiscomfort_LastTimeSlot = discomfort_Room1_lastTimeslot_HighTemperature /totalDiscomfort_lastTimeSlot_HighTemperature
                        shareOfRoom2ForThermalDiscomfort_LastTimeSlot = discomfort_Room2_lastTimeslot_HighTemperature / totalDiscomfort_lastTimeSlot_HighTemperature
                        shareOfRoom3ForThermalDiscomfort_LastTimeSlot = discomfort_Room3_lastTimeslot_HighTemperature / totalDiscomfort_lastTimeSlot_HighTemperature
                        shareOfRoom4ForThermalDiscomfort_LastTimeSlot = discomfort_Room4_lastTimeslot_HighTemperature / totalDiscomfort_lastTimeSlot_HighTemperature
                        shareOfRoom5ForThermalDiscomfort_LastTimeSlot = discomfort_Room5_lastTimeslot_HighTemperature / totalDiscomfort_lastTimeSlot_HighTemperature

                        #Assign cooling energy to the rooms based on discomfort
                        coolingEnergy_Room1 = shareOfRoom1ForThermalDiscomfort_LastTimeSlot * coolingEnergyOfTheHeatPump
                        coolingEnergy_Room2 = shareOfRoom2ForThermalDiscomfort_LastTimeSlot * coolingEnergyOfTheHeatPump
                        coolingEnergy_Room3 = shareOfRoom3ForThermalDiscomfort_LastTimeSlot * coolingEnergyOfTheHeatPump
                        coolingEnergy_Room4 = shareOfRoom4ForThermalDiscomfort_LastTimeSlot * coolingEnergyOfTheHeatPump
                        coolingEnergy_Room5 = shareOfRoom5ForThermalDiscomfort_LastTimeSlot * coolingEnergyOfTheHeatPump



            #Adjust counters for number of running and standby steps of the HP
            if coolingEnergyOfTheHeatPump == 0:
                currentNumberTimeStepsHP_StandBy += 1
                currentNumberTimeStepsHP_Running = 0

            if coolingEnergyOfTheHeatPump > 0.1:
                currentNumberTimeStepsHP_Running += 1
                currentNumberTimeStepsHP_StandBy = 0


            #Calculate the resulting output values for this timeslot
            output_electricalLoadHeatPumpPerTimeSlot [index_timeslot] = desiredModulationDegreeHeatPump * maximumElectricalPowerHeatPump
            output_electricityCostsPerTimeSlot [index_timeslot] = output_electricalLoadHeatPumpPerTimeSlot [index_timeslot] * timeResolutionInMinutes * 60 * (electricityTarifCurrentDay [helpCounterTimeSlotsForUpdatingEDF]/3600000)
            output_electricityTarifPerTimeSlot [index_timeslot] = electricityTarifCurrentDay [helpCounterTimeSlotsForUpdatingEDF]




            #Assign temperature values to the variables
            if index_timeslot ==0:
                temperatureRoom1 = initialTemperature_Room1
                temperatureRoom2 = initialTemperature_Room2
                temperatureRoom3 = initialTemperature_Room3
                temperatureRoom4 = initialTemperature_Room4
                temperatureRoom5 = initialTemperature_Room5

            if index_timeslot >0:
                temperatureRoom1 = output_temperatureRoom1PerTimeSlot[index_timeslot - 1]
                temperatureRoom2 = output_temperatureRoom2PerTimeSlot[index_timeslot - 1]
                temperatureRoom3 = output_temperatureRoom3PerTimeSlot[index_timeslot - 1]
                temperatureRoom4 = output_temperatureRoom4PerTimeSlot[index_timeslot - 1]
                temperatureRoom5 = output_temperatureRoom5PerTimeSlot[index_timeslot - 1]


            # Assign the cooling variables of the model and calculate new temperatures
            hvacRoom1 = (coolingEnergy_Room1 * (-1))/(timeResolutionInMinutes * 60)
            hvacRoom2 = (coolingEnergy_Room2 * (-1))/(timeResolutionInMinutes * 60)
            hvacRoom3 = (coolingEnergy_Room3 * (-1))/(timeResolutionInMinutes * 60)
            hvacRoom4 = (coolingEnergy_Room4 * (-1))/(timeResolutionInMinutes * 60)
            hvacRoom5 = (coolingEnergy_Room5 * (-1))/(timeResolutionInMinutes * 60)

            output_coolingPowerRoom1PerTimeSlot [index_timeslot] = hvacRoom1 * (-1)
            output_coolingPowerRoom2PerTimeSlot [index_timeslot] = hvacRoom2 * (-1)
            output_coolingPowerRoom3PerTimeSlot [index_timeslot] = hvacRoom3 * (-1)
            output_coolingPowerRoom4PerTimeSlot [index_timeslot] = hvacRoom4 * (-1)
            output_coolingPowerRoom5PerTimeSlot [index_timeslot] = hvacRoom5 * (-1)

            new_temperatureRoom1, new_temperatureRoom2, new_temperatureRoom3, new_temperatureRoom4, new_temperatureRoom5 = control_loop.getTemperatures(fmu, outsideTemperature, solarRadiation, temperatureRoom1, temperatureRoom2, temperatureRoom3, temperatureRoom4, temperatureRoom5, hvacRoom1, hvacRoom2, hvacRoom3, hvacRoom4, hvacRoom5)

            output_temperatureRoom1PerTimeSlot [index_timeslot] = round(new_temperatureRoom1,1)
            output_temperatureRoom2PerTimeSlot[index_timeslot] = round(new_temperatureRoom2,1)
            output_temperatureRoom3PerTimeSlot[index_timeslot] = round(new_temperatureRoom3,1)
            output_temperatureRoom4PerTimeSlot[index_timeslot] = round(new_temperatureRoom4,1)
            output_temperatureRoom5PerTimeSlot[index_timeslot] = round(new_temperatureRoom5,1)


            output_coolingEfficiencyEERPerTimeSlot [index_timeslot] = round(EEF_currentTimeSlot,2)
            output_priceFactorPerTimeSlot  [index_timeslot] = round(priceFactor,2)
            output_storageFactorPerTimeSlot [index_timeslot] = round(storageFactor_Combined,2)
            output_storagePercentagCombinedPerTimeSlot [index_timeslot] = round(storagePercentage_fullRange_combined,2)
            output_modulationDegreeHeatPumpPerTimeSlot  [index_timeslot] = round(desiredModulationDegreeHeatPump,2)
            output_storagePercentagCombinedPerTimeSlotAverage[index_timeslot] = round(storagePercentage_fullRange_combined_average, 2)
            output_outsideTemperaturePerTimeSlot [index_timeslot] = round (outsideTemperature, 1)


            #Round values in the arrays
            output_electricalLoadHeatPumpPerTimeSlot= np.round(output_electricalLoadHeatPumpPerTimeSlot, 1)
            output_electricityCostsPerTimeSlot = np.round(output_electricityCostsPerTimeSlot, 2)
            output_coolingPowerRoom1PerTimeSlot = np.round(output_coolingPowerRoom1PerTimeSlot, 1)
            output_coolingPowerRoom2PerTimeSlot = np.round(output_coolingPowerRoom2PerTimeSlot, 1)
            output_coolingPowerRoom3PerTimeSlot = np.round(output_coolingPowerRoom3PerTimeSlot, 1)
            output_coolingPowerRoom4PerTimeSlot = np.round(output_coolingPowerRoom4PerTimeSlot, 1)
            output_coolingPowerRoom5PerTimeSlot = np.round(output_coolingPowerRoom5PerTimeSlot, 1)

            #Calculate thermal discomfort
            thermalDiscomfortScoreRoom1 = HelpFunctions.calculateThermalDisComfort_ComfortModel2(output_temperatureRoom1PerTimeSlot[index_timeslot], comfortSzearioNumber, 1, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes)
            thermalDiscomfortScoreRoom2 = HelpFunctions.calculateThermalDisComfort_ComfortModel2(output_temperatureRoom2PerTimeSlot[index_timeslot], comfortSzearioNumber, 2, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes)
            thermalDiscomfortScoreRoom3 = HelpFunctions.calculateThermalDisComfort_ComfortModel2(output_temperatureRoom3PerTimeSlot[index_timeslot], comfortSzearioNumber, 3, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes)
            thermalDiscomfortScoreRoom4 = HelpFunctions.calculateThermalDisComfort_ComfortModel2(output_temperatureRoom4PerTimeSlot[index_timeslot], comfortSzearioNumber, 4, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes)
            thermalDiscomfortScoreRoom5 = HelpFunctions.calculateThermalDisComfort_ComfortModel2(output_temperatureRoom5PerTimeSlot[index_timeslot], comfortSzearioNumber, 5, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes)

            discomfort_Room1_lastTimeslot_HighTemperature = 0
            discomfort_Room1_lastTimeslot_LowTemperature = 0
            if output_temperatureRoom1PerTimeSlot[index_timeslot] > idealTemperature_Room1 + temperatureBufferForGettingPerfectScoreForThermalDiscomfort:
                discomfort_Room1_lastTimeslot_HighTemperature = thermalDiscomfortScoreRoom1
            if output_temperatureRoom1PerTimeSlot[index_timeslot] < idealTemperature_Room1 - temperatureBufferForGettingPerfectScoreForThermalDiscomfort:
                discomfort_Room1_lastTimeslot_LowTemperature = thermalDiscomfortScoreRoom1

            discomfort_Room2_lastTimeslot_HighTemperature = 0
            discomfort_Room2_lastTimeslot_LowTemperature = 0
            if output_temperatureRoom2PerTimeSlot[index_timeslot] > idealTemperature_Room2 + temperatureBufferForGettingPerfectScoreForThermalDiscomfort:
                discomfort_Room2_lastTimeslot_HighTemperature = thermalDiscomfortScoreRoom2
            if output_temperatureRoom2PerTimeSlot[index_timeslot] < idealTemperature_Room2 - temperatureBufferForGettingPerfectScoreForThermalDiscomfort:
                discomfort_Room2_lastTimeslot_LowTemperature = thermalDiscomfortScoreRoom3

            discomfort_Room3_lastTimeslot_HighTemperature = 0
            discomfort_Room3_lastTimeslot_LowTemperature = 0
            if output_temperatureRoom3PerTimeSlot[index_timeslot] > idealTemperature_Room3 + temperatureBufferForGettingPerfectScoreForThermalDiscomfort:
                discomfort_Room3_lastTimeslot_HighTemperature = thermalDiscomfortScoreRoom3
            if output_temperatureRoom3PerTimeSlot[index_timeslot] < idealTemperature_Room3 - temperatureBufferForGettingPerfectScoreForThermalDiscomfort:
                discomfort_Room3_lastTimeslot_LowTemperature = thermalDiscomfortScoreRoom3

            discomfort_Room4_lastTimeslot_HighTemperature = 0
            discomfort_Room4_lastTimeslot_LowTemperature = 0
            if output_temperatureRoom4PerTimeSlot[index_timeslot] > idealTemperature_Room4 + temperatureBufferForGettingPerfectScoreForThermalDiscomfort:
                discomfort_Room4_lastTimeslot_HighTemperature = thermalDiscomfortScoreRoom4
            if output_temperatureRoom4PerTimeSlot[index_timeslot] < idealTemperature_Room4 - temperatureBufferForGettingPerfectScoreForThermalDiscomfort:
                discomfort_Room4_lastTimeslot_LowTemperature = thermalDiscomfortScoreRoom4

            discomfort_Room5_lastTimeslot_HighTemperature = 0
            discomfort_Room5_lastTimeslot_LowTemperature = 0
            if output_temperatureRoom5PerTimeSlot[index_timeslot] > idealTemperature_Room5 + temperatureBufferForGettingPerfectScoreForThermalDiscomfort:
                discomfort_Room5_lastTimeslot_HighTemperature = thermalDiscomfortScoreRoom5
            if output_temperatureRoom5PerTimeSlot[index_timeslot] < idealTemperature_Room5 - temperatureBufferForGettingPerfectScoreForThermalDiscomfort:
                discomfort_Room5_lastTimeslot_LowTemperature = thermalDiscomfortScoreRoom5


            output_scoreThermalDiscomfortPerTimeSlot [index_timeslot] = round((thermalDiscomfortScoreRoom1 + thermalDiscomfortScoreRoom2 + thermalDiscomfortScoreRoom3 + thermalDiscomfortScoreRoom4 + thermalDiscomfortScoreRoom5), 2)
            thermalDiscomfort_HighTemperature_lastTimeSlot = discomfort_Room1_lastTimeslot_HighTemperature + discomfort_Room2_lastTimeslot_HighTemperature + discomfort_Room3_lastTimeslot_HighTemperature + discomfort_Room4_lastTimeslot_HighTemperature + discomfort_Room5_lastTimeslot_HighTemperature

            #Create result dataframe and return it

            if index_timeslot == optimizationHorizon -1:
                df_results = pd.DataFrame({'Timestamp': output_timeStamp [:],'Outside Temperature': output_outsideTemperaturePerTimeSlot[:],'T_Air_1': output_temperatureRoom1PerTimeSlot[:],'T_Air_2': output_temperatureRoom2PerTimeSlot[:], 'T_Air_3': output_temperatureRoom3PerTimeSlot[:], 'T_Air_4': output_temperatureRoom4PerTimeSlot[:], 'T_Air_5': output_temperatureRoom5PerTimeSlot[:],'Storage_1': output_storagePercentageRoom1PerTimeSlot[:], 'Storage_2': output_storagePercentageRoom2PerTimeSlot[:], 'Storage_3': output_storagePercentageRoom3PerTimeSlot[:], 'Storage_4': output_storagePercentageRoom4PerTimeSlot[:], 'Storage_5': output_storagePercentageRoom5PerTimeSlot[:], 'Storage_Combined': output_storagePercentagCombinedPerTimeSlot [:], 'Storage_Av': output_storagePercentagCombinedPerTimeSlotAverage [:], 'thermalDiscomfort': output_scoreThermalDiscomfortPerTimeSlot [:],'T_ideal':  output_idealTemperature[:] , 'P_Max':  output_maximumPowerHeatPump[:], 'P_elect': output_electricalLoadHeatPumpPerTimeSlot[:], 'Costs': output_electricityCostsPerTimeSlot[:], 'coolingPowerRoom1': output_coolingPowerRoom1PerTimeSlot[:], 'coolingPowerRoom2': output_coolingPowerRoom2PerTimeSlot[:], 'coolingPowerRoom3': output_coolingPowerRoom3PerTimeSlot[:], 'coolingPowerRoom4': output_coolingPowerRoom4PerTimeSlot[:], 'coolingPowerRoom5': output_coolingPowerRoom5PerTimeSlot[:], 'EER': output_coolingEfficiencyEERPerTimeSlot[:], 'Electricity Price': output_electricityTarifPerTimeSlot[:],'Price Factor': output_priceFactorPerTimeSlot[:], 'Storage Factor': output_storageFactorPerTimeSlot[:], 'Modulation Degree': output_modulationDegreeHeatPumpPerTimeSlot[:]})
                fileAdditionAdjustedPriceSignal = "adjPri0_"
                fileAdditionAdjustedModulationDegree = "adjMod0_"
                fileAdditionAdjustedCoolingDistribution = "adjDis0"
                if useAdjustedPriceSignal==True:
                    fileAdditionAdjustedPriceSignal = "adjPri1_"
                if adjustedModulationDegree == True:
                    fileAdditionAdjustedModulationDegree = "adjMod1_"
                if considerDiscomfortForTheDistributionOfCoolingEnergy == True:
                    fileAdditionAdjustedCoolingDistribution = "adjDis1"

                fileName = pathForTheRun + "/PriceStorageControl_"+ fileAdditionAdjustedPriceSignal + fileAdditionAdjustedModulationDegree + fileAdditionAdjustedCoolingDistribution + ".csv"
                df_results.to_csv(fileName, sep=';')

                #Print results
                result_averageThermalDiscomfort = round(output_scoreThermalDiscomfortPerTimeSlot.mean(axis=0),2)
                result_sumThermalDiscomfort = round(output_scoreThermalDiscomfortPerTimeSlot.sum(axis=0),2)
                result_sumCosts = round(output_electricityCostsPerTimeSlot.sum(axis=0)/100,2)


                return result_sumCosts, result_averageThermalDiscomfort


#Control algorithm 2: Conventional control (hysteresis based two point control)
def controlAlgorithm_AdaptiveHysteresis (startDate, endDate, pathForTheRun, frequencyForUpdatingIdealComfortTemperatures):
     #Read input data from pkl file
    import pandas as pd
    from src.model.simulation import loadFMU
    import examples.control_loop as control_loop
    pathToFMU = "./building-model/MultiZone.fmu"


    #Read pkl file with the weather data
    filename_pkl_data = "./data/data.pkl"
    filename_pkl_data_raw = "./data/control/data.pkl"
    df_data = pd.read_pickle(filename_pkl_data)
    df_data_raw = pd.read_pickle(filename_pkl_data_raw)

    df_data.rename(columns={'time': 'timestamp'}, inplace=True)
    df_data_raw.rename(columns={'time': 'timestamp'}, inplace=True)

    #df_data_raw['timestamp'] = df_data_raw.index
    df_data_raw.index = np.arange(0, len(df_data_raw))
    df_data_raw['timestamp'] = pd.to_datetime(df_data_raw['timestamp']).dt.strftime('%d.%m.%Y %H:%M')

    # Choose the relevant weather data for the simulation
    indexStartDate = df_data_raw.index[df_data_raw['timestamp'] == startDate].tolist() [0]
    indexEndDate = df_data_raw.index[df_data_raw['timestamp'] == endDate].tolist()[0]
    df_weatherDataForSimulation = df_data_raw  [:] [indexStartDate: indexEndDate+1]
    df_weatherDataForSimulation.reset_index(drop=True, inplace=True)

    #Read price data from csv file
    df_priceDayAhead = pd.read_csv("./Daten/DSM/Stromtarif_FlexKälte_DayAheadMarkt_2021.csv", sep=';')
    #df_priceTarif1 = pd.read_csv("./Daten/DSM/Stromtarif_FlexKälte_DynamischerTarif1_2021.csv", sep=';')

    #Duplicate the values of the price to have a timestamp for every 15 minutes
    df_priceDayAhead.columns = ['timestamp', 'Price']
    df_priceDayAhead['timestamp'] = pd.to_datetime(df_priceDayAhead['timestamp'], dayfirst=True)
    df_priceDayAhead = df_priceDayAhead.set_index('timestamp').asfreq('15T', method='ffill').reset_index()
    df_priceDayAhead['timestamp'] = pd.to_datetime(df_priceDayAhead['timestamp']).dt.strftime('%d.%m.%Y %H:%M')

    #Choose the relevant price data for the simulation
    indexStartDatePrice = df_priceDayAhead.index[df_priceDayAhead['timestamp'] == startDate].tolist()[0]
    indexEndDatePrice = df_priceDayAhead.index[df_priceDayAhead['timestamp'] == endDate].tolist()[0]
    df_priceDataForSimulation  = df_priceDayAhead [indexStartDatePrice: indexEndDatePrice+1]
    df_priceDataForSimulation.reset_index(drop=True, inplace=True)

    optimizationHorizon = len(df_weatherDataForSimulation)

    #Define the variables for checking if the room is being cooled down
    room1CoolingPeriod = False
    room2CoolingPeriod = False
    room3CoolingPeriod = False
    room4CoolingPeriod = False
    room5CoolingPeriod = False

    #Define the output arrays
    output_timeStamp = ["" for x in range(optimizationHorizon)]
    output_electricityCostsPerTimeSlot = np.zeros(optimizationHorizon)
    output_electricityTarifPerTimeSlot = np.zeros(optimizationHorizon)
    output_electricalLoadHeatPumpPerTimeSlot = np.zeros(optimizationHorizon)
    output_temperatureRoom1PerTimeSlot = np.zeros(optimizationHorizon)
    output_temperatureRoom2PerTimeSlot = np.zeros(optimizationHorizon)
    output_temperatureRoom3PerTimeSlot = np.zeros(optimizationHorizon)
    output_temperatureRoom4PerTimeSlot = np.zeros(optimizationHorizon)
    output_temperatureRoom5PerTimeSlot = np.zeros(optimizationHorizon)

    output_coolingPowerRoom1PerTimeSlot = np.zeros(optimizationHorizon)
    output_coolingPowerRoom2PerTimeSlot = np.zeros(optimizationHorizon)
    output_coolingPowerRoom3PerTimeSlot = np.zeros(optimizationHorizon)
    output_coolingPowerRoom4PerTimeSlot = np.zeros(optimizationHorizon)
    output_coolingPowerRoom5PerTimeSlot = np.zeros(optimizationHorizon)

    output_coolingEfficiencyEERPerTimeSlot = np.zeros(optimizationHorizon)
    output_modulationDegreeHeatPumpPerTimeSlot = np.zeros(optimizationHorizon)
    output_scoreThermalDiscomfortPerTimeSlot = np.zeros(optimizationHorizon)
    output_outsideTemperaturePerTimeSlot = np.zeros(optimizationHorizon)
    output_idealTemperature = np.zeros(optimizationHorizon)
    output_maximumPowerHeatPump = np.zeros(optimizationHorizon)

    helpCounterTimeSlotsForUpdatingEDF =0
    currentNumberTimeStepsHP_Running = 0
    currentNumberTimeStepsHP_StandBy = 0

    with src.model.fmu.yieldFMU(pathToFMU) as fmu:
        #Initialize room temperatures at the beginning
        initalOutsideTemperature = df_weatherDataForSimulation.loc[0, 'Ta']

        initialTemperature_Room1 = 26.1
        initialTemperature_Room2 = 26.1
        initialTemperature_Room3 = 26.1
        initialTemperature_Room4 = 26.1
        initialTemperature_Room5 = 26.1
        control_loop.setStartingTemperatures(fmu, initialTemperature_Room1, initialTemperature_Room2, initialTemperature_Room3 ,initialTemperature_Room4, initialTemperature_Room5)
        #Loop over all timeslots
        for index_timeslot in range (0, optimizationHorizon):
            outsideTemperature = df_weatherDataForSimulation.loc[index_timeslot, 'Ta']
            solarRadiation = df_weatherDataForSimulation.loc[index_timeslot, 'Phis']
            timeStamp  = df_weatherDataForSimulation.loc [index_timeslot, 'timestamp']
            output_timeStamp[index_timeslot] = str(timeStamp)
            helpCounterTimeSlotsForUpdatingEDF +=1
            output_outsideTemperaturePerTimeSlot[index_timeslot] = round(outsideTemperature, 1)

            #Function call for optimal comfort temeprature and maximum power of the heat pump
            idealTemperature_Room1 = HelpFunctions.calculateOptimalComfortTemperature_ComfortModel2(comfortSzearioNumber, 1, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes, optimalComfortTemperatureOffTimes)
            idealTemperature_Room2 = HelpFunctions.calculateOptimalComfortTemperature_ComfortModel2(comfortSzearioNumber, 2, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes, optimalComfortTemperatureOffTimes)
            idealTemperature_Room3 = HelpFunctions.calculateOptimalComfortTemperature_ComfortModel2(comfortSzearioNumber, 3, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes, optimalComfortTemperatureOffTimes)
            idealTemperature_Room4 = HelpFunctions.calculateOptimalComfortTemperature_ComfortModel2(comfortSzearioNumber, 4, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes, optimalComfortTemperatureOffTimes)
            idealTemperature_Room5 = HelpFunctions.calculateOptimalComfortTemperature_ComfortModel2(comfortSzearioNumber, 5, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes, optimalComfortTemperatureOffTimes)
            maximumElectricalPowerHeatPump = HelpFunctions.calculateMaximumElectricalPowerOfTheHeatPump(outsideTemperature)
            output_idealTemperature [index_timeslot] = idealTemperature_Room1
            output_maximumPowerHeatPump[index_timeslot] = maximumElectricalPowerHeatPump

            # Update comfort temperature for control continuously if desired
            if frequencyForUpdatingIdealComfortTemperatures == 'continuous':
                idealTemperatureForControl_Room1 = idealTemperature_Room1
                idealTemperatureForControl_Room2 = idealTemperature_Room2
                idealTemperatureForControl_Room3 = idealTemperature_Room3
                idealTemperatureForControl_Room4 = idealTemperature_Room4
                idealTemperatureForControl_Room5 = idealTemperature_Room5

            if helpCounterTimeSlotsForUpdatingEDF ==timeStepsForUpdatingEmpiricalDistributionFunction or index_timeslot==0:
                # Calculate empirial cumulative distribution function (ECDF) for the future prices
                from statsmodels.distributions.empirical_distribution import ECDF
                # ToDo: Question Adjust electricity Tarif with COP values (by dividing it by the COP values)
                electricityTarifCurrentDay = df_priceDataForSimulation.loc [index_timeslot: index_timeslot + 96 - 1, 'Price'].values
                ecdf_prices = ECDF(electricityTarifCurrentDay)
                helpCounterTimeSlotsForUpdatingEDF =0

                #Update comfort temperature for control on a daily basis if desired
                if frequencyForUpdatingIdealComfortTemperatures == 'daily':
                    temperaturesCurrentDay =  df_weatherDataForSimulation.loc[index_timeslot: index_timeslot + 96 - 1, 'Ta'].values
                    meanTemperatureCurrentDay = temperaturesCurrentDay.mean(axis = 0)
                    idealTemperatureForControl_Room1 = HelpFunctions.calculateOptimalComfortTemperature_ComfortModel2(comfortSzearioNumber, 1, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes, optimalComfortTemperatureOffTimes)
                    idealTemperatureForControl_Room2 = HelpFunctions.calculateOptimalComfortTemperature_ComfortModel2(comfortSzearioNumber, 2, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes, optimalComfortTemperatureOffTimes)
                    idealTemperatureForControl_Room3 = HelpFunctions.calculateOptimalComfortTemperature_ComfortModel2(comfortSzearioNumber, 3, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes, optimalComfortTemperatureOffTimes)
                    idealTemperatureForControl_Room4 = HelpFunctions.calculateOptimalComfortTemperature_ComfortModel2(comfortSzearioNumber, 4, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes, optimalComfortTemperatureOffTimes)
                    idealTemperatureForControl_Room5 = HelpFunctions.calculateOptimalComfortTemperature_ComfortModel2(comfortSzearioNumber, 5, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes, optimalComfortTemperatureOffTimes)

                # Update comfort temperature for control on a weekly basis if desired
                if frequencyForUpdatingIdealComfortTemperatures == 'weekly':
                    temperaturesCurrentDay =  df_weatherDataForSimulation.loc[index_timeslot: index_timeslot + 7* 96 - 1, 'Ta'].values
                    meanTemperatureCurrentWeek = temperaturesCurrentDay.mean(axis = 0)
                    idealTemperatureForControl_Room1 = HelpFunctions.calculateOptimalComfortTemperature_ComfortModel2(comfortSzearioNumber, 1, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes, optimalComfortTemperatureOffTimes)
                    idealTemperatureForControl_Room2 = HelpFunctions.calculateOptimalComfortTemperature_ComfortModel2(comfortSzearioNumber, 2, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes, optimalComfortTemperatureOffTimes)
                    idealTemperatureForControl_Room3 = HelpFunctions.calculateOptimalComfortTemperature_ComfortModel2(comfortSzearioNumber, 3, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes, optimalComfortTemperatureOffTimes)
                    idealTemperatureForControl_Room4 = HelpFunctions.calculateOptimalComfortTemperature_ComfortModel2(comfortSzearioNumber, 4, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes, optimalComfortTemperatureOffTimes)
                    idealTemperatureForControl_Room5 = HelpFunctions.calculateOptimalComfortTemperature_ComfortModel2(comfortSzearioNumber, 5, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes, optimalComfortTemperatureOffTimes)

            #Get room temperatures from the previous time slot
            if index_timeslot ==0:
                temperature_lastTimeslot_Room1 = idealTemperature_Room1
                temperature_lastTimeslot_Room2 = idealTemperature_Room2
                temperature_lastTimeslot_Room3 = idealTemperature_Room3
                temperature_lastTimeslot_Room4 = idealTemperature_Room4
                temperature_lastTimeslot_Room5 = idealTemperature_Room5


            else:
                temperature_lastTimeslot_Room1 = output_temperatureRoom1PerTimeSlot [index_timeslot - 1]
                temperature_lastTimeslot_Room2 = output_temperatureRoom2PerTimeSlot [index_timeslot - 1]
                temperature_lastTimeslot_Room3 = output_temperatureRoom3PerTimeSlot [index_timeslot - 1]
                temperature_lastTimeslot_Room4 = output_temperatureRoom4PerTimeSlot [index_timeslot - 1]
                temperature_lastTimeslot_Room5 = output_temperatureRoom5PerTimeSlot [index_timeslot - 1]

            #Check if cooling is necessary for the rooms
            if room1CoolingPeriod ==False:
                if temperature_lastTimeslot_Room1 > idealTemperatureForControl_Room1 + allowedDeviationTemperature_Room1:
                    room1CoolingPeriod = True

            if room2CoolingPeriod ==False:
                if temperature_lastTimeslot_Room2 > idealTemperatureForControl_Room2 + allowedDeviationTemperature_Room2:
                    room2CoolingPeriod = True

            if room3CoolingPeriod ==False:
                if temperature_lastTimeslot_Room3 > idealTemperatureForControl_Room3 + allowedDeviationTemperature_Room3:
                    room3CoolingPeriod = True

            if room4CoolingPeriod ==False:
                if temperature_lastTimeslot_Room4 > idealTemperatureForControl_Room4+ allowedDeviationTemperature_Room4:
                    room4CoolingPeriod = True

            if room5CoolingPeriod ==False:
                if temperature_lastTimeslot_Room5 > idealTemperatureForControl_Room5 + allowedDeviationTemperature_Room5:
                    room5CoolingPeriod = True



            if room1CoolingPeriod == True:
                if temperature_lastTimeslot_Room1 < idealTemperatureForControl_Room1 - allowedDeviationTemperature_Room1:
                    room1CoolingPeriod = False

            if room2CoolingPeriod == True:
                if temperature_lastTimeslot_Room2 < idealTemperatureForControl_Room2 - allowedDeviationTemperature_Room2 :
                    room2CoolingPeriod = False

            if room3CoolingPeriod == True:
                if temperature_lastTimeslot_Room3 < idealTemperature_Room3 - allowedDeviationTemperature_Room3:
                    room3CoolingPeriod = False

            if room4CoolingPeriod == True:
                if temperature_lastTimeslot_Room4 < idealTemperatureForControl_Room4 - allowedDeviationTemperature_Room4:
                    room4CoolingPeriod = False

            if room5CoolingPeriod == True:
                if temperature_lastTimeslot_Room5 < idealTemperatureForControl_Room5 - allowedDeviationTemperature_Room5 :
                    room5CoolingPeriod = False


            numberOfRoomsNeedingCooling = 0
            if room1CoolingPeriod == True:
                numberOfRoomsNeedingCooling +=1
            if room2CoolingPeriod == True:
                numberOfRoomsNeedingCooling +=1
            if room3CoolingPeriod == True:
                numberOfRoomsNeedingCooling +=1
            if room4CoolingPeriod == True:
                numberOfRoomsNeedingCooling +=1
            if room5CoolingPeriod == True:
                numberOfRoomsNeedingCooling +=1

            #Determine desired modulation degree of the heat pump
            if numberOfRoomsNeedingCooling == 0:
                desiredModulationDegreeHeatPump = 0
            if numberOfRoomsNeedingCooling == 1 or numberOfRoomsNeedingCooling == 2:
                desiredModulationDegreeHeatPump = 0.5
            if numberOfRoomsNeedingCooling == 3 or numberOfRoomsNeedingCooling == 4:
                desiredModulationDegreeHeatPump = 0.8
            if numberOfRoomsNeedingCooling == 5:
                desiredModulationDegreeHeatPump = 1


            # Calculate the cooling energy of the heat pump if the desired modulation degree is used and
            EEF_currentTimeSlot = HelpFunctions.calculateEfficiency_EEF(outsideTemperature, desiredModulationDegreeHeatPump)
            coolingEnergyOfTheHeatPump = desiredModulationDegreeHeatPump * timeResolutionInMinutes * EEF_currentTimeSlot * maximumElectricalPowerHeatPump * 60

            #Distribute the cooling Energy to the rooms
            if numberOfRoomsNeedingCooling >0:
                coolingEnergyPerRoom = coolingEnergyOfTheHeatPump / numberOfRoomsNeedingCooling
            else:
                coolingEnergyPerRoom = 0

            if room1CoolingPeriod == True:
                coolingEnergy_Room1 = coolingEnergyPerRoom
            else:
                coolingEnergy_Room1 = 0

            if room2CoolingPeriod == True:
                coolingEnergy_Room2 = coolingEnergyPerRoom
            else:
                coolingEnergy_Room2 = 0

            if room3CoolingPeriod == True:
                coolingEnergy_Room3 = coolingEnergyPerRoom
            else:
                coolingEnergy_Room3 = 0

            if room4CoolingPeriod == True:
                coolingEnergy_Room4 = coolingEnergyPerRoom
            else:
                coolingEnergy_Room4 = 0

            if room5CoolingPeriod == True:
                coolingEnergy_Room5 = coolingEnergyPerRoom
            else:
                coolingEnergy_Room5 = 0



            #Assign temperature values to the variables
            if index_timeslot ==0:
                temperatureRoom1 = initialTemperature_Room1
                temperatureRoom2 = initialTemperature_Room2
                temperatureRoom3 = initialTemperature_Room3
                temperatureRoom4 = initialTemperature_Room4
                temperatureRoom5 = initialTemperature_Room5

            if index_timeslot >0:
                temperatureRoom1 = output_temperatureRoom1PerTimeSlot[index_timeslot - 1]
                temperatureRoom2 = output_temperatureRoom2PerTimeSlot[index_timeslot - 1]
                temperatureRoom3 = output_temperatureRoom3PerTimeSlot[index_timeslot - 1]
                temperatureRoom4 = output_temperatureRoom4PerTimeSlot[index_timeslot - 1]
                temperatureRoom5 = output_temperatureRoom5PerTimeSlot[index_timeslot - 1]


            #Assign the cooling variables of the model and calculate new temperatures
            hvacRoom1 = (coolingEnergy_Room1 * (-1))/(timeResolutionInMinutes * 60)
            hvacRoom2 = coolingEnergy_Room2 * (-1)/(timeResolutionInMinutes * 60)
            hvacRoom3 = coolingEnergy_Room3 * (-1)/(timeResolutionInMinutes * 60)
            hvacRoom4 = coolingEnergy_Room4 * (-1)/(timeResolutionInMinutes * 60)
            hvacRoom5 = coolingEnergy_Room5 * (-1)/(timeResolutionInMinutes * 60)


            new_temperatureRoom1, new_temperatureRoom2, new_temperatureRoom3, new_temperatureRoom4, new_temperatureRoom5 = control_loop.getTemperatures(fmu, outsideTemperature, solarRadiation, temperatureRoom1, temperatureRoom2, temperatureRoom3, temperatureRoom4, temperatureRoom5, hvacRoom1, hvacRoom2, hvacRoom3, hvacRoom4, hvacRoom5)

            output_temperatureRoom1PerTimeSlot [index_timeslot] = round(new_temperatureRoom1,1)
            output_temperatureRoom2PerTimeSlot[index_timeslot] = round(new_temperatureRoom2,1)
            output_temperatureRoom3PerTimeSlot[index_timeslot] = round(new_temperatureRoom3,1)
            output_temperatureRoom4PerTimeSlot[index_timeslot] = round(new_temperatureRoom4,1)
            output_temperatureRoom5PerTimeSlot[index_timeslot] = round(new_temperatureRoom5,1)


            #Calculate the resulting output values for this timeslot
            output_electricalLoadHeatPumpPerTimeSlot [index_timeslot] = desiredModulationDegreeHeatPump * maximumElectricalPowerHeatPump
            output_electricityCostsPerTimeSlot [index_timeslot] = output_electricalLoadHeatPumpPerTimeSlot [index_timeslot] * timeResolutionInMinutes * 60 * (electricityTarifCurrentDay [helpCounterTimeSlotsForUpdatingEDF]/3600000)


            output_coolingPowerRoom1PerTimeSlot [index_timeslot] = coolingEnergy_Room1 / (timeResolutionInMinutes *60)
            output_coolingPowerRoom2PerTimeSlot [index_timeslot] = coolingEnergy_Room2 / (timeResolutionInMinutes *60)
            output_coolingPowerRoom3PerTimeSlot [index_timeslot] = coolingEnergy_Room3 / (timeResolutionInMinutes *60)
            output_coolingPowerRoom4PerTimeSlot [index_timeslot] = coolingEnergy_Room4 / (timeResolutionInMinutes *60)
            output_coolingPowerRoom5PerTimeSlot [index_timeslot] = coolingEnergy_Room5 / (timeResolutionInMinutes *60)

            output_coolingEfficiencyEERPerTimeSlot [index_timeslot] = round(EEF_currentTimeSlot,2)

            output_modulationDegreeHeatPumpPerTimeSlot  [index_timeslot] = round(desiredModulationDegreeHeatPump,2)



            #Round values in the arrays
            output_electricalLoadHeatPumpPerTimeSlot= np.round(output_electricalLoadHeatPumpPerTimeSlot, 1)
            output_electricityCostsPerTimeSlot = np.round(output_electricityCostsPerTimeSlot, 2)
            output_coolingPowerRoom1PerTimeSlot = np.round(output_coolingPowerRoom1PerTimeSlot, 1)
            output_coolingPowerRoom2PerTimeSlot = np.round(output_coolingPowerRoom2PerTimeSlot, 1)
            output_coolingPowerRoom3PerTimeSlot = np.round(output_coolingPowerRoom3PerTimeSlot, 1)
            output_coolingPowerRoom4PerTimeSlot = np.round(output_coolingPowerRoom4PerTimeSlot, 1)
            output_coolingPowerRoom5PerTimeSlot = np.round(output_coolingPowerRoom5PerTimeSlot, 1)

            #Calculate thermal discomfort
            thermalDiscomfortScoreRoom1 = HelpFunctions.calculateThermalDisComfort_ComfortModel2(output_temperatureRoom1PerTimeSlot[index_timeslot], comfortSzearioNumber, 1, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes)
            thermalDiscomfortScoreRoom2 = HelpFunctions.calculateThermalDisComfort_ComfortModel2(output_temperatureRoom2PerTimeSlot[index_timeslot], comfortSzearioNumber, 2, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes)
            thermalDiscomfortScoreRoom3 = HelpFunctions.calculateThermalDisComfort_ComfortModel2(output_temperatureRoom3PerTimeSlot[index_timeslot], comfortSzearioNumber, 3, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes)
            thermalDiscomfortScoreRoom4 = HelpFunctions.calculateThermalDisComfort_ComfortModel2(output_temperatureRoom4PerTimeSlot[index_timeslot], comfortSzearioNumber, 4, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes)
            thermalDiscomfortScoreRoom5 = HelpFunctions.calculateThermalDisComfort_ComfortModel2(output_temperatureRoom5PerTimeSlot[index_timeslot], comfortSzearioNumber, 5, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes)

            output_scoreThermalDiscomfortPerTimeSlot [index_timeslot] = round((thermalDiscomfortScoreRoom1 + thermalDiscomfortScoreRoom2 + thermalDiscomfortScoreRoom3 + thermalDiscomfortScoreRoom4 + thermalDiscomfortScoreRoom5), 2)
            output_electricityTarifPerTimeSlot [index_timeslot] = electricityTarifCurrentDay [helpCounterTimeSlotsForUpdatingEDF]

            #Create result dataframe and return it
            if index_timeslot == optimizationHorizon -1:
                df_results = pd.DataFrame({'Timestamp': output_timeStamp [:],'Outside Temperature': output_outsideTemperaturePerTimeSlot[:],'T_Air_1': output_temperatureRoom1PerTimeSlot[:],'T_Air_2': output_temperatureRoom2PerTimeSlot[:], 'T_Air_3': output_temperatureRoom3PerTimeSlot[:], 'T_Air_4': output_temperatureRoom4PerTimeSlot[:], 'T_Air_5': output_temperatureRoom5PerTimeSlot[:],'thermalDiscomfort': output_scoreThermalDiscomfortPerTimeSlot [:],'T_ideal':  output_idealTemperature[:] , 'P_Max':  output_maximumPowerHeatPump[:],  'P_elect': output_electricalLoadHeatPumpPerTimeSlot[:], 'Costs': output_electricityCostsPerTimeSlot[:],'Electricity Price':output_electricityTarifPerTimeSlot [:], 'coolingPowerRoom1': output_coolingPowerRoom1PerTimeSlot[:], 'coolingPowerRoom2': output_coolingPowerRoom2PerTimeSlot[:], 'coolingPowerRoom3': output_coolingPowerRoom3PerTimeSlot[:], 'coolingPowerRoom4': output_coolingPowerRoom4PerTimeSlot[:], 'coolingPowerRoom5': output_coolingPowerRoom5PerTimeSlot[:], 'EER': output_coolingEfficiencyEERPerTimeSlot[:], 'Electricity Price': output_electricityTarifPerTimeSlot[:], 'Modulation Degree': output_modulationDegreeHeatPumpPerTimeSlot[:]})
                fileName = pathForTheRun + "/AdaptiveHysteresis_"+ frequencyForUpdatingIdealComfortTemperatures + ".csv"
                df_results.to_csv(fileName, sep=';')

                #Print results
                result_averageThermalDiscomfort = round(output_scoreThermalDiscomfortPerTimeSlot.mean(axis=0),2)
                result_sumThermalDiscomfort = round(output_scoreThermalDiscomfortPerTimeSlot.sum(axis=0),2)
                result_sumCosts = round(output_electricityCostsPerTimeSlot.sum(axis=0)/100,2)

                return result_sumCosts, result_averageThermalDiscomfort


#This function calculates the thermal discomfort of measured temperature data
def calculateThermalDiscomfortOfMeasuredTemperatureData (startDate, endDate):
    #Read input data from pkl file
    import pandas as pd



    #Read pkl file with the weather data
    #filename_pkl_data = "./data/data.pkl"
    filename_pkl_data_raw = "./data/influx/data.pkl"
    #resources / data / influx / data.pkl
    df_data_raw = pd.read_pickle(filename_pkl_data_raw)

    df_data_raw.rename(columns={'time': 'timestamp'}, inplace=True)

    #df_data_raw['timestamp'] = df_data_raw.index
    df_data_raw.index = np.arange(0, len(df_data_raw))
    df_data_raw['timestamp'] = pd.to_datetime(df_data_raw['timestamp']).dt.strftime('%d.%m.%Y %H:%M')

    # Choose the relevant weather data for the simulation
    indexStartDate = df_data_raw.index[df_data_raw['timestamp'] == startDate].tolist() [0]
    indexEndDate = df_data_raw.index[df_data_raw['timestamp'] == endDate].tolist()[0]
    df_weatherDataForSimulation = df_data_raw  [:] [indexStartDate: indexEndDate+1]
    df_weatherDataForSimulation.reset_index(drop=True, inplace=True)

    optimizationHorizon = len(df_weatherDataForSimulation)
    output_scoreThermalDiscomfortPerTimeSlot = np.zeros(optimizationHorizon)
    helpCounterTimeSlotsForUpdatingEDF = 0
    for index_timeslot in range(0, optimizationHorizon):
        outsideTemperature = df_weatherDataForSimulation.loc[index_timeslot, 'Ta']


        # Function call for optimal comfort temeprature and maximum power of the heat pump
        maximumElectricalPowerHeatPump = HelpFunctions.calculateMaximumElectricalPowerOfTheHeatPump(outsideTemperature)

        temperatureRoom1 =  df_weatherDataForSimulation.loc[index_timeslot, 'Ti1']
        temperatureRoom2 = df_weatherDataForSimulation.loc[index_timeslot, 'Ti2']
        temperatureRoom3 = df_weatherDataForSimulation.loc[index_timeslot, 'Ti3']
        temperatureRoom4 = df_weatherDataForSimulation.loc[index_timeslot, 'Ti4']
        temperatureRoom5 = df_weatherDataForSimulation.loc[index_timeslot, 'Ti5']

            #Calculate thermal discomfort
        thermalDiscomfortScoreRoom1 = HelpFunctions.calculateThermalDisComfort_ComfortModel2(temperatureRoom1, comfortSzearioNumber, 1, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes)
        thermalDiscomfortScoreRoom2 = HelpFunctions.calculateThermalDisComfort_ComfortModel2(temperatureRoom2, comfortSzearioNumber, 2, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes)
        thermalDiscomfortScoreRoom3 = HelpFunctions.calculateThermalDisComfort_ComfortModel2(temperatureRoom3, comfortSzearioNumber, 3, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes)
        thermalDiscomfortScoreRoom4 = HelpFunctions.calculateThermalDisComfort_ComfortModel2(temperatureRoom4, comfortSzearioNumber, 4, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes)
        thermalDiscomfortScoreRoom5 = HelpFunctions.calculateThermalDisComfort_ComfortModel2(temperatureRoom5, comfortSzearioNumber, 5, helpCounterTimeSlotsForUpdatingEDF, timeResolutionInMinutes)

        output_scoreThermalDiscomfortPerTimeSlot [index_timeslot] = round((thermalDiscomfortScoreRoom1 + thermalDiscomfortScoreRoom2 + thermalDiscomfortScoreRoom3 + thermalDiscomfortScoreRoom4 + thermalDiscomfortScoreRoom5), 2)
        helpCounterTimeSlotsForUpdatingEDF += 1
        if (helpCounterTimeSlotsForUpdatingEDF>=96):
            helpCounterTimeSlotsForUpdatingEDF = 0

    sumOfDiscomfort = output_scoreThermalDiscomfortPerTimeSlot.sum(axis=0)
    averageDiscomfortPerTimeSlot = round((sumOfDiscomfort/optimizationHorizon),2)

    return averageDiscomfortPerTimeSlot

#This is the script for executing the algorithms
if __name__ == "__main__":
    import pandas as pd
    import HelpFunctions
    import os
    from time import gmtime, strftime
    import time


    runprefix = 'Run_Date' + strftime("%d_%m_Time_%H_%M", gmtime()) + "_Dev" + str(allowedDeviationHelpValue) + "_OffTemp" + str(optimalComfortTemperatureOffTimes) + "_Sz" + str(comfortSzearioNumber)


    #Set Parameters for the control algorithm
    considerTemperatureRangesInsteadOfSetPoint = False
    sumUpIndividualStoragePercentagesInsteadOfAverage = False
    multiplyFactorsInsteadOfSum = True
    onlyUsePriceFactor = False

    useAdjustedPriceSingal = False
    considerDiscomfortForTheDistributionOfCoolingEnergy = False
    adjustedModulationDegree = True

    #Choose weeks for the simulation
    startingDateSimulation = ['30.05.2022 00:00', '06.06.2022 00:00' ,'13.06.2022 00:00', '20.06.2022 00:00', '27.06.2022 00:00', '04.07.2022 00:00', '11.07.2022 00:00', '18.07.2022 00:00', '25.07.2022 00:00', '01.08.2022 00:00', '08.08.2022 00:00', '15.08.2022 00:00', '22.08.2022 00:00']
    endDateSimulation =      ['05.06.2022 23:45', '12.06.2022 23:45' ,'19.06.2022 23:45', '26.06.2022 23:45', '03.07.2022 23:45', '10.07.2022 23:45', '17.07.2022 23:45', '24.07.2022 23:45', '31.07.2022 23:45', '07.08.2022 23:45', '14.08.2022 23:45', '21.08.2022 23:45', '28.08.2022 23:45']
    startingWeek = 13
    endWeek = 13
    weeksForTheSimulations = [2,3,4,5,6,7,9,11,12]
    nameOfMethods = ['PriceStorageControl_adjPri0_adjDis0_adjMod0',  'PriceStorageControl_adjPri1_adjDis0_adjMod0','PriceStorageControl_adjPri0_adjDis1_adjMod0', 'PriceStorageControl_adjPri0_adjDis0_adjMod1', 'PriceStorageControl_adjPri1_adjDis1_adjMod0', 'PriceStorageControl_adjPri0_adjDis1_adjMod1', 'PriceStorageControl_adjPri1_adjDis0_adjMod01', 'PriceStorageControl_adjPri1_adjDis1_adjMod1', 'AdaptiveHysteresis_Continuous', 'AdaptiveHysteresis_Daily', 'AdaptiveHysteresis_Weekly' ]
    numberOfDifferentMethods = 2

    #Set up arrays for storing the results of each week for each method
    costsPerWeek_Method = np.zeros((numberOfDifferentMethods, len(weeksForTheSimulations)))
    averageDiscomfortPerWeek_Method = np.zeros((numberOfDifferentMethods, len(weeksForTheSimulations)))
    averageDiscomfortPerWeek_MeasuredData = np.zeros((len(weeksForTheSimulations)))


    pathForTheRun = "./Ergebnisse/Paper Applied Energy/" + runprefix
    if not os.path.isdir(pathForTheRun):
        os.makedirs(pathForTheRun)

    index_week = -1
    for week in weeksForTheSimulations:
        index_week += 1

        discomfortMeasuredData = calculateThermalDiscomfortOfMeasuredTemperatureData(startingDateSimulation[week-1], endDateSimulation[week-1])


        df_results_oneWeek = pd.DataFrame(columns=['Configuration','Costs [Euro]','AverageThermalDiscomfort [°C]'])
        pathForTheSubRun = pathForTheRun  + "/Week" + str(week)
        if not os.path.isdir(pathForTheSubRun):
            os.makedirs(pathForTheSubRun)
        print(f"Calculate Week {week}")
        

        

        startTime = time.time()

        result_priceStorageControl_costsInEuro, result_priceStorageControl_averageThermalDiscomfort = controlAlgorithm_PriceStorageControl(considerTemperatureRangesInsteadOfSetPoint, sumUpIndividualStoragePercentagesInsteadOfAverage, multiplyFactorsInsteadOfSum, onlyUsePriceFactor, startingDateSimulation[week-1], endDateSimulation[week-1], pathForTheSubRun, useAdjustedPriceSignal= False, considerDiscomfortForTheDistributionOfCoolingEnergy=True, adjustedModulationDegree=False)
        df_results_oneWeek.loc[2] = ['PriceStorageControl_adjPri0_adjDis1_adjMod0'] + [round(result_priceStorageControl_costsInEuro,2 )] + [round(result_priceStorageControl_averageThermalDiscomfort,2)]
        costsPerWeek_Method [2, index_week] = round(result_priceStorageControl_costsInEuro,2 )
        averageDiscomfortPerWeek_Method [2, index_week] = round(result_priceStorageControl_averageThermalDiscomfort,2)

        result_AdaptiveHysteresis_costsInEuro, result_AdaptiveHysteresis_averageThermalDiscomfort = controlAlgorithm_AdaptiveHysteresis( startingDateSimulation[week-1], endDateSimulation[week-1], pathForTheSubRun, 'continuous')
        df_results_oneWeek.loc[8] = ['AdaptiveHysteresis_Continuous'] + [round(result_AdaptiveHysteresis_costsInEuro,2 )] + [round(result_AdaptiveHysteresis_averageThermalDiscomfort,2)]
        costsPerWeek_Method[8] [index_week] = round(result_AdaptiveHysteresis_costsInEuro, 2)
        averageDiscomfortPerWeek_Method[8] [index_week] = round(result_AdaptiveHysteresis_averageThermalDiscomfort, 2)


        endTime = time.time()

        averageTimeNeeded = (endTime-  startTime )/numberOfDifferentMethods

        #print(f"Average Time Needed (Week {week}) = {round(averageTimeNeeded, 2)} sec.")
        print("")

        df_results_oneWeek.to_excel(pathForTheRun + "/Results_Week"+ str(week) +".xlsx")


    #Print results for all weeks to excel file
    df_results_allWeek = pd.DataFrame(columns=['Name', 'Average Costs [€]', 'Average Discomfort [°C]'])
    for i in range(0, numberOfDifferentMethods):
        costsAllWeeks_Method = round(costsPerWeek_Method[i].mean(axis=0), 2)
        averageDiscomfortAllWeeks_Method = round(averageDiscomfortPerWeek_Method[i].mean(axis=0), 2)

        df_results_allWeek.loc[i] = [nameOfMethods [i], costsAllWeeks_Method, averageDiscomfortAllWeeks_Method]

    averageDiscomfortAllWeeks_MeasuredData = averageDiscomfortPerWeek_MeasuredData.mean(axis=0)
    df_results_allWeek.loc[numberOfDifferentMethods] = ['MeasuredData', "-", averageDiscomfortAllWeeks_MeasuredData]

    df_results_allWeek.to_excel(pathForTheRun + "/Results_AllWeeks.xlsx")






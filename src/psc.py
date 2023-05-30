from datetime import datetime
from functools import partial
from pathlib import Path
import os
import numpy as np
import pandas as pd
from do_mpc.data import save_results

from paper_revision.core.dataprovider import DataProvider
from paper_revision.core.externaldata import retrieve_data
from paper_revision.core.heatpump import HeatPumpModel
from paper_revision.core.model import create_model
from paper_revision.core.modelparameters import ModelParametersType
from paper_revision.core.simulator import create_simulator
from paper_revision.core.temperaturebounds import TemperatureBoundsType
from paper_revision.core.zones import ZoneDataProvider, Zones
from paper_revision.utils.json import JSONConfig

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def main() -> None:
    """
    Runs the psc simulation for each week of the provided external data.

    The function loads and generates the necessary data and configuration files, creates the model and simulator.
    Then, for each week of the external data, it runs the main loop of the simulation,
    saves the results, and moves on to the next week.
    """

    ###############################################################################################
    ###############               define scenario here                              ###############
    ###############################################################################################

    result_dir = DATA_DIR / "results" / "psc" / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    temperature_bounds_type = TemperatureBoundsType.BASE
    model_parameters_type = ModelParametersType.LOW
    initial_states = np.array([22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0])
    useAdjustedPriceSignal = False

    minimumModulationDegree = 0.25
    minimalTimeStepsHP_Running = 2
    minimalTimeStepsHP_StandBy = 2

    timeResolutionInMinutes = 15

    #Parameters to be adjusted for tuning

    onlyUsePriceFactor = False
    thresholdNotHeating_AverageStorageFactor_onlyUsePriceFactor = 0.75
    thresholdNotHeating_AverageStorageFactor_PSC = 0.75


    #Create Folder for the results
    currentDatetimeString = datetime.today().strftime('%d_%m_%Y_Time_%H_%M_%S')
    if temperature_bounds_type == TemperatureBoundsType.BASE:
        additional_name = "Base_"
    else:
        additional_name = "Adaptive_"

    if model_parameters_type == ModelParametersType.LOW:
        additional_name = additional_name + "_Low"
    else:
        additional_name = additional_name + "_High"

    if onlyUsePriceFactor == True:
        additional_name = additional_name + "_OnlyPF"

    pathForTheRun = "C:/Users/wi9632/Desktop/Ergebnisse/Paper Applied Energy/PSC/" + currentDatetimeString + "_" + additional_name

    try:
        os.makedirs(pathForTheRun)
    except OSError:
        print ("Creation of the directory %s failed" % pathForTheRun)

    #weeks_for_simulation = [2]
    weeks_for_simulation = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    #Parameters of PSC
    considerTemperatureRangesInsteadOfSetPoint = True
    sumUpIndividualStoragePercentagesInsteadOfAverage = False




    multiplyFactorsInsteadOfSum = True
    adjustedModulationDegree = True
    considerDiscomfortForTheDistributionOfHeatingEnergy = False
    buffer_allowed_temperature_deviaton = 0.5

    ###############################################################################################
    ###############################################################################################

    # load / generate data
    external_data = retrieve_data(DATA_DIR / "external" / "timeseries_data.csv")

    heat_pump_config = JSONConfig(
        DATA_DIR / "config" / "heat_pump_config.json", DATA_DIR / "config" / "heat_pump_schema.json"
    )
    heat_pump_model = HeatPumpModel(config=heat_pump_config)

    model_parameters_config = JSONConfig(
        DATA_DIR / "config" / "model_parameters_config.json", DATA_DIR / "config" / "model_parameters_schema.json"
    )
    temperature_bounds_config = JSONConfig(
        DATA_DIR / "config" / "temperature_bounds_config.json", DATA_DIR / "config" / "temperature_bounds_schema.json"
    )

    zone_data_provider = ZoneDataProvider(
        model_parameters_data=model_parameters_config.load(), temperature_bounds_data=temperature_bounds_config.load()
    )

    data_provider = DataProvider()
    data_provider.generate_dataset(
        data=external_data,
        heat_pump_model=heat_pump_model,
        temperature_bounds=partial(zone_data_provider.temperature_bounds, bounds_type=temperature_bounds_type),
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    data_provider.save_dataset(result_dir / "dataset.csv")

    # generate model
    model = create_model(
        model_parameters=partial(zone_data_provider.model_parameters, parameters_type=model_parameters_type)
    )

    #Create results dataframe for the total results of each week
    df_results_weeks = pd.DataFrame(columns=["Week", "Costs", "Average Thermal Discomfort"])

    # run each week
    #for week in range(len(data_provider.weekly_datasets)):
    for week in weeks_for_simulation:
        # get data for that week
        data = data_provider.get_data(week)
        current_states = initial_states

        # generate simulator
        simulator = create_simulator(model, data)
        simulator.x0 = initial_states

        ###########################################################################################
        ###############               setup pre-loop here                           ###############
        ###########################################################################################
        # Initialize the output arrays of the results
        optimizationHorizon = len(data.index)
        output_timeStamp = ["" for x in range(optimizationHorizon)]
        output_electricityCostsPerTimeSlot = np.zeros(optimizationHorizon)
        output_electricityTarifPerTimeSlot = np.zeros(optimizationHorizon)
        output_electricalLoadHeatPumpPerTimeSlot = np.zeros(optimizationHorizon)
        output_temperatureRoom1PerTimeSlot = np.zeros(optimizationHorizon)
        output_temperatureRoom2PerTimeSlot = np.zeros(optimizationHorizon)
        output_temperatureRoom3PerTimeSlot = np.zeros(optimizationHorizon)
        output_temperatureRoom4PerTimeSlot = np.zeros(optimizationHorizon)
        output_temperatureRoom5PerTimeSlot = np.zeros(optimizationHorizon)


        output_storagePercentageRoom1PerTimeSlot = np.zeros(optimizationHorizon)
        output_storagePercentageRoom2PerTimeSlot = np.zeros(optimizationHorizon)
        output_storagePercentageRoom3PerTimeSlot = np.zeros(optimizationHorizon)
        output_storagePercentageRoom4PerTimeSlot = np.zeros(optimizationHorizon)
        output_storagePercentageRoom5PerTimeSlot = np.zeros(optimizationHorizon)
        output_storagePercentagCombinedPerTimeSlot = np.zeros(optimizationHorizon)
        output_storagePercentagCombinedPerTimeSlotAverage = np.zeros(optimizationHorizon)

        output_heatingPowerRoom1PerTimeSlot = np.zeros(optimizationHorizon)
        output_heatingPowerRoom2PerTimeSlot = np.zeros(optimizationHorizon)
        output_heatingPowerRoom3PerTimeSlot = np.zeros(optimizationHorizon)
        output_heatingPowerRoom4PerTimeSlot = np.zeros(optimizationHorizon)
        output_heatingPowerRoom5PerTimeSlot = np.zeros(optimizationHorizon)

        output_priceFactorPerTimeSlot = np.zeros(optimizationHorizon)
        output_storageFactorPerTimeSlot = np.zeros(optimizationHorizon)
        output_modulationDegreeHeatPumpPerTimeSlot = np.zeros(optimizationHorizon)
        output_scoreThermalDiscomfortPerTimeSlot = np.zeros(optimizationHorizon)
        output_maximumPowerHeatPump = np.zeros(optimizationHorizon)
        output_heatingEfficiencyCOPPerTimeSlot = np.zeros(optimizationHorizon)

        currentNumberTimeStepsHP_Running = 0
        currentNumberTimeStepsHP_StandBy = 0
        helpCounterTimeSlotsForUpdatingEDF = 0
        thermalDiscomfort_LowTemperature_lastTimeSlot = 0
        timeStepsForUpdatingEmpiricalDistributionFunction = 96

        thermalDiscomfort_LowTemperature_lastTimeSlot = 0
        thermalDiscomfort_HighTemperature_lastTimeSlot = 0

        # nothing to do here

        ###########################################################################################
        ###########################################################################################

        # run main loop
        for index in range(len(data.index)):
            #######################################################################################
            ###############               calculate inputs here                     ###############
            #######################################################################################
            timeStamp = data.loc[index, 'timestamp']
            output_timeStamp[index] = str(timeStamp)
            helpCounterTimeSlotsForUpdatingEDF +=1

            outsideTemperature = data.loc[index, 'ambient_temperature']


            #optimal comfort temeprature and maximum power of the heat pump
            idealTemperature_Room1 = (data.loc[index, 'temperature_room_1_max'] + data.loc[index, 'temperature_room_1_min']) / 2
            idealTemperature_Room2 = (data.loc[index, 'temperature_room_2_max'] + data.loc[index, 'temperature_room_2_min']) / 2
            idealTemperature_Room3 = (data.loc[index, 'temperature_room_3_max'] + data.loc[index, 'temperature_room_3_min']) / 2
            idealTemperature_Room4 = (data.loc[index, 'temperature_room_4_max'] + data.loc[index, 'temperature_room_4_min']) / 2
            idealTemperature_Room5 = (data.loc[index, 'temperature_room_5_max'] + data.loc[index, 'temperature_room_5_min']) / 2

            maximumElectricalPowerHeatPump = data.loc[index, 'heat_pump_power_max']

            output_maximumPowerHeatPump[index] = maximumElectricalPowerHeatPump


            # Update the electricity prices for the next day and the corresponding empirical distribution function at the beginning of each day
            if helpCounterTimeSlotsForUpdatingEDF ==timeStepsForUpdatingEmpiricalDistributionFunction or index==0:
                # Calculate empirial cumulative distribution function (ECDF) for the future prices
                from statsmodels.distributions.empirical_distribution import ECDF
                electricityTarifCurrentDay = data.loc [index: index + 96 - 1, 'electricity_price'].values
                adjustedPriceSignal = np.zeros(timeStepsForUpdatingEmpiricalDistributionFunction)
                #Calculate the EEF values for the next day in advance
                future_eef_values = np.zeros(96)
                for i in range(0, len(electricityTarifCurrentDay)):
                    future_eef_values [i] = data.loc[index + i, 'heat_pump_cop']
                    adjustedPriceSignal [i] = electricityTarifCurrentDay [i] / future_eef_values [i]


                if useAdjustedPriceSignal==True:
                    inputSignalPrice = adjustedPriceSignal
                else:
                    inputSignalPrice = electricityTarifCurrentDay
                ecdf_prices = ECDF(inputSignalPrice)
                helpCounterTimeSlotsForUpdatingEDF =0




            #Get room temperatures from the previous time slot
            if index ==0:
                temperature_lastTimeslot_Room1 = idealTemperature_Room1
                temperature_lastTimeslot_Room2 = idealTemperature_Room2
                temperature_lastTimeslot_Room3 = idealTemperature_Room3
                temperature_lastTimeslot_Room4 = idealTemperature_Room4
                temperature_lastTimeslot_Room5 = idealTemperature_Room5


            else:
                temperature_lastTimeslot_Room1 = output_temperatureRoom1PerTimeSlot [index - 1]
                temperature_lastTimeslot_Room2 = output_temperatureRoom2PerTimeSlot [index - 1]
                temperature_lastTimeslot_Room3 = output_temperatureRoom3PerTimeSlot [index - 1]
                temperature_lastTimeslot_Room4 = output_temperatureRoom4PerTimeSlot [index - 1]
                temperature_lastTimeslot_Room5 = output_temperatureRoom5PerTimeSlot [index - 1]





            #Calculate deviation to temperature limits by considering the ideal temperature setpoint

            #Room1
            temperature_distanceToLowerLimit_Room1 = temperature_lastTimeslot_Room1 - data.loc[index, 'temperature_room_1_min']
            allowedDeviationTemperature_Room1 = idealTemperature_Room1 - data.loc[index, 'temperature_room_1_min'] + buffer_allowed_temperature_deviaton

            if temperature_lastTimeslot_Room1 > idealTemperature_Room1:
                storagePercentage_Room1 = 1.0
            else:
                storagePercentage_Room1 = temperature_distanceToLowerLimit_Room1 / allowedDeviationTemperature_Room1
                if temperature_distanceToLowerLimit_Room1 <= 0:
                    storagePercentage_Room1 = 0


            #Room2
            temperature_distanceToLowerLimit_Room2 = temperature_lastTimeslot_Room2 - data.loc[index, 'temperature_room_2_min']
            allowedDeviationTemperature_Room2 = idealTemperature_Room2 - data.loc[index, 'temperature_room_2_min'] + buffer_allowed_temperature_deviaton

            if temperature_lastTimeslot_Room2 > idealTemperature_Room2:
                storagePercentage_Room2 = 1.0
            else:
                storagePercentage_Room2 = temperature_distanceToLowerLimit_Room2 / allowedDeviationTemperature_Room2
                if temperature_distanceToLowerLimit_Room2 <= 0:
                    storagePercentage_Room2 = 0


            #Room3
            temperature_distanceToLowerLimit_Room3 = temperature_lastTimeslot_Room3 - data.loc[index, 'temperature_room_3_min']
            allowedDeviationTemperature_Room3 = idealTemperature_Room3 - data.loc[index, 'temperature_room_3_min'] + buffer_allowed_temperature_deviaton

            if temperature_lastTimeslot_Room3 > idealTemperature_Room3:
                storagePercentage_Room3 = 1.0
            else:
                storagePercentage_Room3 = temperature_distanceToLowerLimit_Room3 / allowedDeviationTemperature_Room3
                if temperature_distanceToLowerLimit_Room3 <= 0:
                    storagePercentage_Room3 = 0



            #Room4
            temperature_distanceToLowerLimit_Room4 = temperature_lastTimeslot_Room4 - data.loc[index, 'temperature_room_4_min']
            allowedDeviationTemperature_Room4 = idealTemperature_Room4 - data.loc[index, 'temperature_room_4_min'] + buffer_allowed_temperature_deviaton

            if temperature_lastTimeslot_Room4 > idealTemperature_Room4:
                storagePercentage_Room4 = 1.0
            else:
                storagePercentage_Room4 = temperature_distanceToLowerLimit_Room4 / allowedDeviationTemperature_Room4
                if temperature_distanceToLowerLimit_Room4 <= 0:
                    storagePercentage_Room4 = 0



            #Room5
            temperature_distanceToLowerLimit_Room5 = temperature_lastTimeslot_Room5 - data.loc[index, 'temperature_room_5_min']
            allowedDeviationTemperature_Room5 = idealTemperature_Room5 - data.loc[index, 'temperature_room_5_min'] + buffer_allowed_temperature_deviaton

            if temperature_lastTimeslot_Room5 > idealTemperature_Room5:
                storagePercentage_Room5 = 1.0
            else:
                storagePercentage_Room5 = temperature_distanceToLowerLimit_Room5 / allowedDeviationTemperature_Room5
                if temperature_distanceToLowerLimit_Room5 <= 0:
                    storagePercentage_Room5 = 0



            #Calculate deviation to temperature limits by considering the full range of the temperature limits (storage_Percentage_fullRange=1 if temperature at lower limit and 0 at upper limit)

            # Room1
            storagePercentage_fullRange_Room1 = (temperature_lastTimeslot_Room1 - data.loc[index, 'temperature_room_1_min']) / (data.loc[index, 'temperature_room_1_max'] - data.loc[index, 'temperature_room_1_min'])
            if data.loc[index, 'temperature_room_1_max'] == 30:
                storagePercentage_fullRange_Room1 = storagePercentage_fullRange_Room1 + 0.2
            if helpCounterTimeSlotsForUpdatingEDF >= 24 and  helpCounterTimeSlotsForUpdatingEDF <= 31:
                storagePercentage_fullRange_Room1 = storagePercentage_fullRange_Room1  - 0.6
            if storagePercentage_fullRange_Room1 < 0:
                storagePercentage_fullRange_Room1 = 0
            if storagePercentage_fullRange_Room1 > 1:
                storagePercentage_fullRange_Room1 = 1



            # Room2
            storagePercentage_fullRange_Room2 = (temperature_lastTimeslot_Room2 - data.loc[index, 'temperature_room_2_min']) / (data.loc[index, 'temperature_room_2_max'] - data.loc[index, 'temperature_room_2_min'])
            if data.loc[index, 'temperature_room_2_max'] == 30:
                storagePercentage_fullRange_Room2 = storagePercentage_fullRange_Room2 + 0.2
            if helpCounterTimeSlotsForUpdatingEDF >= 24 and  helpCounterTimeSlotsForUpdatingEDF <= 31:
                storagePercentage_fullRange_Room2 = storagePercentage_fullRange_Room2  - 0.6
            if storagePercentage_fullRange_Room2 < 0:
                storagePercentage_fullRange_Room2 = 0
            if storagePercentage_fullRange_Room2 > 1:
                storagePercentage_fullRange_Room2 = 1


            # Room3
            storagePercentage_fullRange_Room3 = (temperature_lastTimeslot_Room3 - data.loc[index, 'temperature_room_3_min']) / (data.loc[index, 'temperature_room_3_max'] - data.loc[index, 'temperature_room_3_min'])
            if data.loc[index, 'temperature_room_3_max'] == 30:
                storagePercentage_fullRange_Room3 = storagePercentage_fullRange_Room3 + 0.2
            if helpCounterTimeSlotsForUpdatingEDF >= 24 and  helpCounterTimeSlotsForUpdatingEDF <= 31:
                storagePercentage_fullRange_Room3 = storagePercentage_fullRange_Room3  - 0.2
            if storagePercentage_fullRange_Room3 < 0:
                storagePercentage_fullRange_Room3 = 0
            if storagePercentage_fullRange_Room3 > 1:
                storagePercentage_fullRange_Room3 = 1


            # Room4
            storagePercentage_fullRange_Room4 = (temperature_lastTimeslot_Room4 - data.loc[index, 'temperature_room_4_min']) / (data.loc[index, 'temperature_room_4_max'] - data.loc[index, 'temperature_room_4_min'])
            if data.loc[index, 'temperature_room_4_max'] == 30:
                storagePercentage_fullRange_Room4 = storagePercentage_fullRange_Room4 + 0.2
            if helpCounterTimeSlotsForUpdatingEDF >= 24 and  helpCounterTimeSlotsForUpdatingEDF <= 31:
                storagePercentage_fullRange_Room4 = storagePercentage_fullRange_Room4  - 0.2
            if storagePercentage_fullRange_Room4 < 0:
                storagePercentage_fullRange_Room4 = 0
            if storagePercentage_fullRange_Room4 > 1:
                storagePercentage_fullRange_Room4 = 1


            # Room5
            storagePercentage_fullRange_Room5 = (temperature_lastTimeslot_Room5 - data.loc[index, 'temperature_room_5_min']) / (data.loc[index, 'temperature_room_5_max'] - data.loc[index, 'temperature_room_5_min'])
            if data.loc[index, 'temperature_room_5_max'] == 30:
                storagePercentage_fullRange_Room5 = storagePercentage_fullRange_Room5 + 0.2
            if helpCounterTimeSlotsForUpdatingEDF >= 24 and  helpCounterTimeSlotsForUpdatingEDF <= 31:
                storagePercentage_fullRange_Room5 = storagePercentage_fullRange_Room5  - 0.2
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

            #Reduce heating towards  the end of the optimization horizon by reducing the price factor
            if index > 625:
                priceFactor = priceFactor /2


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

            output_storagePercentageRoom1PerTimeSlot [index] = round(storagePercentage_Room1, 2)
            output_storagePercentageRoom2PerTimeSlot[index] = round(storagePercentage_Room2, 2)
            output_storagePercentageRoom3PerTimeSlot[index] = round(storagePercentage_Room3, 2)
            output_storagePercentageRoom4PerTimeSlot[index] = round(storagePercentage_Room4, 2)
            output_storagePercentageRoom5PerTimeSlot[index] = round(storagePercentage_Room5, 2)





            #Derive the desired modulation degree of the heat pump
            if multiplyFactorsInsteadOfSum == True:
                desiredModulationDegreeHeatPump = storageFactor_Combined * priceFactor
            else:
                weight_storageFactor = 0.5
                weight_priceFactor = 0.5
                desiredModulationDegreeHeatPump = storageFactor_Combined * weight_storageFactor + priceFactor * weight_priceFactor


            if onlyUsePriceFactor==True:
                desiredModulationDegreeHeatPump = priceFactor * priceFactor
                if storagePercentage_fullRange_combined_average > thresholdNotHeating_AverageStorageFactor_onlyUsePriceFactor :
                    desiredModulationDegreeHeatPump = 0
                if storagePercentage_fullRange_combined_average < 0.10 :
                    desiredModulationDegreeHeatPump = desiredModulationDegreeHeatPump + 0.25
            else:
                if storagePercentage_fullRange_combined_average > thresholdNotHeating_AverageStorageFactor_PSC :
                    desiredModulationDegreeHeatPump = 0

            #Adust modulation degree if average storage percentage is small
            if adjustedModulationDegree ==True:
                #Adjust desiredModulaitonDegree if average storage is too low
                if storagePercentage_fullRange_combined_average < 0.1 and desiredModulationDegreeHeatPump < minimumModulationDegree:
                    desiredModulationDegreeHeatPump = minimumModulationDegree


                #Adjust desiredModulationDegree if there is too high discomfort because of too high temperatures
                if thermalDiscomfort_HighTemperature_lastTimeSlot > 0 and desiredModulationDegreeHeatPump > minimumModulationDegree:
                    desiredModulationDegreeHeatPump = 0
                if thermalDiscomfort_LowTemperature_lastTimeSlot > 1:
                    desiredModulationDegreeHeatPump = desiredModulationDegreeHeatPump + 0.2


                if storagePercentage_fullRange_combined_average <0.05:
                    desiredModulationDegreeHeatPump = desiredModulationDegreeHeatPump + 0.1

            #Check for minimum and maximum modulation degree
            if desiredModulationDegreeHeatPump < minimumModulationDegree:
                desiredModulationDegreeHeatPump = 0

            if desiredModulationDegreeHeatPump > 1:
                desiredModulationDegreeHeatPump = 1



            #Distribute the heating energy among the rooms
            freeCapacity_storageP_fullRange_combined = (1-storagePercentage_fullRange_Room1) + (1-storagePercentage_fullRange_Room2) + (1-storagePercentage_fullRange_Room3) + (1- storagePercentage_fullRange_Room4) + (1-storagePercentage_fullRange_Room5)

            #Stop heating if the storage is full (temperatur too cold)
            if storagePercentage_fullRange_combined >5 - 5* 0.1 and heatingEnergyOfTheHeatPump > 0.01:
                desiredModulationDegreeHeatPump =0

            #Force heating with full power if the storage is empty (temperatur too cold)
            if storagePercentage_fullRange_combined < 0.1 and heatingEnergyOfTheHeatPump < 0.01:
                desiredModulationDegreeHeatPump = 1


            #Consider minimal runtime of HP
            if desiredModulationDegreeHeatPump < 0.01 and currentNumberTimeStepsHP_Running >=1 and currentNumberTimeStepsHP_Running < minimalTimeStepsHP_Running:
                desiredModulationDegreeHeatPump = minimumModulationDegree

            #Consider minimal standby time of HP
            if desiredModulationDegreeHeatPump > 0.01 and currentNumberTimeStepsHP_StandBy >=1 and currentNumberTimeStepsHP_StandBy < minimalTimeStepsHP_StandBy:
                desiredModulationDegreeHeatPump =0


            # Calculate the cooling energy of the heat pump if the desired modulation degree is used and
            COP_currentTimeSlot = data.loc[index, 'heat_pump_cop']
            heatingEnergyOfTheHeatPump = desiredModulationDegreeHeatPump * timeResolutionInMinutes * COP_currentTimeSlot * maximumElectricalPowerHeatPump * 60
            output_heatingEfficiencyCOPPerTimeSlot [index] = COP_currentTimeSlot
            #Assign the heating load to the different rooms if the combined storage percentage is below a threshold value (e.g. 90%)
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

            heatingEnergy_Room1 = (storageShare_OppositeValue_Room1/sumStorageShares_OppositeValue) * heatingEnergyOfTheHeatPump
            heatingEnergy_Room2 = (storageShare_OppositeValue_Room2/sumStorageShares_OppositeValue) * heatingEnergyOfTheHeatPump
            heatingEnergy_Room3 = (storageShare_OppositeValue_Room3/sumStorageShares_OppositeValue) * heatingEnergyOfTheHeatPump
            heatingEnergy_Room4 = (storageShare_OppositeValue_Room4/sumStorageShares_OppositeValue) * heatingEnergyOfTheHeatPump
            heatingEnergy_Room5 = (storageShare_OppositeValue_Room5/sumStorageShares_OppositeValue) * heatingEnergyOfTheHeatPump


            #Assign the share of cooling energy based on the discomfort score
            if index >0 and considerDiscomfortForTheDistributionOfHeatingEnergy == True:
                if output_scoreThermalDiscomfortPerTimeSlot [index -1] > 0:
                    #Don't heat rooms that are already too warm
                    if discomfort_Room1_lastTimeslot_HighTemperature >0:
                        heatingEnergy_Room1 =0
                    if discomfort_Room2_lastTimeslot_HighTemperature > 0:
                        heatingnergy_Room2 = 0
                    if discomfort_Room3_lastTimeslot_HighTemperature > 0:
                        heatingEnergy_Room3 = 0
                    if discomfort_Room4_lastTimeslot_HighTemperature > 0:
                        heatingEnergy_Room4 = 0
                    if discomfort_Room5_lastTimeslot_HighTemperature > 0:
                        heatingEnergy_Room5 = 0
                    if heatingEnergy_Room1 ==0 and heatingEnergy_Room2 ==0 and heatingEnergy_Room3 ==0 and heatingEnergy_Room4 ==0 and heatingEnergy_Room5 ==0:
                        desiredModulationDegreeHeatPump = 0
                        heatingEnergyOfTheHeatPump =0

                    #Calculate total discomfort because of too high temperatures
                    totalDiscomfort_lastTimeSlot_HighTemperature = discomfort_Room1_lastTimeslot_HighTemperature + discomfort_Room2_lastTimeslot_HighTemperature + discomfort_Room3_lastTimeslot_HighTemperature + discomfort_Room4_lastTimeslot_HighTemperature + discomfort_Room5_lastTimeslot_HighTemperature

                    #Calculate total discomfort because of too low temperatures
                    totalDiscomfort_lastTimeSlot_LowTemperature = discomfort_Room1_lastTimeslot_LowTemperature + discomfort_Room2_lastTimeslot_LowTemperature + discomfort_Room3_lastTimeslot_LowTemperature + discomfort_Room4_lastTimeslot_LowTemperature + discomfort_Room5_lastTimeslot_LowTemperature


                    #Calculate the share of each room for causing thermal discomfort because of too low temperatures
                    if totalDiscomfort_lastTimeSlot_LowTemperature >0:
                        shareOfRoom1ForThermalDiscomfort_LastTimeSlot = discomfort_Room1_lastTimeslot_LowTemperature /totalDiscomfort_lastTimeSlot_LowTemperature
                        shareOfRoom2ForThermalDiscomfort_LastTimeSlot = discomfort_Room2_lastTimeslot_LowTemperature / totalDiscomfort_lastTimeSlot_LowTemperature
                        shareOfRoom3ForThermalDiscomfort_LastTimeSlot = discomfort_Room3_lastTimeslot_LowTemperature / totalDiscomfort_lastTimeSlot_LowTemperature
                        shareOfRoom4ForThermalDiscomfort_LastTimeSlot = discomfort_Room4_lastTimeslot_LowTemperature / totalDiscomfort_lastTimeSlot_LowTemperature
                        shareOfRoom5ForThermalDiscomfort_LastTimeSlot = discomfort_Room5_lastTimeslot_LowTemperature / totalDiscomfort_lastTimeSlot_LowTemperature

                        #Assign heating energy to the rooms based on discomfort
                        heatingEnergy_Room1 = shareOfRoom1ForThermalDiscomfort_LastTimeSlot * heatingEnergyOfTheHeatPump
                        heatingEnergy_Room2 = shareOfRoom2ForThermalDiscomfort_LastTimeSlot * heatingEnergyOfTheHeatPump
                        heatingEnergy_Room3 = shareOfRoom3ForThermalDiscomfort_LastTimeSlot * heatingEnergyOfTheHeatPump
                        heatingEnergy_Room4 = shareOfRoom4ForThermalDiscomfort_LastTimeSlot * heatingEnergyOfTheHeatPump
                        heatingEnergy_Room5 = shareOfRoom5ForThermalDiscomfort_LastTimeSlot * heatingEnergyOfTheHeatPump



            #Adjust counters for number of running and standby steps of the HP
            if heatingEnergyOfTheHeatPump == 0:
                currentNumberTimeStepsHP_StandBy += 1
                currentNumberTimeStepsHP_Running = 0

            if heatingEnergyOfTheHeatPump > 0.1:
                currentNumberTimeStepsHP_Running += 1
                currentNumberTimeStepsHP_StandBy = 0


            #Calculate the resulting output values for this timeslot
            output_electricalLoadHeatPumpPerTimeSlot [index] = desiredModulationDegreeHeatPump * maximumElectricalPowerHeatPump
            output_electricityCostsPerTimeSlot [index] = output_electricalLoadHeatPumpPerTimeSlot [index] * timeResolutionInMinutes * 60 * (electricityTarifCurrentDay [helpCounterTimeSlotsForUpdatingEDF]/3600000)
            output_electricityTarifPerTimeSlot [index] = electricityTarifCurrentDay [helpCounterTimeSlotsForUpdatingEDF]




            #Assign temperature values to the variables
            if index ==0:
                temperatureRoom1 = idealTemperature_Room1
                temperatureRoom2 = idealTemperature_Room2
                temperatureRoom3 = idealTemperature_Room3
                temperatureRoom4 = idealTemperature_Room4
                temperatureRoom5 = idealTemperature_Room5

            if index >0:
                temperatureRoom1 = output_temperatureRoom1PerTimeSlot[index - 1]
                temperatureRoom2 = output_temperatureRoom2PerTimeSlot[index - 1]
                temperatureRoom3 = output_temperatureRoom3PerTimeSlot[index - 1]
                temperatureRoom4 = output_temperatureRoom4PerTimeSlot[index - 1]
                temperatureRoom5 = output_temperatureRoom5PerTimeSlot[index - 1]


            # Assign the heating variables of the model and calculate new temperatures
            hvacRoom1 = (heatingEnergy_Room1)/(timeResolutionInMinutes * 60)
            hvacRoom2 = (heatingEnergy_Room2)/(timeResolutionInMinutes * 60)
            hvacRoom3 = (heatingEnergy_Room3)/(timeResolutionInMinutes * 60)
            hvacRoom4 = (heatingEnergy_Room4)/(timeResolutionInMinutes * 60)
            hvacRoom5 = (heatingEnergy_Room5)/(timeResolutionInMinutes * 60)

            output_heatingPowerRoom1PerTimeSlot [index] = hvacRoom1
            output_heatingPowerRoom2PerTimeSlot [index] = hvacRoom2
            output_heatingPowerRoom3PerTimeSlot [index] = hvacRoom3
            output_heatingPowerRoom4PerTimeSlot [index] = hvacRoom4
            output_heatingPowerRoom5PerTimeSlot [index] = hvacRoom5

            #Innputs for simulation
            inputs = np.array([[hvacRoom1], [hvacRoom2], [hvacRoom3], [hvacRoom4], [hvacRoom5]])

            # simulate model and obtain new states
            current_states = simulator.make_step(inputs)

            discomfort_values = [
                max(
                    [0],
                    simulator.data["_aux", f"temperature_{zone}_lower_bound"][index],
                    simulator.data["_aux", f"temperature_{zone}_upper_bound"][index],
                )
                for zone in Zones
            ]

            temperature_room_1 = current_states[0]
            temperature_room_2 = current_states[2]
            temperature_room_3 = current_states[4]
            temperature_room_4 = current_states[6]
            temperature_room_5 = current_states[8]

            output_temperatureRoom1PerTimeSlot [index] = round(temperature_room_1[0],1)
            output_temperatureRoom2PerTimeSlot[index] = round(temperature_room_2[0],1)
            output_temperatureRoom3PerTimeSlot[index] = round(temperature_room_3[0],1)
            output_temperatureRoom4PerTimeSlot[index] = round(temperature_room_4[0],1)
            output_temperatureRoom5PerTimeSlot[index] = round(temperature_room_5[0],1)



            output_priceFactorPerTimeSlot  [index] = round(priceFactor,2)
            output_storageFactorPerTimeSlot [index] = round(storageFactor_Combined,2)
            output_storagePercentagCombinedPerTimeSlot [index] = round(storagePercentage_fullRange_combined,2)
            output_modulationDegreeHeatPumpPerTimeSlot  [index] = round(desiredModulationDegreeHeatPump,2)
            output_storagePercentagCombinedPerTimeSlotAverage[index] = round(storagePercentage_fullRange_combined_average, 2)


            #Round values in the arrays
            output_electricalLoadHeatPumpPerTimeSlot= np.round(output_electricalLoadHeatPumpPerTimeSlot, 1)
            output_electricityCostsPerTimeSlot = np.round(output_electricityCostsPerTimeSlot, 2)
            output_heatingPowerRoom1PerTimeSlot = np.round(output_heatingPowerRoom1PerTimeSlot, 1)
            output_heatingPowerRoom2PerTimeSlot = np.round(output_heatingPowerRoom2PerTimeSlot, 1)
            output_heatingPowerRoom3PerTimeSlot = np.round(output_heatingPowerRoom3PerTimeSlot, 1)
            output_heatingPowerRoom4PerTimeSlot = np.round(output_heatingPowerRoom4PerTimeSlot, 1)
            output_heatingPowerRoom5PerTimeSlot = np.round(output_heatingPowerRoom5PerTimeSlot, 1)
            output_heatingEfficiencyCOPPerTimeSlot  = np.round(output_heatingEfficiencyCOPPerTimeSlot, 2)
            output_maximumPowerHeatPump  = np.round(output_maximumPowerHeatPump, 1)
            output_electricityTarifPerTimeSlot = np.round(output_electricityTarifPerTimeSlot, 2)

            #Calculate thermal discomfort
            thermalDiscomfortScoreRoom1 = discomfort_values [0][0]
            thermalDiscomfortScoreRoom2 = discomfort_values [1][0]
            thermalDiscomfortScoreRoom3 = discomfort_values [2][0]
            thermalDiscomfortScoreRoom4 = discomfort_values [3][0]
            thermalDiscomfortScoreRoom5 = discomfort_values [4][0]

            discomfort_Room1_lastTimeslot_HighTemperature = 0
            discomfort_Room1_lastTimeslot_LowTemperature = 0
            if output_temperatureRoom1PerTimeSlot[index] > data.loc[index, 'temperature_room_1_max']:
                discomfort_Room1_lastTimeslot_HighTemperature = thermalDiscomfortScoreRoom1
            if output_temperatureRoom1PerTimeSlot[index] < data.loc[index, 'temperature_room_1_min']:
                discomfort_Room1_lastTimeslot_LowTemperature = thermalDiscomfortScoreRoom1

            discomfort_Room2_lastTimeslot_HighTemperature = 0
            discomfort_Room2_lastTimeslot_LowTemperature = 0
            if output_temperatureRoom2PerTimeSlot[index] > data.loc[index, 'temperature_room_2_max']:
                discomfort_Room2_lastTimeslot_HighTemperature = thermalDiscomfortScoreRoom2
            if output_temperatureRoom2PerTimeSlot[index] < data.loc[index, 'temperature_room_2_min']:
                discomfort_Room2_lastTimeslot_LowTemperature = thermalDiscomfortScoreRoom3

            discomfort_Room3_lastTimeslot_HighTemperature = 0
            discomfort_Room3_lastTimeslot_LowTemperature = 0
            if output_temperatureRoom3PerTimeSlot[index] > data.loc[index, 'temperature_room_3_max']:
                discomfort_Room3_lastTimeslot_HighTemperature = thermalDiscomfortScoreRoom3
            if output_temperatureRoom3PerTimeSlot[index] < data.loc[index, 'temperature_room_3_min']:
                discomfort_Room3_lastTimeslot_LowTemperature = thermalDiscomfortScoreRoom3

            discomfort_Room4_lastTimeslot_HighTemperature = 0
            discomfort_Room4_lastTimeslot_LowTemperature = 0
            if output_temperatureRoom4PerTimeSlot[index] > data.loc[index, 'temperature_room_4_max']:
                discomfort_Room4_lastTimeslot_HighTemperature = thermalDiscomfortScoreRoom4
            if output_temperatureRoom4PerTimeSlot[index] < data.loc[index, 'temperature_room_4_min']:
                discomfort_Room4_lastTimeslot_LowTemperature = thermalDiscomfortScoreRoom4

            discomfort_Room5_lastTimeslot_HighTemperature = 0
            discomfort_Room5_lastTimeslot_LowTemperature = 0
            if output_temperatureRoom5PerTimeSlot[index] > data.loc[index, 'temperature_room_5_max']:
                discomfort_Room5_lastTimeslot_HighTemperature = thermalDiscomfortScoreRoom5
            if output_temperatureRoom5PerTimeSlot[index] < data.loc[index, 'temperature_room_5_min']:
                discomfort_Room5_lastTimeslot_LowTemperature = thermalDiscomfortScoreRoom5


            output_scoreThermalDiscomfortPerTimeSlot [index] = round((thermalDiscomfortScoreRoom1 + thermalDiscomfortScoreRoom2 + thermalDiscomfortScoreRoom3 + thermalDiscomfortScoreRoom4 + thermalDiscomfortScoreRoom5), 2)
            thermalDiscomfort_LowTemperature_lastTimeSlot = discomfort_Room1_lastTimeslot_LowTemperature + discomfort_Room2_lastTimeslot_LowTemperature + discomfort_Room3_lastTimeslot_LowTemperature + discomfort_Room4_lastTimeslot_LowTemperature + discomfort_Room5_lastTimeslot_LowTemperature
            thermalDiscomfort_HighTemperature_lastTimeSlot = discomfort_Room1_lastTimeslot_HighTemperature + discomfort_Room2_lastTimeslot_HighTemperature + discomfort_Room3_lastTimeslot_HighTemperature + discomfort_Room4_lastTimeslot_HighTemperature + discomfort_Room5_lastTimeslot_HighTemperature
            #Create result dataframe and return it

            if index == optimizationHorizon -1:
                df_results = pd.DataFrame({'Timestamp': output_timeStamp [:],'T_Air_1': output_temperatureRoom1PerTimeSlot[:],'T_Air_2': output_temperatureRoom2PerTimeSlot[:], 'T_Air_3': output_temperatureRoom3PerTimeSlot[:], 'T_Air_4': output_temperatureRoom4PerTimeSlot[:], 'T_Air_5': output_temperatureRoom5PerTimeSlot[:],'Storage_1': output_storagePercentageRoom1PerTimeSlot[:], 'Storage_2': output_storagePercentageRoom2PerTimeSlot[:], 'Storage_3': output_storagePercentageRoom3PerTimeSlot[:], 'Storage_4': output_storagePercentageRoom4PerTimeSlot[:], 'Storage_5': output_storagePercentageRoom5PerTimeSlot[:], 'Storage_Combined': output_storagePercentagCombinedPerTimeSlot [:], 'Storage_Av': output_storagePercentagCombinedPerTimeSlotAverage [:], 'thermalDiscomfort': output_scoreThermalDiscomfortPerTimeSlot [:] , 'P_Max':  output_maximumPowerHeatPump[:], 'P_elect': output_electricalLoadHeatPumpPerTimeSlot[:], 'Costs': output_electricityCostsPerTimeSlot[:], 'heatingPowerRoom1': output_heatingPowerRoom1PerTimeSlot[:], 'heatingPowerRoom2': output_heatingPowerRoom2PerTimeSlot[:], 'heatingPowerRoom3': output_heatingPowerRoom3PerTimeSlot[:], 'heatingPowerRoom4': output_heatingPowerRoom4PerTimeSlot[:], 'heatingPowerRoom5': output_heatingPowerRoom5PerTimeSlot[:], 'COP': output_heatingEfficiencyCOPPerTimeSlot[:],  'Electricity Price': output_electricityTarifPerTimeSlot[:],'Price Factor': output_priceFactorPerTimeSlot[:], 'Storage Factor': output_storageFactorPerTimeSlot[:], 'Modulation Degree': output_modulationDegreeHeatPumpPerTimeSlot[:]})
                fileAdditionAdjustedPriceSignal = "adjPri0_"
                fileAdditionAdjustedModulationDegree = "adjMod0_"
                fileAdditionAdjustedCoolingDistribution = "adjDis0"
                fileAdditionOnlyPriceFactor = ""
                if useAdjustedPriceSignal==True:
                    fileAdditionAdjustedPriceSignal = "adjPri1_"
                if adjustedModulationDegree == True:
                    fileAdditionAdjustedModulationDegree = "adjMod1_"
                if considerDiscomfortForTheDistributionOfHeatingEnergy == True:
                    fileAdditionAdjustedCoolingDistribution = "adjDis1"
                if onlyUsePriceFactor == True:
                    fileAdditionOnlyPriceFactor = "onlyPF"

                #fileName = pathForTheRun + "/PriceStorageControl_"+ fileAdditionAdjustedPriceSignal + fileAdditionAdjustedModulationDegree + fileAdditionAdjustedCoolingDistribution  + fileAdditionOnlyPriceFactor + "_Week" + str(week + 1) + ".csv"
                fileName = pathForTheRun + "/PriceStorageControl_" + fileAdditionOnlyPriceFactor + "_Week" + str(week + 1) + ".csv"
                df_results.to_csv(fileName, sep=';')

                #Print results
                result_averageThermalDiscomfort = round(output_scoreThermalDiscomfortPerTimeSlot.mean(axis=0),2)
                result_sumThermalDiscomfort = round(output_scoreThermalDiscomfortPerTimeSlot.sum(axis=0),2)
                result_sumCosts = round(output_electricityCostsPerTimeSlot.sum(axis=0),2)



                #Print results
                result_averageThermalDiscomfort = round(output_scoreThermalDiscomfortPerTimeSlot.mean(axis=0),2)
                result_sumThermalDiscomfort = round(output_scoreThermalDiscomfortPerTimeSlot.sum(axis=0),2)
                result_sumCosts = round(output_electricityCostsPerTimeSlot.sum(axis=0),2)
                print("Results")
                print(f"Week: {week + 1}")
                print(f"result_averageThermalDiscomfort: {result_averageThermalDiscomfort}")
                print(f"result_sumCosts: {result_sumCosts}")

                df_results_weeks = df_results_weeks._append({"Week": week + 1, "Costs": result_sumCosts, "Average Thermal Discomfort": result_averageThermalDiscomfort}, ignore_index=True)

    if onlyUsePriceFactor == True:
        df_results_weeks.to_excel(pathForTheRun + "\Results_PSC_OnlyPF.xlsx", index=False)
    else:
        df_results_weeks.to_excel(pathForTheRun + "\Results_PSC.xlsx", index=False)





        ###########################################################################################
        ###############               process post-loop data here                   ###############
        ###########################################################################################

        # nothing to do here

        ###########################################################################################
        ###########################################################################################


if __name__ == "__main__":
    main()

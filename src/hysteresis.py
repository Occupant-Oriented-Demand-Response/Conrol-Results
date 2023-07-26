import os
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from do_mpc.data import save_results

from src.core.dataprovider import DataProvider
from src.core.externaldata import retrieve_data
from src.core.heatpump import HeatPumpModel
from src.core.model import create_model
from src.core.modelparameters import ModelParametersType
from src.core.simulator import create_simulator
from src.core.temperaturebounds import TemperatureBoundsType
from src.core.zones import ZoneDataProvider, Zones
from src.utils.json import JSONConfig

DATA_DIR = Path(__file__).parent.parent / "data"


def main() -> None:
    """
    Runs the hysteresis simulation for each week of the provided external data.

    The function loads and generates the necessary data and configuration files, creates the model and simulator.
    Then, for each week of the external data, it runs the main loop of the simulation,
    saves the results, and moves on to the next week.
    """

    ###############################################################################################
    ###############               define scenario here                              ###############
    ###############################################################################################

    temperature_bounds_type = TemperatureBoundsType.BASE
    model_parameters_type = ModelParametersType.LOW
    initial_states = np.array([22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0])
    result_dir = DATA_DIR / "results" / "hyteresis" / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    frequencyForUpdatingIdealComfortTemperatures = 'continuous'
    timeStepsForUpdatingEmpiricalDistributionFunction = 96
    timeResolutionInMinutes = 15


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
    pathForTheRun = "YOUR PATH" + currentDatetimeString + "_" + additional_name # add your path here

    try:
        os.makedirs(pathForTheRun)
    except OSError:
        print ("Creation of the directory %s failed" % folderPath_WholeSimulation)

    allowedDeviationHelpValue = 1.5  # Range [1.0, 1.5, 2.0]

    # Temperatures for the different rooms
    allowedDeviationTemperature_Room1 = allowedDeviationHelpValue  # Unit: [°C]
    allowedDeviationTemperature_Room2 = allowedDeviationHelpValue  # Unit: [°C]
    allowedDeviationTemperature_Room3 = allowedDeviationHelpValue  # Unit: [°C]
    allowedDeviationTemperature_Room4 = allowedDeviationHelpValue  # Unit: [°C]
    allowedDeviationTemperature_Room5 = allowedDeviationHelpValue  # Unit: [°C]

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
    for week in range(0, 10):
        # get data for that week
        data = data_provider.get_data(week)
        current_states = initial_states

        # generate simulator
        simulator = create_simulator(model, data)
        simulator.x0 = initial_states

        ###########################################################################################
        ###############               setup pre-loop here                           ###############
        ###########################################################################################

        # Define the variables for checking if the room is being heated up
        room1HeatingPeriod = False
        room2HeatingPeriod = False
        room3HeatingPeriod = False
        room4HeatingPeriod = False
        room5HeatingPeriod = False

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

        output_heatingPowerRoom1PerTimeSlot = np.zeros(optimizationHorizon)
        output_heatingPowerRoom2PerTimeSlot = np.zeros(optimizationHorizon)
        output_heatingPowerRoom3PerTimeSlot = np.zeros(optimizationHorizon)
        output_heatingPowerRoom4PerTimeSlot = np.zeros(optimizationHorizon)
        output_heatingPowerRoom5PerTimeSlot = np.zeros(optimizationHorizon)
        output_heatingEfficiencyCOPPerTimeSlot  = np.zeros(optimizationHorizon)

        output_modulationDegreeHeatPumpPerTimeSlot = np.zeros(optimizationHorizon)
        output_scoreThermalDiscomfortPerTimeSlot = np.zeros(optimizationHorizon)
        output_outsideTemperaturePerTimeSlot = np.zeros(optimizationHorizon)
        output_idealTemperature = np.zeros(optimizationHorizon)
        output_maximumPowerHeatPump = np.zeros(optimizationHorizon)

        helpCounterTimeSlotsForUpdatingEDF = 0
        currentNumberTimeStepsHP_Running = 0
        currentNumberTimeStepsHP_StandBy = 0


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



            if helpCounterTimeSlotsForUpdatingEDF ==timeStepsForUpdatingEmpiricalDistributionFunction or index==0:
                # Calculate empirial cumulative distribution function (ECDF) for the future prices
                from statsmodels.distributions.empirical_distribution import ECDF
                electricityTarifCurrentDay = data.loc [index: index + 96 - 1, 'electricity_price'].values
                adjustedPriceSignal = np.zeros(timeStepsForUpdatingEmpiricalDistributionFunction)
                #Calculate the EEF values for the next day in advance
                future_eef_values = np.zeros(96)
                helpCounterTimeSlotsForUpdatingEDF = 0



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

            #Check if heating is necessary for the rooms

            if room1HeatingPeriod ==False:
                if temperature_lastTimeslot_Room1 < idealTemperature_Room1 - allowedDeviationTemperature_Room1:
                    room1HeatingPeriod = True

            if room2HeatingPeriod ==False:
                if temperature_lastTimeslot_Room2 < idealTemperature_Room2 - allowedDeviationTemperature_Room2:
                    room2HeatingPeriod = True

            if room3HeatingPeriod ==False:
                if temperature_lastTimeslot_Room3 < idealTemperature_Room3 - allowedDeviationTemperature_Room3:
                    room3HeatingPeriod = True

            if room4HeatingPeriod ==False:
                if temperature_lastTimeslot_Room4 < idealTemperature_Room4 - allowedDeviationTemperature_Room4:
                    room4HeatingPeriod = True

            if room5HeatingPeriod ==False:
                if temperature_lastTimeslot_Room5 < idealTemperature_Room5 - allowedDeviationTemperature_Room5:
                    room5HeatingPeriod = True



            if room1HeatingPeriod == True:
                if temperature_lastTimeslot_Room1 > idealTemperature_Room1 + allowedDeviationTemperature_Room1:
                    room1HeatingPeriod = False

            if room2HeatingPeriod == True:
                if temperature_lastTimeslot_Room2 > idealTemperature_Room2+ allowedDeviationTemperature_Room2 :
                    room2HeatingPeriod = False

            if room3HeatingPeriod == True:
                if temperature_lastTimeslot_Room3 > idealTemperature_Room3 + allowedDeviationTemperature_Room3:
                    room3HeatingPeriod = False

            if room4HeatingPeriod == True:
                if temperature_lastTimeslot_Room4 > idealTemperature_Room4 + allowedDeviationTemperature_Room4:
                    room4HeatingPeriod = False

            if room5HeatingPeriod == True:
                if temperature_lastTimeslot_Room5 > idealTemperature_Room5 + allowedDeviationTemperature_Room5 :
                    room5HeatingPeriod = False


            numberOfRoomsNeedingHeating = 0
            if room1HeatingPeriod == True:
                numberOfRoomsNeedingHeating +=1
            if room2HeatingPeriod == True:
                numberOfRoomsNeedingHeating +=1
            if room3HeatingPeriod == True:
                numberOfRoomsNeedingHeating +=1
            if room4HeatingPeriod == True:
                numberOfRoomsNeedingHeating +=1
            if room5HeatingPeriod == True:
                numberOfRoomsNeedingHeating +=1

            #Determine desired modulation degree of the heat pump
            if numberOfRoomsNeedingHeating == 0:
                desiredModulationDegreeHeatPump = 0
            if numberOfRoomsNeedingHeating == 1 or numberOfRoomsNeedingHeating == 2:
                desiredModulationDegreeHeatPump = 0.5
            if numberOfRoomsNeedingHeating == 3 or numberOfRoomsNeedingHeating == 4:
                desiredModulationDegreeHeatPump = 0.8
            if numberOfRoomsNeedingHeating == 5:
                desiredModulationDegreeHeatPump = 1


            # Calculate the heating energy of the heat pump if the desired modulation degree is used and
            COP_currentTimeSlot = data.loc[index, 'heat_pump_cop']
            heatingEnergyOfTheHeatPump = desiredModulationDegreeHeatPump * timeResolutionInMinutes * COP_currentTimeSlot * maximumElectricalPowerHeatPump * 60

            #Distribute the heating Energy to the rooms
            if numberOfRoomsNeedingHeating >0:
                heatingEnergyPerRoom = heatingEnergyOfTheHeatPump / numberOfRoomsNeedingHeating
            else:
                heatingEnergyPerRoom = 0

            if room1HeatingPeriod == True:
                heatingEnergy_Room1 = heatingEnergyPerRoom
            else:
                heatingEnergy_Room1 = 0

            if room2HeatingPeriod == True:
                heatingEnergy_Room2 = heatingEnergyPerRoom
            else:
                heatingEnergy_Room2 = 0

            if room3HeatingPeriod == True:
                heatingEnergy_Room3 = heatingEnergyPerRoom
            else:
                heatingEnergy_Room3 = 0

            if room4HeatingPeriod == True:
                heatingEnergy_Room4 = heatingEnergyPerRoom
            else:
                heatingEnergy_Room4 = 0

            if room5HeatingPeriod == True:
                heatingEnergy_Room5 = heatingEnergyPerRoom
            else:
                heatingEnergy_Room5 = 0



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


            #Assign the heating variables of the model and calculate new temperatures
            hvacRoom1 = heatingEnergy_Room1 /(timeResolutionInMinutes * 60)
            hvacRoom2 = heatingEnergy_Room2 /(timeResolutionInMinutes * 60)
            hvacRoom3 = heatingEnergy_Room3 /(timeResolutionInMinutes * 60)
            hvacRoom4 = heatingEnergy_Room4 /(timeResolutionInMinutes * 60)
            hvacRoom5 = heatingEnergy_Room5 /(timeResolutionInMinutes * 60)


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


            output_modulationDegreeHeatPumpPerTimeSlot  [index] = round(desiredModulationDegreeHeatPump,2)



            #Round values in the arrays
            output_electricalLoadHeatPumpPerTimeSlot= np.round(output_electricalLoadHeatPumpPerTimeSlot, 1)
            output_electricityCostsPerTimeSlot = np.round(output_electricityCostsPerTimeSlot, 2)
            output_heatingPowerRoom1PerTimeSlot = np.round(output_heatingPowerRoom1PerTimeSlot, 1)
            output_heatingPowerRoom2PerTimeSlot = np.round(output_heatingPowerRoom2PerTimeSlot, 1)
            output_heatingPowerRoom3PerTimeSlot = np.round(output_heatingPowerRoom3PerTimeSlot, 1)
            output_heatingPowerRoom4PerTimeSlot = np.round(output_heatingPowerRoom4PerTimeSlot, 1)
            output_heatingPowerRoom5PerTimeSlot = np.round(output_heatingPowerRoom5PerTimeSlot, 1)
            
            
            output_temperatureRoom1PerTimeSlot [index] = round(temperature_room_1[0],1)
            output_temperatureRoom2PerTimeSlot[index] = round(temperature_room_2[0],1)
            output_temperatureRoom3PerTimeSlot[index] = round(temperature_room_3[0],1)
            output_temperatureRoom4PerTimeSlot[index] = round(temperature_room_4[0],1)
            output_temperatureRoom5PerTimeSlot[index] = round(temperature_room_5[0],1)


            #Calculate the resulting output values for this timeslot
            output_electricalLoadHeatPumpPerTimeSlot [index] = desiredModulationDegreeHeatPump * maximumElectricalPowerHeatPump
            output_electricityCostsPerTimeSlot [index] = output_electricalLoadHeatPumpPerTimeSlot [index] * timeResolutionInMinutes * 60 * (electricityTarifCurrentDay [helpCounterTimeSlotsForUpdatingEDF]/3600000)


            output_heatingPowerRoom1PerTimeSlot [index] = heatingEnergy_Room1 / (timeResolutionInMinutes *60)
            output_heatingPowerRoom2PerTimeSlot [index] = heatingEnergy_Room2 / (timeResolutionInMinutes *60)
            output_heatingPowerRoom3PerTimeSlot [index] = heatingEnergy_Room3 / (timeResolutionInMinutes *60)
            output_heatingPowerRoom4PerTimeSlot [index] = heatingEnergy_Room4 / (timeResolutionInMinutes *60)
            output_heatingPowerRoom5PerTimeSlot [index] = heatingEnergy_Room5 / (timeResolutionInMinutes *60)

            output_heatingEfficiencyCOPPerTimeSlot [index] = round(COP_currentTimeSlot,2)

            output_modulationDegreeHeatPumpPerTimeSlot  [index] = round(desiredModulationDegreeHeatPump,2)



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

            # Calculate thermal discomfort
            thermalDiscomfortScoreRoom1 = discomfort_values[0][0]
            thermalDiscomfortScoreRoom2 = discomfort_values[1][0]
            thermalDiscomfortScoreRoom3 = discomfort_values[2][0]
            thermalDiscomfortScoreRoom4 = discomfort_values[3][0]
            thermalDiscomfortScoreRoom5 = discomfort_values[4][0]

            output_scoreThermalDiscomfortPerTimeSlot [index] = round((thermalDiscomfortScoreRoom1 + thermalDiscomfortScoreRoom2 + thermalDiscomfortScoreRoom3 + thermalDiscomfortScoreRoom4 + thermalDiscomfortScoreRoom5), 2)
            output_electricityTarifPerTimeSlot [index] = electricityTarifCurrentDay [helpCounterTimeSlotsForUpdatingEDF]

            #Create result dataframe and return it
            if index == optimizationHorizon -1:
                df_results = pd.DataFrame({'Timestamp': output_timeStamp [:],'Outside Temperature': output_outsideTemperaturePerTimeSlot[:],'T_Air_1': output_temperatureRoom1PerTimeSlot[:],'T_Air_2': output_temperatureRoom2PerTimeSlot[:], 'T_Air_3': output_temperatureRoom3PerTimeSlot[:], 'T_Air_4': output_temperatureRoom4PerTimeSlot[:], 'T_Air_5': output_temperatureRoom5PerTimeSlot[:],'thermalDiscomfort': output_scoreThermalDiscomfortPerTimeSlot [:],'T_ideal':  output_idealTemperature[:] , 'P_Max':  output_maximumPowerHeatPump[:],  'P_elect': output_electricalLoadHeatPumpPerTimeSlot[:], 'Costs': output_electricityCostsPerTimeSlot[:],'Electricity Price':output_electricityTarifPerTimeSlot [:], 'heatingPowerRoom1': output_heatingPowerRoom1PerTimeSlot[:], 'heatingPowerRoom2': output_heatingPowerRoom2PerTimeSlot[:], 'heatingPowerRoom3': output_heatingPowerRoom3PerTimeSlot[:], 'heatingPowerRoom4': output_heatingPowerRoom4PerTimeSlot[:], 'heatingPowerRoom5': output_heatingPowerRoom5PerTimeSlot[:], 'COP': output_heatingEfficiencyCOPPerTimeSlot[:], 'Electricity Price': output_electricityTarifPerTimeSlot[:], 'Modulation Degree': output_modulationDegreeHeatPumpPerTimeSlot[:]})
                fileName = pathForTheRun + "/AdaptiveHysteresis_Week"+ str(week + 1)  + ".csv"
                df_results.to_csv(fileName, sep=';')

                #Print results
                result_averageThermalDiscomfort = round(output_scoreThermalDiscomfortPerTimeSlot.mean(axis=0),2)
                result_sumThermalDiscomfort = round(output_scoreThermalDiscomfortPerTimeSlot.sum(axis=0),2)
                result_sumCosts = round(output_electricityCostsPerTimeSlot.sum(axis=0),2)
                print("Results")
                print(f"Week: {week + 1}")
                print(f"result_averageThermalDiscomfort: {result_averageThermalDiscomfort}")
                print(f"result_sumCosts: {result_sumCosts}")

                df_results_weeks = df_results_weeks._append({"Week": week + 1, "Costs": result_sumCosts, "Average Thermal Discomfort": result_averageThermalDiscomfort}, ignore_index=True)



    df_results_weeks.to_excel(pathForTheRun + "\Results_Hysteresis.xlsx", index=False)





if __name__ == "__main__":
    main()

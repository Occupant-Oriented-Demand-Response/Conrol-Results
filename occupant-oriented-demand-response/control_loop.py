import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import signal

import src.utils.pickle as pkl
from config.definitions import FMU_DIR, FMU_NAME, MO_CLASS, MO_DIR, MO_NAME, ROOT_DIR
from src.control.mpc import MPC
from src.model.fmu import getVariableValues, setVariableValues, simulateStep, yieldFMU
from src.model.omc import getParameters, linearize, yieldOMC
from src.utils.dataframe import save


# helper function
def setStartingTemperatures(
    fmu, temperatureRoom1, temperatureRoom2, temperatureRoom3, temperatureRoom4, temperatureRoom5
):
    statesDict = {
        "Luft1.T": temperatureRoom1,
        "Luft2.T": temperatureRoom2,
        "Luft3.T": temperatureRoom3,
        "Luft4.T": temperatureRoom4,
        "Luft5.T": temperatureRoom5,
        "Wand1.T": temperatureRoom1,
        "Wand2.T": temperatureRoom2,
        "Wand3.T": temperatureRoom3,
        "Wand4.T": temperatureRoom4,
        "Wand5.T": temperatureRoom5,
    }
    setVariableValues(fmu, statesDict)


# helper function
def getTemperatures(
    fmu,
    temperatureOutside,
    solarRadiation,
    temperatureRoom1,
    temperatureRoom2,
    temperatureRoom3,
    temperatureRoom4,
    temperatureRoom5,
    hvacRoom1,
    hvacRoom2,
    hvacRoom3,
    hvacRoom4,
    hvacRoom5,
):
    initials = {
        "Ti1": temperatureRoom1,
        "Ti2": temperatureRoom2,
        "Ti3": temperatureRoom3,
        "Ti4": temperatureRoom4,
        "Ti5": temperatureRoom5,
    }
    inputs = {
        "Ta": temperatureOutside,
        "phis": solarRadiation,
        "HVAC1": hvacRoom1,
        "HVAC2": hvacRoom2,
        "HVAC3": hvacRoom3,
        "HVAC4": hvacRoom4,
        "HVAC5": hvacRoom5,
    }

    outputs = simulateStep(fmu, inputs, initialsDict=initials)
    return outputs["Ti1"], outputs["Ti2"], outputs["Ti3"], outputs["Ti4"], outputs["Ti5"]


# example of a control loop using the fmu model 576000 (32*900) | 547200 (64*900)
def control_loop(df, result_dir, start=0, stepSize=900, stop=547200+900, predictionHorizon=64*900):

    # load initial temperatures
    temperatureRoom1 = 26.1
    temperatureRoom2 = 26.1
    temperatureRoom3 = 26.1
    temperatureRoom4 = 26.1
    temperatureRoom5 = 26.1

    # load fmu
    with yieldFMU(os.path.join(FMU_DIR, FMU_NAME)) as fmu:
        # set initial temperatures
        setStartingTemperatures(
            fmu, temperatureRoom1, temperatureRoom2, temperatureRoom3, temperatureRoom4, temperatureRoom5
        )

        # additional state information (needs knowledge of internal model)
        states = []
        stateNames = [
            "Luft1.T",
            "Luft2.T",
            "Luft3.T",
            "Luft4.T",
            "Luft5.T",
            "Wand1.T",
            "Wand2.T",
            "Wand3.T",
            "Wand4.T",
            "Wand5.T",
        ]
        states.append(getVariableValues(fmu, stateNames))

        # prepare controller
        model_filepath = os.path.join(ROOT_DIR, "resources", "control", "state-space-model", "MultiZone.pkl")
        if not os.path.exists(model_filepath):
            with yieldOMC(MO_CLASS, os.path.join(MO_DIR, MO_NAME)) as omc:
                omc = linearize(omc, MO_CLASS)
                param = getParameters(omc, parameters=["A", "B", "C", "D"])
            sys = signal.StateSpace(
                np.matrix(param["A"]), np.matrix(param["B"]), np.matrix(param["C"]), np.matrix(param["D"])
            )
            pkl.save(sys, model_filepath)
        sys = pkl.load(model_filepath)
        mpc = MPC(sys, os.path.join(result_dir, "mpc"))

        # control loop
        results = []
        end = (int)(stop / stepSize)
        for time in range(start, end):

            # calculate inputs hvac1-5 here
            ###################################################
            x0 = np.array([states[-1][v] for v in stateNames])

            df_in = df.loc[time : time + int(predictionHorizon / stepSize)].reset_index()
            df_out = mpc.controlStep(predictionHorizon, x0, df_in, weight=0.5)
            
            ###################################################

            if time == end-1:
                for j in range(len(df_out.index)):
                    results.append(
                    {
                        "time": df["time"][time+j],
                        "temperature_ambient": df["temperature_ambient"][time+j],
                        "insolation_diffuse": df["insolation_diffuse"][time+j],
                        "electricity_price_day_ahead": df["electricity_price_day_ahead"][time+j],
                        "electrical_power": df_out["Pel"][j],
                        "temperature_room_1": df_out["Ti1"][j],
                        "temperature_room_2": df_out["Ti2"][j],
                        "temperature_room_3": df_out["Ti3"][j],
                        "temperature_room_4": df_out["Ti4"][j],
                        "temperature_room_5": df_out["Ti5"][j],
                        "hvac_room_1": df_out["Phih1"][j],
                        "hvac_room_2": df_out["Phih2"][j],
                        "hvac_room_3": df_out["Phih3"][j],
                        "hvac_room_3": df_out["Phih4"][j],
                        "hvac_room_4": df_out["Phih5"][j],
                    }
                )
            else:
                # save results here
                ###################################################
                results.append(
                    {
                        "time": df["time"][time],
                        "temperature_ambient": df["temperature_ambient"][time],
                        "insolation_diffuse": df["insolation_diffuse"][time],
                        "electricity_price_day_ahead": df["electricity_price_day_ahead"][time],
                        "electrical_power": df_out["Pel"][0],
                        "temperature_room_1": df_out["Ti1"][0],
                        "temperature_room_2": df_out["Ti2"][0],
                        "temperature_room_3": df_out["Ti3"][0],
                        "temperature_room_4": df_out["Ti4"][0],
                        "temperature_room_5": df_out["Ti5"][0],
                        "hvac_room_1": df_out["Phih1"][0],
                        "hvac_room_2": df_out["Phih2"][0],
                        "hvac_room_3": df_out["Phih3"][0],
                        "hvac_room_4": df_out["Phih4"][0],
                        "hvac_room_5": df_out["Phih5"][0],
                    }
                )
                ###################################################

                # evaluate model to determine new x0
                temperatureRoom1, temperatureRoom2, temperatureRoom3, temperatureRoom4, temperatureRoom5 = getTemperatures(
                    fmu,
                    df["temperature_ambient"][time],
                    df["insolation_diffuse"][time],
                    df_out["temperature_room_1"][0],
                    df_out["temperature_room_2"][0],
                    df_out["temperature_room_3"][0],
                    df_out["temperature_room_4"][0],
                    df_out["temperature_room_5"][0],
                    df_out["Phih1"][0],
                    df_out["Phih2"][0],
                    df_out["Phih3"][0],
                    df_out["Phih4"][0],
                    df_out["Phih5"][0],
                )

                
                states.append(getVariableValues(fmu, stateNames))
                ###################################################

    df_results = pd.DataFrame.from_records(results)
    save(df_results, os.path.join(result_dir, f"simulation_results.csv"))

def main():
    # run preprocessing
    from src.preprocessing.preprocessing import main as preprocessing

    # preprocessing()
    scenarios = ["base_scenario", "multizone_scenario"] #["adaptive_scenario", "base_scenario", "multizone_scenario"]
    for scenario in scenarios:
        print(f"Begin {scenario} simulation.")
        DATA_DIR = os.path.join(ROOT_DIR, "resources", "data", "control", f"{scenario}", "weeks")

        # select weeks
        weeks = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

        # try running control loop for each week
        for week in weeks:
            print(f"Starting with KW{week} ...")
            try:
                RESULT_DIR = os.path.join(
                        ROOT_DIR, "results", "mpc3", f"{scenario}", f"{week}", datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                    )
                df = pd.read_pickle(os.path.join(DATA_DIR, f"data_2022_KW{week}.pkl"))
                control_loop(df, RESULT_DIR)
                print(f"KW{week} done!")
            except:
                print(f"Error in KW{week}.")
        print(f"{scenario} simulatoin done!")


if __name__ == "__main__":
    main()

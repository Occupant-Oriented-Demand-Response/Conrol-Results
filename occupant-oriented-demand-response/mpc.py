import os
from datetime import datetime
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import src.utils.dataframe as util

class MPC:
    def __init__(
        self,
        sys,
        outputFolder,
        samplingRateSeconds=900,
        stateNames=["Ti1", "Ti2", "Ti3", "Ti4", "Ti5", "Tm1", "Tm2", "Tm3", "Tm4", "Tm5"],
        controlNames=["Phih1", "Phih2", "Phih3", "Phih4", "Phih5"],
        disturbanceNames=["temperature_ambient", "insolation_diffuse"],
        outputNames=["temperature_room_1", "temperature_room_2", "temperature_room_3", "temperature_room_4", "temperature_room_5"],
    ):
        self.sys = sys
        self.dsys = self.sys.to_discrete(samplingRateSeconds)
        assert self.dsys.A.shape[1] == len(stateNames)
        self.stateNames = stateNames
        assert self.dsys.B.shape[1] == len(controlNames) + len(disturbanceNames)
        self.controlNames = controlNames
        self.disturbanceNames = disturbanceNames
        assert self.dsys.C.shape[0] == len(outputNames)
        self.outputNames = outputNames
        self.folder = outputFolder

    def controlStep(self, predictionHorizonSeconds, x0, df_in, comfortRange=1.0, weight=0.5):
        assert 0 <= comfortRange
        assert 0 <= weight <= 1

        # prediction horizon steps
        N = (int)(predictionHorizonSeconds / self.dsys.dt)

        ## create new model
        m = gp.Model("MPC")
        m.ModelSense = GRB.MINIMIZE
        m.setParam(GRB.Param.PoolSolutions, 1)
        # m.setParam(GRB.Param.MIPGap, 10**-6)
        m.Params.LogToConsole = 0

        ## define variables

        # states
        x = m.addMVar(shape=(N + 1, len(self.stateNames)), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")

        statesDict = {}
        for i in range(len(self.stateNames)):
            statesDict[self.stateNames[i]] = x[:, i]

        # inputs
        u = m.addMVar(
            shape=(N, len(self.controlNames) + len(self.disturbanceNames)), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="u"
        )
        controlsDict = {}
        for i in range(len(self.controlNames)):
            controlsDict[self.controlNames[i]] = u[:, i]

        disturbancesDict = {}
        for i in range(len(self.disturbanceNames)):
            disturbancesDict[self.disturbanceNames[i]] = u[:, len(self.controlNames) + i]

        # outputs
        y = m.addMVar(shape=(N, len(self.outputNames)), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="y")

        outputsDict = {}
        for i in range(len(self.outputNames)):
            outputsDict[self.outputNames[i]] = y[:, i]

        # discomfort factor
        fdis = m.addMVar(shape=y.shape, lb=0, ub=GRB.INFINITY, name="fdis")

        fdisDict = {}
        for i in range(len(self.outputNames)):
            fdisDict[self.outputNames[i]] = fdis[:, i]

        # heat pump utilization factor
        futil = m.addMVar(shape=(N), lb=0, ub=1, name="futil")

        # electrical power of heat pump
        Pel = m.addMVar(shape=(N), lb=0, ub=GRB.INFINITY, name="Pel")
        # cooling power of heat pump
        Pcool = m.addMVar(shape=(N), lb=0, ub=GRB.INFINITY, name="Pcool")
        # heating/cooling power of hvac
        Phih = m.addMVar(shape=(N), lb=-GRB.INFINITY, ub=0, name="Phih")

        ## constraints

        # initial state
        for i in range(len(self.stateNames)):
            m.addConstr(x[0, i] == x0[i])

        for k in range(N):
            # system dynamics
            m.addConstr(x[k + 1, :] == self.dsys.A @ x[k, :] + self.dsys.B @ u[k, :])
            m.addConstr(y[k, :] == self.dsys.C @ x[k, :] + self.dsys.D @ u[k, :])

            # disturbances
            for n, v in disturbancesDict.items():
                m.addConstr(v[k] == df_in[n][k])

            # outputs/discomfort
            for n, v in outputsDict.items():
                m.addConstr(v[k] <= df_in[f"{n}_max"][k] + fdisDict[n][k])
                m.addConstr(v[k] >= df_in[f"{n}_min"][k] - fdisDict[n][k])

            # heat pump utilization
            m.addGenConstrPWL(futil[k], futil[k], [0, 0.2, 0.2, 1], [0, 0, 0.2, 1])

            # heat pump (cooling case)
            m.addConstr(Pel[k] == df_in["heat_pump_power_max"][k] * futil[k])
            m.addConstr(Pcool[k] == df_in["heat_pump_eer"][k] * Pel[k])
            m.addConstr(Phih[k] == -Pcool[k])

            # controls (cooling case)
            m.addConstr(Phih[k] == sum([v[k] for v in controlsDict.values()]))
            for v in controlsDict.values():
                m.addConstr(v[k] <= 0)

        # objectives
        obj1 = sum([v for vs in fdis.tolist() for v in vs])  # fdis.sum()
        obj2 = sum(Pel[k]/3500 * (df_in["electricity_price_day_ahead"][k]+6.9)/(44.3+6.9) for k in range(N))

        m.setObjectiveN(obj1, 1, weight=weight, name="comfort")
        m.setObjectiveN(obj2, 2, weight=(1 - weight), name="cost")

        # optimize
        m.optimize()

        df_out = pd.DataFrame()
        df_out["time"] = df_in["time"][0:N]        
        for k, v in controlsDict.items():
            df_out[k] = v.X
        for k, v in outputsDict.items():
            df_out[k] = v.X
        for k, v in fdisDict.items():
            df_out[f"fdis_{k}"] = v.X
        df_out["fdis"] = sum(fdis[:, k].X for k in range(len(self.outputNames)))
        df_out["futil"] = futil.X
        df_out["Pel"] = Pel.X
        df_out["Pcool"] = Pcool.X
        for k, v in statesDict.items():
            df_out[k] = v.X[0:N]
        

        m.terminate()

        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        util.save(df_out, os.path.join(self.folder, f"results_mpc_{timestamp}.pkl"))

        return df_out

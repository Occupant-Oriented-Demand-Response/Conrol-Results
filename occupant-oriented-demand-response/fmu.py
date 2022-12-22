import os
import shutil
from random import randint
from contextlib import contextmanager

from fmpy import extract, read_model_description
from fmpy.fmi2 import FMU2Slave


def __initialize(filepath, instance=None):
    description = read_model_description(filepath)

    if not instance:
        instance = f"fmu-temp-{randint(0, 99999):05d}"

    dir, filename = os.path.split(filepath)

    unzipdir = extract(
        filepath,
        os.path.join(dir, instance),
    )
    fmu = FMU2Slave(
        guid=description.guid,
        unzipDirectory=unzipdir,
        modelIdentifier=description.coSimulation.modelIdentifier,
        instanceName=instance,
    )
    fmu.instantiate()
    fmu.setupExperiment()
    fmu.enterInitializationMode()
    fmu.exitInitializationMode()
    return fmu


def __cleanup(fmu):
    if fmu:
        fmu.terminate()
        fmu.freeInstance()
        shutil.rmtree(fmu.unzipDirectory, ignore_errors=True)


@contextmanager
def yieldFMU(filepath, instance=None):
    # Before yield as the enter method
    fmu = __initialize(filepath, instance)
    yield fmu
    # After yield as the exit method
    __cleanup(fmu)


def _getInputs(fmu):
    inputs = [v for v in read_model_description(fmu.unzipDirectory).modelVariables if v.causality == "input"]
    return inputs


def _getOutputs(fmu):
    outputs = [v for v in read_model_description(fmu.unzipDirectory).modelVariables if v.causality == "output"]
    return outputs


def _getVariables(fmu, names):
    variables = [v for v in read_model_description(fmu.unzipDirectory).modelVariables if v.name in names]
    return variables


def getVariableNames(fmu):
    names = [v.name for v in read_model_description(fmu.unzipDirectory).modelVariables]
    return names


def getVariableValues(fmu, names):
    valuesDict = {}
    for v in _getVariables(fmu, names):
        valuesDict[v.name] = fmu.getReal([v.valueReference])[0]
    return valuesDict


def setVariableValues(fmu, variablesDict):
    for v in _getVariables(fmu, variablesDict.keys()):
        fmu.setReal([v.valueReference], [variablesDict[v.name]])


def simulateStep(fmu, inputsDict, initialsDict=None, time=0, step_size=900):
    if initialsDict:
        for v in _getOutputs(fmu):
            fmu.setReal([v.valueReference], [initialsDict[v.name]])
    for v in _getInputs(fmu):
        fmu.setReal([v.valueReference], [inputsDict[v.name]])
    fmu.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)
    outputsDict = {}
    for v in _getOutputs(fmu):
        outputsDict[v.name] = fmu.getReal([v.valueReference])[0]
    return outputsDict

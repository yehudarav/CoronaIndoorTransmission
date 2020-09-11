import pandas
import numpy
from .person import getPersonClass
from .room import getRoomClass

from .person import SUSCEPTIBLE,EXPOSED,INFECTED,RECOVERED
from unum.units import *
from . import ml


def getModelClass(JSON):
    simType = JSON['simulation']['numericalMethod']
    clsName = f"singleRoomEnvironmentCloseContant_{simType}"
    return globals()[clsName]

class Model(object):
    """
        Basic model class.
    """
    _simulationStart = None
    random = None

    _locations = None
    _agentList = None
    _settings  = None

    @property
    def simulationStart(self):
        return self._simulationStart

    _currentTime = None

    @property
    def getCurrentDatetime(self):
        return self._currentTime


    @property
    def settings(self):
        return self._settings


    @property
    def dt_base(self):
        return self.settings["simulation"]["dt"]

    @property
    def dt_datetime_base(self):
        return pandas.to_timedelta("%sm" % self.dt_base.asNumber(min), unit='ms')


    _dt = None
    @property
    def dt(self):
        return self._dt

    def __init__(self,JSON,randomSeed):

        self._agentList = []
        self._settings  = self._ConvertJSON_to_conf(JSON)
        self._locations = {}

        self._simulationStart  = pandas.to_datetime(pandas.to_datetime("1/1/2020 00:00"), unit='ms')
        self._currentTime = self._simulationStart

        self._dt = self.dt_base
        self._dt_datetime = self.dt_datetime_base

        try:
            self.random = numpy.random.RandomState(randomSeed)
        except ValueError as e:
            print(e)
            print("Error seed %s too large " % randomSeed)

    def _ConvertJSON_to_conf(self,JSON):
        """
            Traverse the JSON and replace all the unum values with objects.

        :param JSON:
        :return:
        """
        ret ={}
        for key,value in JSON.items():
            if isinstance(value,dict):
                ret[key] = self._ConvertJSON_to_conf(JSON[key])
            elif isinstance(value,list):
                ret[key] = value
            else:
                try:
                    ret[key] = eval(str(value))
                except NameError:
                    ret[key] = value

        return ret

    def addAgent(self,agent):
        self._agentList.append(agent)

    @property
    def agents(self):
        return self._agentList

    def addLocation(self,locationRef):
        self._locations[locationRef.unique_id] = locationRef

    def getLocation(self,name):
        return self._locations[name]


class singleRoomEnvironmentCloseContant(Model):
    """
        Simulation of 2 agents, primary and secondary.
        The primary begins as exposed and the secondary as susceptible.
    """
    _room = None
    _primary = None
    _secondary = None


    @property
    def terminatePrimaryInfected(self):
        return self.settings["simulation"]["terminatePrimaryInfected"]

    @property
    def room(self):
        return self._locations["room"]

    @property
    def primary(self):
        return self._primary

    @property
    def secondary(self):
        return self._secondary


    def __init__(self,JSON,randomSeed):
        """
            Initializes a singleRoom environment.

            The config has the followind structure:
            {
                "simulation" : {
                    "dt" : 30*min,
                },
                "person" : {
                    "type" : "simple",
                    "shedvirus" : {
                            "name" : "simple",
                            "params" : {
                                    "expulsionVolume" : 5*ml,
                                    "expulsionFrequency" : 2/h
                            }
                    },
                    "incubation" : {
                            "name" : "lognormal",
                            "params" : {
                                    "mean" : 5*d,
                                    "std"  : 0.5
                            }
                    },
                    "doseresponse" : {
                            "name" : "exp",
                            "mode" : "currentExposure",
                            "params" : {
                                    "k" : 410
                            }

                    },
                    "maxviralload" : {
                            "name" : "const",
                            "params" : {
                                    "value" : 1e6/ml
                            }
                    },
                    "exposuredecontamination" : {
                            "name" : "clearPeriod",
                            "params" : {
                                    "period" : 6*h
                            }

                    }

                },
                "room" : {
                    "type" : "simple",
                    "decayrate" : 0/h
                }
            }

        :param config:
                JSON config. In JSON, all the unum objects (objects with units) are string.

        """
        super().__init__(JSON,randomSeed)
        numericalMethod = self.settings["simulation"]["numericalMethod"]
        PersonClass = getPersonClass(f"Agent{numericalMethod}")
        RoomClass   = getRoomClass(f"Agent{numericalMethod}")

        room      = RoomClass("room",self)
        self._primary   = PersonClass("primary",self,startState=EXPOSED)
        self._secondary = PersonClass("secondary",self)

        self.addLocation(room)

        self.room.enterRoom(self.primary)
        self.room.enterRoom(self.secondary)

        self.addAgent(self.primary)
        self.addAgent(self.secondary)
        self.addAgent(self.room)


    def step(self):
        raise NotImplementedError("Implement in specialized class")

    def _update_dt(self,new_dt):
        self._dt = new_dt.total_seconds()*s
        self._dt_datetime = new_dt

    def runSimulation(self,terminatePrimaryInfected=True):
        """
            Running until primary is infected or the secondary is exposed

        :return:
        """
        running = True
        while (running):
            self.step()
            if self.secondary.currentState == EXPOSED:
                running = False

            if (self.getCurrentDatetime > self.primary.incubationEnd) and (self.primary.viralLoad.asNumber(1 / ml) < 1):
                running = False

            if terminatePrimaryInfected:
                if self.primary.currentState == INFECTED:
                    running = False
            else:
                if self.primary.currentState == RECOVERED:
                    running = False



class singleRoomEnvironmentCloseContant_EquiDistance(singleRoomEnvironmentCloseContant):
    """
        Simulation of 2 agents, primary and secondary.
        The primary begins as exposed and the secondary as susceptible.
    """

    def __init__(self,JSON,randomSeed):
        super().__init__(JSON,randomSeed)
        speriod = pandas.to_timedelta("%sd" % self.primary.sicknessPeriod.asNumber(d))
        totalsimulation = ((self.primary.incubationEnd+speriod)-self.getCurrentDatetime ).total_seconds()*s
        self.fillAllEvents(totalsimulation)

    def fillAllEvents(self,totalsimulation):
        self.primary.fillEvents(self.getCurrentDatetime,totalsimulation)
        self.secondary.fillEvents(self.getCurrentDatetime,totalsimulation)
        self.room.fillEvents(self.getCurrentDatetime,totalsimulation)

    def step(self):
        upcommingEventDate = None
        for agent in self.agents:
            agent.handle_event()


        for agent in self.agents:
            agent.step()
            if agent.upcomingEvent is not None:
                if upcommingEventDate is None:
                    upcommingEventDate = agent.upcomingEvent["date"]
                elif agent.upcomingEvent["date"] < upcommingEventDate :
                    upcommingEventDate = agent.upcomingEvent["date"]


        if upcommingEventDate  is not None:
            dt_diff = upcommingEventDate - self.getCurrentDatetime
            if dt_diff < self._dt_datetime:
                self._update_dt(dt_diff)
            else:
                self._update_dt(self.dt_datetime_base)
        else:
            self._update_dt(self.dt_datetime_base)

        self._currentTime += self._dt_datetime

class singleRoomEnvironmentCloseContant_Events(singleRoomEnvironmentCloseContant):
    """
        Simulation of 2 agents, primary and secondary.
        The primary begins as exposed and the secondary as susceptible.
    """
    def step(self):
        for agent in self.agents:
            agent.handle_event()

        for agent in self.agents:
            agent.step()

        self._currentTime += self._dt_datetime

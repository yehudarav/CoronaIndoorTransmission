import pandas
import numpy
from random import Random
from unum import Unum
from unum.units import *

class Agent(object):
    _history = None
    _loggingFields = None

    _fieldChange = None # holds fields names and their change (after *dt).

    _agentType = None

    model = None

    @property
    def settings(self):
        return self.model.settings[self._agentType]

    @property
    def agentType(self):
        return self._agentType

    def __init__(self, unique_id, model,agentType, loggingFields=[]):
        """
            Creates a new agent.

        :param unique_id: str
                The name of the agent
        :param model: model
                ref to the model
        :param agentType: str
                type of agent. In single room simulation can be 'person' or 'room'.
                In the aircabin can be 'passenger','seatingCompartment','aisle','kitchen','toilets',
                'airCrew'.

        :param loggingFields: list
                The list of parameters to hold the history for.
        """

        self.unique_id = unique_id
        self.model = model
        self._history = []
        self._loggingFields = loggingFields
        self._agentType = agentType
        self._fieldChange = {}

    def history(self,unitless=False):
        """
            If true, remove the units.
        :param unitless:
        :return:
            pandas.DataFrame
        """
        pandasData = pandas.DataFrame(self._history)
        if unitless:
            unitless_pandasData = pandas.DataFrame()
            for col in pandasData.columns:
                if isinstance(pandasData[col].iloc[0],Unum):
                    unitless_pandasData = unitless_pandasData.assign(**{col : [x.asNumber() for x in pandasData[col]]})
                else:
                    unitless_pandasData[col]=pandasData[col]

            pandasData = unitless_pandasData
        return pandasData

    @property
    def hisoryUnits(self):
        """
            Return a JSON with field name and the unit (or type) of the columns.
        :return:
           dict
        """
        pandasData = pandas.DataFrame(self._history)
        ret = {}
        for col in pandasData.columns:
            if isinstance(pandasData[col].iloc[0], Unum):
                ret[col] = pandasData[col].iloc[0].strUnit()

        return ret

    @property
    def random(self) -> Random:
        return self.model.random

    def addLoggingFields(self,fieldNameList):
        self._loggingFields += numpy.atleast_1d(fieldNameList)


    def collect(self):
        ret = dict(AgentID=self.unique_id,date=self.model.getCurrentDatetime)
        for field in self._loggingFields:
            ret[field] = getattr(self,field)
        self._history.append(ret)

    @property
    def currentState(self):
        return None


    def getActionList(self,state=None):
        """
            Return the action list of the agent.
            may depend on the compartment and the state of the person.

        :param state:
        :return:
        """
        theState = self.currentState if state is None else state

        actionList = []
        for event, eventData in self.settings["actions"].items():
            name = event
            freq = eventData["frequency"][theState] if isinstance(eventData["frequency"], dict) \
                else eventData["frequency"]

            actionList.append(dict(name=name, frequency=freq))

        return actionList

    def handle_event(self):
        """
            Check if event takes place, and if it did, handle it.
        :return:
        """
        raise NotImplementedError("Implement in child")

    def func_lognormal(self,mean,std):
        """
            return incubation time that distributes lognormally
        :param self:
        :param mean:
                the mean incubation time.
        :param gstd:
                the std
        :return:
                datetime.timedelat.
        """
        mead_day = mean.asNumber(d)
        inc = self.random.lognormal(numpy.log(mead_day),std)
        return pandas.to_timedelta("%sd" % inc)

    def func_const(self,const):
        return const

    def func_const_td(self,const):
        return pandas.to_timedelta(f"{const}d")

class AgentEquiDistance(Agent):
    """ Base class for a model agent. """

    _eventList = None
    _upcomingEvent = None
    _passedEvents  = None

    @property
    def upcomingEvent(self):
        return self._upcomingEvent

    @property
    def passedEvents(self):
        return self._passedEvents

    def __init__(self, unique_id, model,agentType, loggingFields=[]):
        """ Create a new agent. """
        super().__init__(unique_id,model,agentType,loggingFields)
        self._eventList = []
        self._passedEvents = []

    def step(self):
        """ A single step of the agent. """
        raise NotImplementedError("implement in son")

    def fillEvents(self, fromTime, totalTime, actionList=None, agentState=None):
        """
            Fill in the events according tot he action list.

            The number of events is determined from poisson distribution
            and spread equally over time.

        :param actionList: list
                The list of actions.
                each action is defined by name and by frequency.

                dict(name=..., frequency=...)

        :param SimulationStartTime: datetime
                The date of the begining of the simulation.
        :param TotalSimulationTime: unum
                The total time of the simulation.
        :return:
        """

        cActionList = self.getActionList(agentState)

        actionList = cActionList if actionList is None else cActionList+actionList

        for action in actionList:
            eventsFrequency = action["frequency"]
            events = self.random.poisson((totalTime * eventsFrequency).asNumber())
            eventsTimeDelta =  pandas.to_timedelta("%sm" % totalTime.asNumber(min)) / (events + 1)
            for eventIndex in range(1,events+1):
                self._eventList.append(dict(
                                        date=fromTime + eventIndex * eventsTimeDelta,
                                        name= action['name'],
                                        agent=self.unique_id
                                    )
                                 )
        self._eventList.sort(key=lambda x: x['date'])

    def handle_event(self):
        loopEvent = True
        while loopEvent:
            if self.upcomingEvent is None:
                if len(self._eventList) >0:
                    self._upcomingEvent = self._eventList.pop(0)
                else:
                    loopEvent = False
            elif self.upcomingEvent["date"] == self.model.getCurrentDatetime:

                fname = f"event_{self.upcomingEvent['name']}"
                self._fieldChange[fname] = self._fieldChange.get(fname,0)+1

                getattr(self,"_event_handle_%s" % self.upcomingEvent['name'])()
                self._passedEvents.append(self.upcomingEvent)
                if len(self._eventList) >0:
                    self._upcomingEvent = self._eventList.pop(0)
                else:
                    self._upcomingEvent = None
            else:
                loopEvent = False


class AgentEvents(Agent):
    """
        This implementation randomize the events every time step.

    """

    def __init__(self, unique_id, model,agentType, loggingFields=[]):
        """ Create a new agent. """
        super().__init__(unique_id,model,agentType,loggingFields)


    def handle_event(self):
        """
            random a poisson event to see if the event takes place.
        :return:
        """

        cActionList = self.getActionList()

        # get the currect action list (according to the time).
        #actionList = [x for x in cActionList if (x['fromTime']<=self.model.getCurrentDatetime and x['toTime']>=self.model.getCurrentDatetime)][0]
        for action in cActionList: #['actions']:
            eventsFrequency = action["frequency"]
            events = self.random.poisson((self.model.dt * eventsFrequency).asNumber())

            if events > 0:
                fname = f"event_{action['name']}"
                self._fieldChange[fname] = self._fieldChange.get(fname,0)+1
                getattr(self, "_event_handle_%s" % action['name'])()

                #if action['name'] =='washHands' and self.unique_id=='primary':
                #    print(events,action,self._fieldChange.get(fname,0))


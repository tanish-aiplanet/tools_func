import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from openagi.llms.base import LLMBaseModel
from openagi.queue.message_broker import NameIndexMapper
from openagi.queue.pq import Message, MessageType, MultiThreadPriorityQueue
from openagi.tools.integrations.duckducksearch import getDuckduckgoSearchResults
from openagi.tools.tools_db import (
    TOOLS_DICT_MAPPING,
)
from openagi.tools.utils import search_string_in_list, isLastHGI
from openagi.utils.llmTasks import handleLLMTask, tools_handler, handleReactTask

g_mapper = None
g_timerlist = None
main_condition = threading.Condition()


def createDynamicAgent(agent, *functions):
    """
    Dynamically creates an agent by calling the provided `startAgent` functions and passing the `agent` and `mapper` arguments.

    Args:
        agent (object): The agent object to be created.
        *functions (tuple): A variable number of tuples, where each tuple contains a `startAgent` function and a `mapper` object.

    Returns:
        None
    """
    for startAgent, mapper in functions:
        startAgent(agent, mapper)


def setGMapper(mapper):
    """
    Sets the global `g_mapper` variable to the provided `mapper` object.

    This function is used to initialize the global `g_mapper` variable, which is likely used throughout the codebase. It logs the contents of the `mapper` object's dictionary for debugging purposes.
    """
    global g_mapper
    g_mapper = mapper
    logging.debug(f"mapper object = {g_mapper.__dict__}")


def getGMapper():
    """
    Returns the global `g_mapper` object.

    This function is used to access the global `g_mapper` object, which is likely used throughout the codebase. It logs the contents of the `g_mapper` object's dictionary for debugging purposes.

    Returns:
        object: The global `g_mapper` object.
    """
    logging.debug(f"mapper object from get = {g_mapper.__dict__}")
    return g_mapper


def setGTimerList(timerPool):
    """
    Sets the global `g_timerlist` variable to the provided `timerPool` object.

    This function is used to initialize the global `g_timerlist` variable, which is likely used throughout the codebase.
    """
    global g_timerlist
    g_timerlist = timerPool


def getGTimerList(timerPool):
    """
    Returns the global `g_timerlist` object.

    This function is used to access the global `g_timerlist` object, which is likely used throughout the codebase.

    Returns:
        object: The global `g_timerlist` object.
    """
    return g_timerlist


def waitonConditionAgent(mapper, agentName, duration):
    """
    Waits on a condition for the specified agent.

    This function is used to wait on a condition for the specified agent. It retrieves the index of the agent in the `mapper.COND_LIST`, acquires the condition, waits for the specified duration, and then releases the condition.

    Args:
        mapper (object): The mapper object.
        agentName (str): The name of the agent to wait on.
        duration (float): The duration to wait on the condition.
    """
    lindex = mapper.get_index_by_name(agentName)
    logging.debug(f"index of {agentName} is {lindex}")
    cond = mapper.COND_LIST[lindex]
    cond.acquire()
    cond.wait(timeout=duration)
    cond.release()


def wakeUpAgent(mapper, agentName):
    """
    Wakes up the specified agent by notifying its condition variable.

    This function is used to wake up an agent by acquiring its condition variable, notifying it, and then releasing the condition variable. This allows the agent to continue executing after being paused or waiting on the condition.

    Args:
        mapper (object): The mapper object.
        agentName (str): The name of the agent to wake up.
    """
    lindex = mapper.get_index_by_name(agentName)
    cond = mapper.COND_LIST[lindex]
    cond.acquire()
    cond.notify()
    cond.release()


def waitonConditionMain(condition, duration):
    """
    Waits on the specified condition variable for the given duration.

    This function is used to wait on a condition variable for a specified duration of time. It first acquires the condition variable, then waits on it with the given timeout, and finally releases the condition variable.

    Args:
        condition (threading.Condition): The condition variable to wait on.
        duration (float): The duration in seconds to wait on the condition variable.
    """
    condition.acquire()
    condition.wait(timeout=duration)
    condition.release()


def wakeupMainForKill(condition):
    """
    Wakes up the main thread by notifying its condition variable.

    This function is used to wake up the main thread that may be waiting on a condition variable. It acquires the condition variable, notifies it, and then releases the condition variable. This allows the main thread to continue executing after being paused or waiting on the condition.

    Args:
        condition (threading.Condition): The condition variable to notify.
    """
    condition.acquire()
    condition.notify()
    condition.release()


def exitOpenAGI():
    """
    Wakes up the main thread by notifying its condition variable.

    This function is used to wake up the main thread that may be waiting on a condition variable. It acquires the condition variable, notifies it, and then releases the condition variable. This allows the main thread to continue executing after being paused or waiting on the condition.

    Args:
        condition (threading.Condition): The condition variable to notify.
    """
    wakeupMainForKill(main_condition)


def createAndsendMessage(srcAgent, dstAgent, body, mapper, consumerList):
    """
    Sends a message to one or more destination agents.

    This function creates a new message with the given parameters and inserts it into the priority queue of each destination agent. It then wakes up the destination agents to process the new message.

    Args:
        srcAgent (str): The name of the source agent sending the message.
        dstAgent (str): The name of the destination agent(s) to receive the message.
        body (str): The body or content of the message.
        mapper (Mapper): The mapper object used to manage agent information.
        consumerList (list[str]): A list of destination agent names to send the message to.
    """
    for dstAgent in consumerList:
        priority = 0
        perception_type = 0
        message_type = MessageType.AGENT_MSG
        body = body
        message = Message(
            priority, perception_type, message_type, srcAgent, dstAgent, body
        )
        pq = mapper.get_PQ_by_name(dstAgent)
        pq.insert(message)
        wakeUpAgent(mapper, dstAgent)


def createAndsendMessageProfTrigger(srcAgent, dstAgent, body, mapper):
    """
    Sends a message to a destination agent with a specific trigger type.

    This function creates a new message with the given parameters and inserts it into the priority queue of the destination agent. It then wakes up the destination agent to process the new message. The message has a specific trigger type of MessageType.PROF_AGENT_TRIGGER.

    Args:
        srcAgent (str): The name of the source agent sending the message.
        dstAgent (str): The name of the destination agent to receive the message.
        body (str): The body or content of the message.
        mapper (Mapper): The mapper object used to manage agent information.
    """
    priority = 0
    perception_type = 0
    message_type = MessageType.PROF_AGENT_TRIGGER
    body = body
    message = Message(priority, perception_type, message_type, srcAgent, dstAgent, body)
    pq = mapper.get_PQ_by_name(dstAgent)
    time.sleep(5)
    pq.insert(message)
    wakeUpAgent(mapper, dstAgent)


def evalLLMResponse(resp):
    """
    Evaluates the sentiment of an LLM (Large Language Model) response.

    This function takes an LLM response as input, processes it using the spaCy NLP library, and determines the sentiment polarity of the response. If the polarity is positive, the function returns True, otherwise it returns False.

    Args:
        resp (str): The LLM response to be evaluated.

    Returns:
        bool: True if the sentiment polarity of the response is positive, False otherwise.
    """
    print(f"evalLLMResponse::{resp}")
    nlp = spacy.load(
        "en_core_web_sm"
    )  # Run this command if you face error: `python -m spacy download en_core_web_sm`
    nlp.add_pipe("spacytextblob")
    doc = nlp(resp)
    polarity = doc._.blob.polarity
    print(f"sentiment {polarity}")
    if polarity > 0:
        return True
    return False


def createAndSendFeedbackMessage(srcAgent, dstAgent, body, mapper, feedbackSummary):
    """
    Sends a feedback message to a destination agent.

    This function creates a new message with the given parameters and inserts it into the priority queue of the destination agent. It then wakes up the destination agent to process the new message. The message has a specific message type of MessageType.FEEDBACK_MSG.

    Args:
        srcAgent (str): The name of the source agent sending the message.
        dstAgent (str): The name of the destination agent to receive the message.
        body (str): The body or content of the message.
        mapper (Mapper): The mapper object used to manage agent information.
        feedbackSummary (str): The summary of the feedback to be included in the message.
    """
    priority = 0
    perception_type = 0
    message_type = MessageType.FEEDBACK_MSG
    final_reponse = "revise" + body + "based on feedback " + body
    logging.debug(
        f"the body of feedback messgae...............................{final_reponse}",
    )
    message = Message(
        priority, perception_type, message_type, srcAgent, dstAgent, final_reponse
    )
    if not dstAgent == "profAgent":
        pq = mapper.get_PQ_by_name(dstAgent)
        time.sleep(10)
        pq.insert(message)
        wakeUpAgent(mapper, dstAgent)
        return
    else:
        logging.debug(f"response is going to profAgent f{message}")
        return


def example_callbackTimer(dstAgent, timerName, timerValue):
    """
    Executes a callback function when a timer expires.

    This function is called when a timer set for the specified `dstAgent` and `timerName` expires. It creates a new `Message` object with the `MessageType.TIMER_MSG` type and inserts it into the priority queue of the destination agent. It then wakes up the destination agent to process the new message.

    Args:
        dstAgent (str): The name of the destination agent to receive the timer message.
        timerName (str): The name of the timer that has expired.
        timerValue (float): The value of the timer that has expired.
    """
    logging.debug(f"Callback executed at {dstAgent}:{timerName}:{timerValue}")
    priority = 0
    perception_type = 0
    message_type = MessageType.TIMER_MSG
    body = "This is time out message"
    message = Message(
        priority, perception_type, message_type, "TIMER_AGENT", dstAgent, body
    )
    mapper = getGMapper()
    pq = mapper.get_PQ_by_name(dstAgent)
    pq.insert(message)
    logging.debug(f"timer message is inserted for {dstAgent}  {mapper.__dict__}")
    wakeUpAgent(mapper, dstAgent)


def sendMessagetoHGI(consumer, resp):
    """
    Sends a message to the HGI (Home Gateway Interface) and logs the final message.

    Args:
        consumer (str): The name of the consumer receiving the message.
        resp (str): The final message to be sent to the HGI.

    Returns:
        bool: True if this is the last HGI message, False otherwise.
    """
    logging.debug(f"final message to {consumer}= {resp}")
    print(f"final OUTPUT for {consumer} ::{resp}")
    return isLastHGI()


class Agent:
    """
    The `Agent` class represents an agent in a multi-agent system. It is responsible for executing tasks, formatting input and output data, and managing the communication between agents.

    The `__init__` method initializes the agent with various parameters such as the agent's name, role, goal, backstory, capability, task, output consumer agents, LLM API, tools list, feedback flag, agent type, multiplicity, aggregator, on-aggregation action, creator, HGI interface, LLM response timer value, and LLM model.

    The `format_input` and `format_output` methods are placeholders for implementing the logic to format input and output data, respectively.

    The `execute` method is the main entry point for executing the agent's task. It handles the aggregation of messages, uses tools or LLM models to perform the task, and sends the response to the appropriate consumer agents.

    The `run` method is the main loop of the agent, where it continuously retrieves messages from the priority queue, processes them, and sends the responses to the consumer agents.

    The `start_agent` method is used to start the agent in a new thread.
    """

    def __init__(
        self,
        agentName,
        role,
        goal,
        backstory,
        capability,
        task,
        output_consumer_agent=[],
        llm_api=None,
        tools_list=[],
        feedback=False,
        agent_type="STATIC",
        multiplicity=0,
        aggregator=0,
        onAggregationAction=None,
        creator=None,
        HGI_Intf=None,
        llm_resp_timer_value=20000,
        llm=None,
    ):
        self.agentName = agentName
        self.aggregator = aggregator
        self.onAggregationAction = onAggregationAction
        self.creator = creator
        self.createAgentFlag = 0
        self.pq = MultiThreadPriorityQueue()
        self.condition = threading.Condition()
        self.is_data_found = False
        self.executor = ThreadPoolExecutor(
            max_workers=2
        )  # Adjust max_workers based on your requirement
        self.agent_thread = []
        self.mapper = []
        self.execute_task = "perform the task"
        self.consumerAgent = output_consumer_agent[0]
        self.consumerAgentList = output_consumer_agent
        self.timerNameLLM = "LLM_TIMER"
        self.NoOfFeedbackMsgs = 0
        self.MAX_NoOfFeedbackMsgs = 1
        self.role = role
        self.feedback = feedback
        self.goal = goal
        self.backstory = backstory
        self.capability = capability
        self.agent_type = agent_type
        self.multiplicity = multiplicity
        self.task = task
        self.output_consumer_agent = output_consumer_agent
        self.HGI_Intf = HGI_Intf
        self.llm_api = llm_api
        self.timerValueLLM = llm_resp_timer_value
        self.tools_list = tools_list
        self.FeedBackMsgReceived = False
        self.llm: LLMBaseModel = llm

    def format_input(self, input_data):
        # Implement the logic to format input data
        pass

    def format_output(self, output_data):
        # Implement the logic to format output data
        pass

    def execute(self, messageList):
        """
        Executes the logic for the agent's task, including handling messages, using tools, and providing feedback.

        Args:
            messageList (list): A list of messages to be processed.

        Returns:
            str: The response generated by the agent.
        """
        # Implement the logic for the execution of the task
        message = messageList[0]
        if self.aggregator > 0:
            # overwrite the body of the first message with aggregated content
            message.body = self.onAggregationAction(
                self.llm, self.consumerAgent, None, messageList
            )

        consumer = self.consumerAgent
        resp = f"sending mesage from {self.agentName}"
        if self.tools_list:
            logging.debug("Using tools")
            _tools = []
            for user_tool in self.tools_list:
                tool_name = user_tool.name  # Assuming user_tool has a name attribute
                if tool_name in TOOLS_DICT_MAPPING:
                    # If the tool exists in TOOLS_DICT, append its info
                    _tools.append(TOOLS_DICT_MAPPING[tool_name])
                else:
                    # If the tool is not found, append a custom tool dictionary
                    _tools.append(
                        {
                            "category": "custom",
                            **user_tool.get_tool_info(),
                        }
                    )

            if self.tools_list:
                print(f"\n      role being performed is { self.role } \n       ")
                resp = tools_handler(
                    tools=_tools,
                    task_input={
                        "role": self.role,
                        "name": self.agentName,
                        "backstory": self.backstory,
                        "task": self.task,
                        "goal": self.goal,
                        "capability": self.capability,
                        "instructions": self.task,  # TODO: To be Removed or fixed as its redundant
                    },
                    llm=self.llm,
                )
                logging.info(f"resp from tools_handler : {resp}")

        elif self.capability == "search_executor":
            indices = search_string_in_list(self.tools_list, "duckduckgo-search")
            if indices:
                results = getDuckduckgoSearchResults(self.goal + self.task)
                print(f"executed duckduck search and results :{results}")
                resp = f" {results}"
        elif (
            self.capability == "llm_task_executor"
        ):  # and (self.role=="SUMMARISER" or self.role=="EMAILER" or self.role=="Coder" or self.role=="Reviewer")):
            print(f"\n      role being performed is { self.role } \n       ")
            if not self.feedback or self.NoOfFeedbackMsgs < self.MAX_NoOfFeedbackMsgs:
                resp = handleLLMTask(
                    message.body,
                    self.role,
                    self.backstory,
                    self.goal,
                    self.task,
                    self.llm,
                )
            else:
                logging.info("response for  final feedback ")
                resp = message.body

        elif self.capability == "llm_react_executor":
            print(f"\n      React Agent is executing: { self.role } \n       ")
            if not self.feedback or self.NoOfFeedbackMsgs < self.MAX_NoOfFeedbackMsgs:
                resp = handleReactTask(
                    self.task,
                    self.llm,
                    message.body,
                    # in future need to add other parameters too.
                )
            else:
                logging.info("response for  final feedback ")
                resp = message.body

        logging.debug(f"response of agent {self.agentName} is { resp}")
        # STOP THE TIMER
        self.mapper.timerStopMapper(
            agentName=self.agentName, timerName=self.timerNameLLM
        )
        # feedback is supported for only non aggregator agents
        if (
            self.feedback
            and self.aggregator == 0
            and self.NoOfFeedbackMsgs < self.MAX_NoOfFeedbackMsgs
        ):
            status = evalLLMResponse(resp)
            if not status:
                self.NoOfFeedbackMsgs += 1
                createAndSendFeedbackMessage(
                    self.agentName, message.srcAgent, message.body, self.mapper, resp
                )
                return
            else:
                logging.debug("feedback on message is positive")
                resp = message.body  # update the body with  original of the agent

        if self.HGI_Intf:
            logging.debug("...............executing human tool.....................\n")
            resp, feedback, actionself = self.HGI_Intf(
                self.agentName, resp, self.consumerAgent
            )
            logging.debug(f" the response from human tool:: {resp}")

        if not consumer or consumer == "HGI":
            if sendMessagetoHGI(consumer, resp):
                logging.debug("before wake for exit..........")
                wakeupMainForKill(main_condition)

        if consumer and consumer != "HGI":
            logging.debug(f"{self.agentName}..to.. {consumer} content:: {resp}")
            createAndsendMessage(
                self.agentName, consumer, resp, self.mapper, self.consumerAgentList
            )

    def run(self):
        """
        Runs the agent's main loop, retrieving messages from the priority queue, executing them, and handling feedback and human-in-the-loop interactions.

        The agent retrieves messages from the priority queue, appends them to a message list, and executes them when the list reaches the configured aggregator size or the aggregator is set to 0. If the agent is a creator agent, it creates a dynamic agent before executing the messages.

        The agent also handles feedback and human-in-the-loop interactions, updating the response based on the feedback status and executing any human tools if configured.

        Finally, the agent sends the response to the appropriate consumer, either the HGI or another agent.
        """
        myname = self.agentName
        logging.debug(f"I am agent {myname}")
        multiplicity = self.multiplicity
        msgList = []
        while True:
            message = self.pq.retrieve_first_message()
            if message:
                msgList.append(message)
                logging.debug(f"agent {myname} retrieved message: {message.__dict__}")
                # print(message.__dict__)
                logging.debug("appended received message to list for aggregation")
                if len(msgList) == self.aggregator or self.aggregator == 0:
                    if self.creator and self.createAgentFlag == 0:
                        logging.debug(
                            f"creating dynamic agent {self.consumerAgent} in {self.agentName}",
                        )
                        functions_to_execute = [(Agent.start_agent, self.mapper)]
                        createDynamicAgent(self.creator, *functions_to_execute)
                        self.createAgentFlag = 1
                    self.mapper.timerStartMapper(
                        agentName=self.agentName,
                        timerName=self.timerNameLLM,
                        callback=example_callbackTimer,
                        timervalue=self.timerValueLLM,
                    )
                    self.execute(msgList)
                    msgList = []
            else:
                logging.debug(
                    f"{myname} agent found no messages from PQ .. going to wait."
                )
                waitonConditionAgent(self.mapper, self.agentName, 100)

    def start_agent(self, mapper):
        """
        Starts the agent in a new thread. Initializes the agent's priority queue, condition, and agent thread. Logs a debug message with the agent's name and LLM.

        Args:
            mapper (Mapper): The mapper object used by the agent.

        Returns:
            None
        """
        # Start the agent in a new thread
        self.mapper = mapper  # python - call by value or reference????
        self.pq = mapper.get_PQ_by_name(self.agentName)
        self.condition = mapper.get_COND_by_name(self.agentName)
        self.agent_thread = threading.Thread(target=self.run)
        self.agent_thread.daemon = True
        self.agent_thread.start()
        logging.debug(f"Agent: {self.agentName} LLM: {self.llm}")


# Example Usage:

if __name__ == "__main__":
    # Example usage:
    # Start the agent in a new thread
    agent_list = ["RESEARCHER", "WRITER", "EMAILER"]
    mapper = NameIndexMapper()
    mapper.add_mapping(agent_list[0])
    mapper.add_mapping(agent_list[1])
    mapper.add_mapping(agent_list[2])
    agent1 = Agent(agentName=agent_list[0], consumerAgent=agent_list[1])
    agent2 = Agent(agentName=agent_list[1], consumerAgent=agent_list[2])
    agent3 = Agent(agentName=agent_list[2], consumerAgent=agent_list[0])
    # , role="RESEARCHER", goal="search for latest trends in Carona treatment", capability="INTERNET_SEARCHER", agent_type="STATIC",
    #                multiplicity=3, task="search internet for the goal", output_consumer_agent=agent_list[1],
    #               llm_api="gpt4", go_d_timer=10000, llm_resp_timer=20, tools_list=["Google", "Bing","DuckduckGo-search" ])
    agent1.start_agent(mapper)
    agent2.start_agent(mapper)
    agent3.start_agent(mapper)
    time.sleep(5)  # Simulating data availability after 5 seconds
    msg1 = "message for agent RESEARCHER from ENVIRONMENT "
    createAndsendMessage("ProfAgent", agent_list[0], msg1, mapper)
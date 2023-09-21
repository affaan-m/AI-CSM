from langchain.chat_models import ChatOpenAI
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.llms import OpenAI
from langchain.agents.tools import Tool
from langchain.chains import LLMMathChain
from langchain.agents.agent_toolkits import GmailToolkit
from langchain.agents.agent_toolkits import O365Toolkit
from langchain.tools.gmail.utils import build_resource_service, get_gmail_credentials

class EmailAgentHandler:

    global credential
    global agent
    global llm
    llm = OpenAI(temperature=0)

    def __init__(self): #initialize the email handler with the tools it needs

        # Need to do the oauth dance to get credentials.

        # Can review scopes here https://developers.google.com/gmail/api/auth/scopes
        # For instance, readonly scope is 'https://www.googleapis.com/auth/gmail.readonly'

        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        credentials = get_gmail_credentials(
            token_file="token.json",
            scopes=["https://www.googleapis.com/auth/gmail.modify", 
                    "https://www.googleapis.com/auth/gmail.readonly", 
                    "https://www.googleapis.com/auth/gmail.compose"],
            client_secrets_file="credentials.json",
        )

        api_resource = build_resource_service(credentials=credentials)
        toolkit = GmailToolkit(api_resource=api_resource)

        global llm
        global agent

        gToolKit = GmailToolkit()
        o365ToolKit = O365Toolkit()

        tools = gToolKit.get_tools() + o365ToolKit.get_tools()

        planner = load_chat_planner(llm)

        executor = load_agent_executor(llm, tools, verbose=True) # verbose is for testing purposes NOT PRODUCTION

        agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)


    def main(prompt):

        agent.run(prompt)
    
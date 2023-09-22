from langchain.chat_models import ChatOpenAI
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.llms import OpenAI
from langchain.agents.tools import Tool
from langchain.chains import LLMMathChain
from langchain.agents.agent_toolkits import GmailToolkit
from langchain.agents.agent_toolkits import O365Toolkit
from langchain.tools.gmail.utils import build_resource_service, get_gmail_credentials
from config import CONFIG  # Import the CONFIG dictionary
import requests
import os


class EmailAgentHandler:

    global credential
    global agent
    global llm
    llm = OpenAI(temperature=0)

    def __init__(self, emailclient): #initialize the email handler with the tools it needs

        # Now config is a Python dictionary containing your credentials


        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.

        toolkit # this is the toolkit the agent will use, we determine if we use the gmail toolkit or the o365 toolkit

        if (emailclient == "gmail"):

            # Extract Gmail API-related details from the CONFIG dictionary
            gmail_config = CONFIG["google_api"]
            scopes = gmail_config["scopes"]
            token_file = gmail_config["token_file"]
            client_secrets_file = gmail_config["client_secrets_file"]

            credentials = get_gmail_credentials(
                token_file = token_file,
                scopes = scopes,
                client_secrets_file = client_secrets_file
            )

            api_resource = build_resource_service(credentials=credentials)
            toolkit = GmailToolkit(api_resource=api_resource)

        elif (emailclient == "outlook"):

            # hardcode the environment variables of the machine running this code: probably not good practice and should be changed
            os.environ['CLIENT_ID'] = CONFIG["microsoft_api"]["client_id"]
            os.environ['CLIENT_SECRET'] = CONFIG["microsoft_api"]["client_secret"]

            def get_office365_token_with_config(config): # o365 auth is not as simple as gmail...
                tenant_id = "common"                     # Use "common" for multi-tenant applications
                token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
                
                token_data = {
                    'client_id': CONFIG["microsoft_api"]["client_id"],
                    'scope': CONFIG["microsoft_api"]["scopes"],
                    'client_secret': CONFIG["microsoft_api"]["client_secret"]
                }
                
                response = requests.post(token_url, data=token_data)
                token_json = response.json()
                
                if "error" in token_json:
                    print(f"Error in token acquisition: {token_json['error_description']}")
                    return None
                
                return token_json.get('access_token')

            # Get Access Token
            credentials = get_office365_token_with_config(CONFIG["microsoft_api"])

            toolkit = O365Toolkit()


        global llm
        global agent

        tools = toolkit.get_tools()

        planner = load_chat_planner(llm)

        executor = load_agent_executor(llm, tools, verbose=True) # verbose is for testing purposes !NOT! PRODUCTION

        agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)


    def main(prompt):

        agent.run(prompt)
    

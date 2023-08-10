#Setup
import os
import openai
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
import json

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
user_name = "Haley"

import requests
slack_api = "xoxb-5488807623827-5720729514176-armS1Ag33GrHucEeMw07YaIL"
slack_API_URL = "https://slack.com/api/chat.postMessage"
username = "Haley"

openai.api_key = "OPEN_AI_KEY"

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.8, 
    )
    return response.choices[0].message["content"]


#Console Prompt

#test input
text = "I want to schedule a meeting with Cindy and Peterson for 4:00PM to talk about contracting with Apple. I also need to meet with Lukas Medina at 2:00PM and send an email to him about Katherine's baby."

print("Generating schedule...")

#instead of using StructuredOutputParser and langchain prompts, I ended up doing it manually
meeting_template = """
        From the text, find out if the user wants to schedule a meeting.
        If true, extract the following information for each meeting, respectively:
        1) Date and/or time of the meeting. Do not leave blank.
        2) Give the meeting a concise 2-6 word title based on the users needs. Do not leave blank.
        3) List the individuals that would be involved.
        4) Give brief, 1-2 setence description. Leave blank if limited information is given.
        The keys should be only: Date, Title, Personnel, and Description. 
        The output should be in the form of a python dictionary. 
        If possible, analyze the date arrange in order from earliest to latest with the 
        latest meetings being at the bottom.
        If the time and date is unknown, place the meeting closer to the bottom.
        An example of the format is as follows:
        {{
        "Meeting 1": {{
                "Date": "4:00PM Today",
                "Title": "Phonecall meeting with Diana",
                "Personnel": ["Diana", "Mathew Briggs"],
                "Description": ""
        }},
        "Meeting 2": {{
                "Date": "2:00PM Tomorrow",
                "Title": "Contracting meeting with Jeff",
                "Personnel": ["Jeff"],
                "Description": ""
        }},
        "Meeting 3": {{
                "Date": "5:00PM Tomorrow",
                "Title": "Business dinner with Mclaren",
                "Personnel": ["Mclaren"],
                "Description": "Discuss concerns and brainstorm solutions."
        }}
        }}
        """
email_template = "The user's name is " + user_name + """. 
        From the text, find out if the user wants to send an email.
        If true, extract the following information for each meeting, respectively:
        1) Give the email a concise 2-6 subject of the email based on the user's needs. Do not leave blank.
        2) List the individuals that the email is intended for.
        3) Write a concise, friendly, yet professional email based on the user's needs expressed in the text.
        The keys should be only: Subject, Personnel, and Content.
        The output should be in the form of a python dictionary. 
        An example of the format is as follows:
        {{
            "Email 1": {{
                    "Subject": "Editorial Notes",
                    "Personnel": ["Bobbert"],
                    "Content": "Dear Mr. Bobbert,\n I thoroughly enjoyed your book. \n Sincerely, Ms. Johnson."
        }}""" 
priority_template = "leave blank for now"
template = f"""
    From the following text, extract the following information:

    Meetings: {meeting_template}

    Emails: {email_template}

    Priorities: {priority_template}

        
    Here's the given text from the user. {text}

    Format this like a JSON file. Here's an example:

    }}
        Meetings: {{
                "Meeting 1": {{
                        "Date": "4:00PM Today",
                        "Title": "Phonecall meeting with Diana",
                        "Personnel": ["Diana", "Mathew Briggs"],
                        "Description": ""
                }},
        Emails: {{
                "Email 1": {{
                        "Subject": "Editorial Notes",
                        "Personnel": ["Bobbert"],
                        "Content": "Dear Mr. Bobbert,\n I thoroughly enjoyed your book. \n Sincerely, Ms. Johnson."
                }},
        Priorities:
                {{
                "Task 1": {{
                        "Date": "Tomorrow after work",
                        "Title": "Plant tomatoes",
                        "Description": "Plant tomatoes next to the squash."
                }},
     }}

    """

output = get_completion(template)
schedule_json = json.loads(output)

#Meetings
def meeting_invites():
        meeting_editor = schedule_json["Meetings"]["Meeting " + input(" If you would like to send out a meeting invite,\n please enter the meeting number: ")]
        print(meeting_editor["Date"])
        print(meeting_editor["Title"])
        print(meeting_editor["Personnel"])
        print(meeting_editor["Description"])

        slack_message_template = f"""Write a concise, friendly, yet professional announcement of a meeting 
        given the following information.
        Meeting date: {meeting_editor["Date"]}
        Meeting title: {meeting_editor["Title"]}
        Description: {meeting_editor["Description"]}
        Sender name: {username}
        Only include given information. Do not repeat information.
        """
        slack_message_content = get_completion(slack_message_template)
        print("\nHere's a draft for an announcement on Slack of your meeting.")
        print(slack_message_content)
        send_YN = input("Send? (Y/N) ")

        haleys_slack_id = "U05JMMBAGVA"

        def send_message_slack(destination, content):
                auth = {
                        "Authorization": f"Bearer {slack_api}",
                        "Content-Type": "application/json"
                }
                message = {
                        "channel" : destination,
                        "text" : content
                }
                response = requests.post(slack_API_URL, headers = auth, json = message)
                if response.status_code == 200:
                        print("Message sent.")
                else:
                        print(response.status_code)
                        print(response.json)

        if send_YN =="Y" or send_YN =="y":
                send_message_slack(destination = haleys_slack_id, content = slack_message_content)
        else:
                print("Message canceled")

#Emails-- I should make this go through Google instead of SMTP
def send_email():
        email_editor = schedule_json["Emails"]["Email " + "1"]
        subject = email_editor["Subject"]
        message = email_editor["Content"]

        #test login info
        sender_email = 'haley@dcube.ai'
        receiver_email = 'haleyfchen@gmail.com'
        username = 'haley@dcube.ai'
        password = 'tkitrcdtwwujtomv' #had to enable duo for this
        
        #login
        smtp_server = 'smtp.gmail.com'
        smtp_port = 587 
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject
        msg.attach(MIMEText(message, 'plain'))
        context = smtplib.SMTP(smtp_server, smtp_port)
        context.starttls()
        context.login(username, password)

        #send
        context.sendmail(sender_email, receiver_email, msg.as_string())

        context.quit()
        print("Sent.")

#Priorities - Trello/Jira?

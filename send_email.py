import requests
import json
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
import getpass
from bs4 import BeautifulSoup

def send_email():
    your_email = "david89062388@gmail.com"
    your_password = "tehxfalyzkkdplcn"
    send_email_to = "bitcointest0206@gmail.com"
                   
    # create message object instance
    msg = MIMEMultipart()

    # the parameters of the message
    password = your_password
    msg['From'] = your_email
    msg['To'] = send_email_to
    msg['Subject'] = "ASAP! Someone is falling!!!"

    # your message
    message = "The system detected that someone has fallen, please check as soon as possible to assist rescue, Thank You!"

    # adds in the message from the above variable
    msg.attach(MIMEText( message, "plain" ) )

    # create the gmail server
    server = smtplib.SMTP( "smtp.gmail.com: 587" )

    server.starttls()

    # Login Creds for sending the email
    server.login( msg['From'], password )

    # sends the message
    server.sendmail( msg['From'], msg['To'], msg.as_string() )
    
    print("\nSend Email!!!\n")
    server.quit()



'''
Send an multipart email with HTML and plain text alternatives. The message
should be constructed as a plain-text file of the following format:

    From: Your Name <your@email.com>
    To: Recipient One <recipient@to.com>
    Subject: Your subject line
    ---
    Markdown content here
The script accepts content from stdin and, by default, prints the raw
generated email content to stdout.
Preview your message on OS X by invoking the script with `-p` or
`--preview`, and it will open in your default mail client.
To send the message, invoke with `-s` or `--send`. You must have a
JSON file in your home directory named `.markdown-to-email.json`
with the following keys:
    {
        "username": "smtp-username",
        "smtp": "smtp.gmail.com:587",
        "password": "your-password"
    }
Enjoy!
'''


import os
import re
import sys
import json
import argparse
import smtplib
import subprocess
import yaml
import logging
import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
try:
    import pygments
    import markdown
except ImportError:
    print('This script requires pygements and markdown to be installed.')
    print('Please:')
    print('   pip install pygments markdown')
    sys.exit(0)


def sort_papers(papers):
    output = dict()
    keys = list(papers.keys())
    keys.sort(reverse=True)
    for key in keys:
        output[key] = papers[key]
    return output

class SendEmail(object):
    def __init__(self, config_file="./config.yaml") -> None:
        self.config = self.readconfig(config_file)
        self.css = subprocess.check_output(
            ['pygmentize', '-S', 'default', '-f', 'html'])
        self.markdown_content = ''
        self.html_content = ''

    def readconfig(self, config_file) -> dict:
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        # TODO:
        config['email']['username'] = os.environ["USERNAME"]
        config['email']['smtp'] = os.environ["SMTP"]
        config['email']['password'] = os.environ["PASSWORD"]

        config['email']['headers']['To'] = os.environ["TO"]
        config['email']['headers']['From'] = os.environ["FROM"]
        return config['email']

    def transform_md(self, raw_md):
        """_summary_

        Parameters
        ----------
        raw_md : _type_ string(content or path) #TODO: path
            _description_ raw paper markdown content

        Returns
        -------
        markdown_content, html_content : _type_ string, string
            _description_ md string and html string
        """
        self.markdown_content = raw_md.strip()
        self.html_content = markdown.markdown(self.markdown_content)
        self.html_content = '<style type="text/css">' + self.css.decode('utf-8') +'</style>' + self.html_content

        return self.markdown_content, self.html_content

    def send(self):
        """_summary_

        Parameters
        ----------
        headers : _type_ config['email']['headers']
            _description_
        """

        headers = self.config['headers']

        message = MIMEMultipart('alternative')
        message['To'] = headers['To']
        message['From'] = headers['From']
        message['Subject'] = headers['Subject']

        self.json2md(self.config['today_json_path'], self.config['today_md_path'])
        with open(self.config['today_md_path'], 'r') as file:
            content = file.read()
        self.transform_md(content)

        # attach the message parts
        message.attach(MIMEText(self.markdown_content, 'plain'))
        message.attach(MIMEText(self.html_content, 'html'))

        if headers['send']:
            to = message['To'].split(',')

            try:
                server = smtplib.SMTP(self.config['smtp'], self.config['port'])
                # server.starttls()
                server.login(self.config['username'], self.config['password'])
                server.sendmail(message['From'], to, message.as_string())
            except smtplib.SMTPServerDisconnected as e:
                print("Failed to connect to the server. Incorrect SMTP server details or network issues may be the cause.")
                print(str(e))
            except smtplib.SMTPAuthenticationError:
                print("SMTP Authentication Error. The server didn't accept the username/password combination.")
            except smtplib.SMTPException as e:
                print("An error occurred while sending the email. Check your email settings and network connection.")
                print(str(e))

            server.quit()
        if headers['preview']:
            open('/tmp/preview.eml', 'w').write(message.as_string())
            os.system('thunderbird /tmp/preview.eml')
        else:
            print(message.as_string())

    def json2md(self, filename,
                md_filename,
                use_toc=True):

        def pretty_math(s:str) -> str:
            ret = ''
            match = re.search(r"\$.*\$", s)
            if match == None:
                return s
            math_start,math_end = match.span()
            space_trail = space_leading = ''
            if s[:math_start][-1] != ' ' and '*' != s[:math_start][-1]: space_trail = ' '
            if s[math_end:][0] != ' ' and '*' != s[math_end:][0]: space_leading = ' '
            ret += s[:math_start]
            ret += f'{space_trail}${match.group()[1:-1].strip()}${space_leading}'
            ret += s[math_end:]
            return ret

        DateNow = datetime.date.today()
        DateNow = str(DateNow)
        DateNow = DateNow.replace('-','.')

        with open(filename, "r") as f:
            content = f.read()
            if not content:
                data = {}
            else:
                data = json.loads(content)

        # clean README.md if daily already exist else create it
        with open(md_filename,"w+") as f:
            pass

        # write data into README.md
        def replace_spaces_with_hyphens(s):
            return s.replace(' ', '-')
        with open(md_filename,"a+") as f:
            if use_toc == True:
                f.write("**TOC**  \n")
                for keyword in data.keys():
                    keyword_with_hyphens = replace_spaces_with_hyphens(keyword)
                    f.write(f"  \n[**{keyword}**](#{keyword_with_hyphens})  \n")
                    for _,v in data[keyword].items():
                        title_with_hyphens = replace_spaces_with_hyphens(v['paper_title'])
                        f.write(f"  [{v['paper_title']}](#{title_with_hyphens})  \n")

            for keyword in data.keys():
                day_content = data[keyword]
                if not day_content:
                    continue
                f.write(f"## {keyword}  \n\n")
                day_content = sort_papers(day_content)
                for _,v in day_content.items():
                    if v is not None:
                        content = pretty_math(self.json2str(v))
                        f.write(content) # make latex pretty

                f.write(f"  \n\n\n\n")
        # logging.info(f"{task} finished")

    def json2str(self, data):
        tittle = data['paper_title']
        abstract = data['paper_abstract']
        authors = data['paper_authors']
        comments = data['comments']
        paper_url = data['paper_url']
        code_url = data['code_url']
        content = ''

        content += "### [" + tittle + "](" +paper_url + ")  \n"
        if code_url:
            content += "[[code](: " + code_url + ")]  \n"

        content += authors + "  \n"

        content += "<details>  \n"
        content += "  <summary>Abstract</summary>  \n"
        content += "  <ol>  \n"
        content += "    " + abstract + "  \n"
        content += "  </ol>  \n"
        content += "</details>  \n"

        if comments:
            content += "**comments**: " + comments + "  \n"
        content += "  \n"
        return content
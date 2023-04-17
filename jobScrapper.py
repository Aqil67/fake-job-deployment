import requests
from bs4 import BeautifulSoup


class JobScraper:
    def __init__(self, url):
        self.url = url
        self.res = requests.get(url)
        self.soup = BeautifulSoup(self.res.content, 'html.parser')
        self.data = {}

    def scrape(self):
        # Get the title
        title = self.soup.find('title')
        self.data['title'] = title.text

        # Get the job and company description
        job_and_company_description = self.soup.find('div', {'class': 'z1s6m00 _5135ge0 _5135ge7 _5135gei'})
        if job_and_company_description:
            self.data['job_and_company_description'] = ' '.join(job_and_company_description.stripped_strings)
            self.data['job_and_company_description'] = ' '.join(self.data['job_and_company_description'].split())

        return str(self) # call __str__ method to return string representation of self.data

    def __str__(self):
        # Join the dictionary values into a string
        result = '\n'.join([f'{key}: {value}' for key, value in self.data.items()])
        return result

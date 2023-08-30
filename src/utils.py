import os
import yaml
import time
import requests
from flickrapi import FlickrAPI
import azure.ai.vision as sdk

class AzureImageRetrieval():
    def __init__(self,
                 config_file: str = 'config.yml',) -> None:
        ## Prepare config file
        self.config_file = config_file
        self.load_config()
        ## Configuration in retrieving images in Flickr
        self.API_KEY = self.config['flickr']['API_KEY']
        self.API_SECRET = self.config['flickr']['API_SECRET']
        self.NUMBER_OF_IMAGES = self.config['flickr']['NUMBER_OF_IMAGES']
        ## Configuaration in Azure
        self.CV_ENDPOINT = self.config['azure']['ENDPOINT']
        self.CV_KEY = self.config['azure']['KEY']


    def load_config(self):
        '''
        Load and extract config yml file.
        '''
        try:
            with open(self.config_file) as file:
                self.config = yaml.safe_load(file)
        except Exception as e:
            print(e)
            raise

    def downloadImages(self):
        try:
            # Initialize FlickrAPI
            flickr = FlickrAPI(self.API_KEY, self.API_SECRET, format='parsed-json')

            # Search images under Creative Commons license
            photos = flickr.photos.search(license='1,2,3,4,5,6', per_page=self.NUMBER_OF_IMAGES)  # 1-6 shows creative commons in Flickr

            # Create directory for downloaded images
            if not os.path.exists('downloaded_images'):
                os.makedirs('downloaded_images')

            # Download image 1 by 1
            for i, photo in enumerate(photos['photos']['photo']):
                if i % 10 == 0:
                    print(f'{i} / {self.NUMBER_OF_IMAGES} images downloaded')
                time.sleep(3)
                photo_id = photo['id']
                farm_id = photo['farm']
                server_id = photo['server']
                secret = photo['secret']
                
                # Populate URL for downloading
                url = f'https://farm{farm_id}.staticflickr.com/{server_id}/{photo_id}_{secret}.jpg'
                
                # Download image
                response = requests.get(url, stream=True)
                with open(f'downloaded_images/{photo_id}.jpg', 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)

            print("Complete download!")
        except Exception as e:
            print(e)
            raise



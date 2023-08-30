import os, sys
import yaml
import time
import requests
from flickrapi import FlickrAPI
import azure.ai.vision as sdk
import numpy as np
import pandas as pd

class AzureImageRetrieval():
    def __init__(self,
                 config_file: str) -> None:
        ## Prepare config file
        self.config_file = config_file
        self.load_config()
        ## Configuration in retrieving images in Flickr
        self.API_KEY = self.config['flickr']['API_KEY']
        self.API_SECRET = self.config['flickr']['API_SECRET']
        self.NUMBER_OF_IMAGES = self.config['flickr']['NUMBER_OF_IMAGES']
        self.NUMBER_PROCESS_IMAGES = self.config['flickr']['NUMBER_PROCESS_IMAGES']
        ## Configuaration in Azure
        self.CV_ENDPOINT = self.config['Azure']['ENDPOINT']
        self.CV_KEY = self.config['Azure']['KEY']
        self.endpoint = self.CV_ENDPOINT + '/computervision/retrieval:vectorizeImage?api-version=2023-02-01-preview&modelVersion=latest'
        self.headers = {
            "Content-Type": "application/octet-stream",  # binary data in sending API
            "Ocp-Apim-Subscription-Key": self.CV_KEY
        }
        self.vectors = dict()
        ## initialization of image SDK
        self.analysis_options = sdk.ImageAnalysisOptions()
        self.service_options = sdk.VisionServiceOptions(self.CV_ENDPOINT,
                                                        self.CV_KEY)

        ## VectorDB
        self.df = None
        self.vectorDB = self.config['app']['vectorDB']

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

    def getVector(self, 
                  image: str) -> np.array:
        '''
        Get vector with API for one image
        '''
        try:
            with open(image, mode="rb") as f:
                image_bin = f.read()
            response = requests.post(self.endpoint, headers=self.headers, data=image_bin)
            return np.array(response.json()['vector'], dtype='float32')
        except Exception as e:
            print(e)
            raise

    def getImageProperties(self,
                   image:str) -> str:
        try:
            self.analysis_options.features = (
                sdk.ImageAnalysisFeature.CROP_SUGGESTIONS |
                sdk.ImageAnalysisFeature.CAPTION |
                sdk.ImageAnalysisFeature.DENSE_CAPTIONS |
                sdk.ImageAnalysisFeature.OBJECTS |
                sdk.ImageAnalysisFeature.PEOPLE |
                sdk.ImageAnalysisFeature.TEXT |
                sdk.ImageAnalysisFeature.TAGS
            )
            self.analysis_options.language = "en"
            self.analysis_options.gender_neutral_caption = True
            vision_source = sdk.VisionSource(filename=image)
            image_analyzer = sdk.ImageAnalyzer(self.service_options
                                               , vision_source
                                               , self.analysis_options)
            ## Analyze image
            return image_analyzer.analyze()
        except Exception as e:
            print(e)
            raise

    def convertDataFrame(self):
        try:
            self.df = pd.DataFrame(self.vectors).T
        except Exception as e:
            print(e)
            raise

    def getVectorFromImages(self,
                            folder: str):
        '''
        - input:
            folder: Local folder name including images
        '''
        try:
            ## Get image file name
            images = [image for image in os.listdir(folder) if image.endswith('.jpg')]
            print(images)
            ## Analyze each image with API
            for image in images[:self.NUMBER_PROCESS_IMAGES]:
                ## Set image path
                image_path = os.path.join(folder, image)
                ## Get vector for image
                vector = self.getVector(image=image_path)
                print(vector)
                ## Analyze with Azure Vision API
                analyzed_result = self.getImageProperties(image=image_path)
                try:
                    caption_content = analyzed_result.caption.content
                except:
                    caption_content = None

                ## Store the results
                self.vectors[image] = {}
                self.vectors[image]['vector'] = vector
#                self.vectors[image]['properties'] = analyzed_result
                self.vectors[image]['caption'] = caption_content
                time.sleep(1.5)
            ## Convert to DataFrame
            self.convertDataFrame()
        except Exception as e:
            print(e)
            raise

    def storeDataFrame(self):
        try:
            self.df.to_pickle(self.vectorDB)
        except Exception as e:
            print(e)
            raise

    def loadDataFrame(self):
        try:
            self.df = pd.read_pickle(self.vectorDB)
        except Exception as e:
            print(e)
            raise



from utils import AzureImageRetrieval
import streamlit as st

class App(AzureImageRetrieval):
    def __init__(self,
                 config_file: str) -> None:
        AzureImageRetrieval.__init__(self, config_file)
        ## Load config file
        self.config_file = config_file
        self.load_config()



def main():
    a = App(config_file='../config.yml')

    ## Load MetaIndex & Index
    a.loadMetaIndex()
    a.loadIndex()


if __name__ == '__main__':
    main()
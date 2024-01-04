from box import Box
import yaml

class Hyperparam():
    def __init__(self,path='./hypertuning/params.yaml'):
        self.path = path
        self.params = None
        #print(os.pwd)
    def read(self):
        with open(self.path) as f:
            self.params = yaml.safe_load(f)
            self.params = Box(self.params)
    
    def get_params(self):
        return self.params

# if __name__ == '__main__':
#     hyp = Hyperparam()
#     hyp.read()
#     params = hyp.get_params()['NUMEPOCHS'][0]
#     print(params)

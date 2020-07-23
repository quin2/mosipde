from multiprocessing.managers import BaseManager
from plots import ISOplot

class MyManager(BaseManager):
    pass

MyManager.register('ISO', ISOplot)

if __name__ == '__main__':
	path = "/Users/quinnvinlove/Documents/sugarsBio/excel/24Sept19.xls"
	manager = MyManager()
	manager.start()
	result = manager.ISO(path)
	print(result.std_out())

#import matplotlib.pyplot as plt
import matplotlib
import base64
from io import BytesIO

from plots import ISOplot

matplotlib.use("module://ipykernel.pylab.backend_inline")
print(matplotlib.get_backend())

fig = matplotlib.pyplot.figure(figsize=(20, 50))

ip = ISOplot("/Users/quinnvinlove/Documents/sugarsBio/excel/24Sept19.xls")
ip.overview(fig=fig, gW=4)

matplotlib.pyplot.savefig("test2.png")


"""
tmpfile = BytesIO()
fig.savefig(tmpfile, format='png')
encoded = base64.b64encode(tmpfile.getvalue()).decode('utf8')

html = '<html>' + '<img src=\'data:image/png;base64,{}\'>'.format(encoded) + '</html>'

with open('test.html','w') as f:
    f.write(html)
"""
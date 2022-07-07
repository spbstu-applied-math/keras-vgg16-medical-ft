
import matplotlib.pyplot as plt

import matplotlib.cbook as cbook

with cbook.get_sample_data('D:/TUPD/test/CNV/CNV-697731-43.jpeg') as image_file:
    image = plt.imread(image_file)
with cbook.get_sample_data('D:/TUPD/test/DME/DME-4030787-14.jpeg') as image_file:
    image1 = plt.imread(image_file)
with cbook.get_sample_data('D:/TUPD/test/DRUSEN/DRUSEN-5033521-1.jpeg') as image_file:
    image2 = plt.imread(image_file)
with cbook.get_sample_data('D:/TUPD/test/NORMAL/NORMAL-751527-3.jpeg') as image_file:
    image3 = plt.imread(image_file)
fig, ax = plt.subplots()
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3= plt.subplots()
ax.imshow(image)
ax1.imshow(image1)
ax2.imshow(image2)
ax3.imshow(image3)
ax.set_title('CNV')
ax1.set_title('DME')
ax2.set_title('DRUSEN')
ax3.set_title('NORMAL')

fig.set_figwidth(5)
fig.set_figheight(5)
fig1.set_figwidth(5)
fig1.set_figheight(5)
fig2.set_figwidth(5)
fig2.set_figheight(5)
fig3.set_figwidth(5)
fig3.set_figheight(5)
plt.show()
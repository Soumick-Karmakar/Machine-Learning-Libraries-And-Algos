import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,5,11)
y = x **2

#print(x)
#print(y)

#There are two ways of creating matplotlib graphs: Functinal Method and Object Oriented method

# Functional Method:

#plot:
#plt.plot(x,y,'y-')
#plt.xlabel('X label')
#plt.ylabel('Y label')   
#plt.show()

#subplot:
#plt.subplot(1,2,1)
#plt.plot(x,y,'y')
#plt.subplot(1,2,2)
#plt.plot(y,x,'b')
#plt.show()


# Object Oriented

#plot:
#fig = plt.figure()
#axes = fig.add_axes([0.1,0.1,0.8,0.8])
#axes.plot(x,y)
#axes.set_xlabel('X label')
#axes.set_ylabel('Y label')
#axes.set_title('Set Title')
#plt.show()

#fig = plt.figure()
#axes1 = fig.add_axes([0.1,0.1,0.8,0.8])
#axes2 = fig.add_axes([0.2,0.5,0.4,0.3])
#axes1.plot(x,y)
#axes2.plot(y,x)
#axes1.set_title('Larger')
#axes2.set_title('Smaller')
#plt.show()

#subplots:
#fig,axes = plt.subplots(nrows=1, ncols=2)
#plt.tight_layout()
#axes[0].plot(x,y,'y-')
#axes[0].set_title('First Plot')
#axes[1].plot(y,x,'b-')
#axes[1].set_title('Second Plot')
#plt.show()



# Figure Size, Aspect Ratio and DPI

#fig = plt.figure(figsize=(3,2))
#ax = fig.add_axes([0,0,1,1])
#ax.plot(x,y)
#plt.show()

#fig,axes = plt.subplots(nrows=2, ncols=1, figsize=(8,2))
#axes[0].plot(x,y)
#axes[1].plot(y,x)
#plt.show()
#fig.savefig('my_plots.jpeg',dpi=200)  #to save the plots as an image

# To superimpose two graphs
#fig =  plt.figure()
#ax = fig.add_axes([0,0,1,1])
#ax.plot(x, x**2, label='square')
#ax.plot(x, x**3, label='cube') 
#ax.legend(loc=4)
#plt.show()

# loc determines the legend position:
# 0 = best
# 1 = upper right
# 2 = upper left
# 3 = lower left
# 4 = lower right
# 5 = right
# 6 = center left
# 7 = center right
# 8 = lower center
# 9 = upper center
# 10 = center
# Also we can select a custom position by mentioning : loc = (0.1,0.1)



# Plot Appearence:

#line color:
#fig = plt.figure()
#ax = fig.add_axes([0,0,1,1])
#ax.plot(x,y,color='yellow')  # color parameter takes string as basic color names or color hash codes
#plt.show()

#line width or line style:
#fig = plt.figure()
#ax = fig.add_axes([0,0,1,1])
#ax.plot(x,y,color='purple', lw=2, alpha=0.5, ls='-', marker='o', markersize=10, markerfacecolor='yellow',markeredgewidth=3, markeredgecolor='purple')
##ax.set_xlim([0,1])
##ax.set_ylim([0,2])
#plt.show()

# linewidth determines the lie thickness | instead of writing 'linewidth' we can also write 'lw'
# alphs determines the opacity | it ranges between 0-1
# linestyle determines the line type, whether dotted/dashed or both | linestyles = '-' or '--' or '-.' or ':' or 'steps' | instead of writing 'linestyle' we can also write 'ls'
# marker actually marks the data points on the line| markers = 'o' or '+' or '*' or '1'
# markersize determines the size of the data points displayed
# markerfacecolor determines the color of the data points
# markeredgewidth and markeredgecolor determines the width and color of the boundary of the data points


# Special Plot Types:

#Scatter Plot
#plt.scatter(x,y)
#plt.show()

#Histogram
#from random import sample
#data = sample(range(1,1000),100)
#plt.hist(data)
#plt.show()

#Rectangular Box Plots
#data = [np.random.normal(0,std,100) for std in range(1,4)]
#plt.boxplot(data, vert=True, patch_artist=True);
#plt.show()


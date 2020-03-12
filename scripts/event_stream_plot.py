import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class EventStreamPlotter():

	def __init__(self, w,h,dt=1):
		"""

		"""
		self.w = w
		self.h = h
		self.dt = dt

		self.fig = plt.figure(figsize=(15,15))
		self.ax = self.fig.add_subplot(1,1,1, projection = '3d')
		self.ax.set_xlim3d([0,w])
		self.ax.set_ylim3d([0,dt*1.5])
		self.ax.set_zlim3d([-h,0])
		self.ax.set_xlabel("width")
		self.ax.set_ylabel("time")
		self.ax.set_zlabel("height")

		plt.ion()

	def update(self,events):
		"""
		plot events
		events should be an np array in order : x,y,time,polarity
		polarity should be +1 or -1
		"""
		for e in events:
			self.ax.scatter(int(e[0]), e[2],-int(e[1]), color = 'r' if e[3]>0 else 'b' , s=40)
			#plt.pause(0.0001)

	def stop(self):
		"""
		Call when no more events are to be received
		"""
		plt.show(block=True)




if __name__ == '__main__':

	# example 3d plot
	n = 200
	w = 25
	h = 15
	dt = 2.
	x = np.random.randint(0,w, size=n)
	y = np.random.randint(0,h, size=n)
	t = np.random.rand(n)*dt
	p = np.random.rand(n) - 0.5
	colors = np.array([1 if a > 0 else -1 for a in p])

	events = np.hstack((x.reshape(-1,1),y.reshape(-1,1),t.reshape(-1,1),colors.reshape(-1,1)))

	plot = EventStreamPlotter(w,h,dt)
	plot.update(events)
	plot.stop()

	'''
	plt.ion()

	fig = plt.figure(figsize=(15,15))
	ax = fig.add_subplot(1,1,1, projection = '3d')
	ax.set_xlim3d([0,w])
	ax.set_ylim3d([0,dt*1.5])
	ax.set_zlim3d([0,h])
	ax.set_xlabel("width")
	ax.set_ylabel("time")
	ax.set_zlabel("height")

	for e,c in zip(events,colors):
		ax.scatter(e[0], e[1], e[2], color = c, s=40)
		plt.pause(0.001)

	plt.show(block=True)
	'''
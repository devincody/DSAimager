
import numpy as np      
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
	pos = get_pos(False)

	fig = plt.figure(figsize = (12,6))
	plt.subplot(1,2,1)
	plt.scatter(pos[:,0], pos[:,1])
	plt.xlabel("East")
	plt.ylabel("North")
	plt.title("DSA Antenna Locations")


	ax = fig.add_subplot(122, projection='3d')
	ax.scatter(pos[:,0], pos[:,1], pos[:,2])
	ax.set_zlim3d(-7,7)
	ax.set_zlabel("UP")
	ax.view_init(elev = 13, azim =-89)
	plt.xlabel("East")
	plt.ylabel("North")
	plt.show()

def get_pos(drop_last = False):
	poss = np.array([[-2409464.509451, -4477971.269560, 3839125.030714],
					 [-2409466.444669, -4477974.866145, 3839119.656516],
					 [-2409493.509049, -4478025.165060, 3839044.497361],
					 [-2409547.551800, -4478125.603040, 3838894.417905],
					 [-2409424.013109, -4478297.667883, 3838772.061349],
					 [-2409429.957258, -4478294.469509, 3838772.061349],
					 [-2409682.473515, -4478158.597870, 3838772.061349],
					 [-2409746.758386, -4478124.008054, 3838772.061349],
					 [-2409770.667075, -4478111.143485, 3838772.061349],
					 [-2410525.007124, -4477850.572649, 3838597.061665]])
	#poss[:,0], poss[:,2] = np.copy(poss[:,2]), np.copy(poss[:,0])
	if drop_last:
		poss = poss[:-1,:]
	poss -= poss.mean(axis = 0)

	# 37deg 14'00.2"N 118deg 17'00.2"W
	lat = 37.233386 #(37.0 + 14.0/60 + 00.2/3600) #
	lon = -118.2834 #-(118.0 + 17.0/60 + 00.2/3600) #
	R = Rot(lat, lon)
	pos = np.dot(R,poss.T).T
	pos -= np.median(pos, axis = 0)
	return pos




def Rot(lat, lon):
	lat *= np.pi/180.0
	lon *= np.pi/180.0
	R = np.zeros((3,3))
	R[0,0] = -np.sin(lon)
	R[0,1] = np.cos(lon)
	R[1,0] = -np.sin(lat)*np.cos(lon)
	R[1,1] = -np.sin(lat)*np.sin(lon)
	R[1,2] = np.cos(lat)
	R[2,0] = np.cos(lat)*np.cos(lon)
	R[2,1] = np.cos(lat)*np.sin(lon)
	R[2,2] = np.sin(lat)
	return R


if __name__ == '__main__':
	main()
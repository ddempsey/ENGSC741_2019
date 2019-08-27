
# qk2.py
# 1D earthquake simulation with slip-weakening friction
# author: David Dempsey

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

class Fault(object):
	''' Class and methods that implement 1D crack propoagation with linear slip-weakening friction.
	'''
	# OBJECT METHODS
		# construct the EQ simulation object
	def __init__(self, x, shear, fs=0.8, fd=0.6, normal=30.e6, dc=1.e-2, mu=30.e9, cs=3.e3, dtn=1000, tmax=10.):
		''' Constructor for Fault object
		
			params:
				x - vector of x locations (powers of 2 please)
				shear - shear stress at x locations
				fs - static friction coefficient (default 0.8)
				fd - dynamic friction coefficient (default 0.6)
				normal - normal stress (default 30 MPa)
				dc - critical slip distance (default 1 cm)
				mu - shear modulus (default 30 GPa)
				cs - shear wave speed (default 3 km/s)
				dtn - maximum number of timesteps (default 1000)
				
			notes:
				stresses are in MPa
		'''
		
		# quality control
		if not len(x) == 2**int(round(np.log2(len(x)))):
			raise ValueError('x vector length in powers of 2 only please')			
		if len(x) != len(shear):
			raise ValueError('shear stress vector must be same length as location vector, x')
		if fd >= fs:
			raise ValueError('dynamic friction should be less than static friction')
		
		# check sufficient discretisation to resolve critical crack length
		self.ac = 2*mu*dc/(np.pi*(fs-fd)*normal)
		if (x[1]-x[0]) > self.ac/5.:
			raise ValueError('critical crack length ({:3.2e}) not resolved by at least five elements'.format(self.ac))
			
		# assign values
		self.x = x
		self.nx = len(x)
		self.s = shear
		self.fs = fs
		self.fd = fd
		self.sn = normal
		self.dc = dc
		self.mu = mu
		self.cs = cs
		
		# create solution vectors
		self.f = 0.*self.x+self.fs			# friction coefficient, initially static
		self.u = 0.*self.x                  # slip vector, initially zero
		self.s0 = 1.*self.s					# initial shear stress
		self.L = self.x[-1] - self.x[0]
		self.k = np.fft.fftfreq(len(self.x), self.x[1]-self.x[0]) # wavenumbers of grid
		self.impedance = self.mu/2./self.cs
		self.stiffness = self.mu/2./np.pi
		self.rtol = 1.e-3
		self.dtn = dtn
		self.tmax = tmax
		
		# output vectors
		self.tv = np.zeros((1,self.dtn+1))[0]         # output time
		self.vmax = np.zeros((1,self.dtn+1))[0]       # max slip velocity
		self.ivmax = np.zeros((1,self.dtn+1))[0]      # location of max slip velocity
		
		self.nout = 10 								# number of snapshots
		self.vv = np.zeros((self.nx,self.nout))         	# slip velocity
		self.uv = np.zeros((self.nx,self.nout))         	# slip output
		self.fv = np.zeros((self.nx,self.nout))         	# friction output
		self.tvv = 0.*self.vv[0,:]				         	# time output
		# screen output
	def __repr__(self):
		return 'qk2Fault'
	
	# LIFECYCLE METHODS
		# impose stress loading to trigger an earthquake, 
	def trigger(self,x):
		''' trigger an earthquake by imposing a gaussian loading function
			
			params:
				x = centre of loading function
				
			notes:
				loading function width is set to the nucleation length
		
		'''
		# loading condition to promote failure - Gaussian at trigger location
		self.dsdt = np.exp(-(self.x-x)**2/self.ac**2)*1.e-3
		
		# start time when Mohr-Coulomb failure criterion met somewhere on the fault.
		i = np.argmin(abs(self.x-x))
		dsdti = self.dsdt[i]
		dsi =  self.f[i]*self.sn - self.s[i]
		
		# advance time to approximate MC failure
		self.t = dsi/dsdti
		
		# nucleate the earthquake
		self.nucleate()
		
		# compute the rupture
		self.rupture()
		
		# print simulation summary to screen
		self.summarise()
		# run nucleation calculations
	def nucleate(self):
		''' nucleate an earthquake
		'''		
		# parameters
		self.vmin = 1.e-1			# slip velocity denoting onset of seismic slip
		self.dt = 1.e-2					# initial time step
		
		# compute initial velocity
		self.v = self.compute_velocity(t=self.t, u=self.u, f = self.f)	
		
		# improved Euler iterations 
		err2 = None
		err1 = None
		self.nit = 0
		while np.max(self.v) < self.vmin:
			# predictor step
			f1 = self.compute_friction(f = self.f, du = self.v*self.dt)			
			u1 = self.u + self.v*self.dt
			
			# corrector step
			v1 = self.compute_velocity(t=self.t+self.dt, u=u1, f = f1)	
			self.v = 0.5*(self.v+v1)
			self.f = self.compute_friction(f = self.f, du = self.v*self.dt)	
			self.u = self.u + self.v*self.dt		
						
			# advance time step
			self.t = self.t + self.dt
			
			# compute error terms
			err0 = np.max(abs(self.u - u1)/(self.rtol*self.u))
			if err0 < 1.e-32:
				err1, err2 = None, None
			
			# compute timestep multiplier using PID controller
				# PID = Proportional-Integral-Derivative
			dt_ratio = (1./err0)**0.17
			if err1 is not None:
				dt_ratio *= (err1/err0)**0.245
			if err2 is not None:
				dt_ratio *= (err2**2/(err1*err0))**0.05
				
			# cap the maximum timestep multiplier at 2
			dt_ratio = np.min([dt_ratio, 2])
			self.dt = self.dt*dt_ratio
			
			# cycle error terms
			err2,err1,err0 = err1,err0,None
			
			# increment iteration counter and test exit condition
			self.nit +=1
			if self.nit >= self.dtn:
				raise ValueError('nucleation did not occur in {:d} iterations'.format(self.nit))
		# run rupture calculations
	def rupture(self):
		''' run the rupture
		'''
		
		# improved Euler iterations 
		err2 = None
		err1 = None
		self.nit = 0
		
		# output indices
		tout_inds = np.logspace(np.log10(self.dt), np.log10(self.tmax),self.nout)
		tout_inds = list(tout_inds)
		tout_inds.reverse()
		tout_next = tout_inds.pop()
		io = 0
		
		while np.max(self.v) > self.vmin:
			# predictor step
			f1 = self.compute_friction(f = self.f, du = self.v*self.dt)			
			u1 = self.u + self.v*self.dt
			
			# corrector step
			v1 = self.compute_velocity(t=self.t+self.dt, u=u1, f = f1)	
			self.v = 0.5*(self.v+v1)
			self.f = self.compute_friction(f = self.f, du = self.v*self.dt)	
			self.u = self.u + self.v*self.dt		
			
			# advance time step
			self.t = self.t + self.dt
					
			# compute error terms
			err0 = np.max(abs(self.u - u1)/(self.rtol*self.u))
			if err0 < 1.e-32:
				err1, err2 = None, None
				
			# update time step
			dt_ratio = (1./err0)**0.17
			if err1 is not None:
				dt_ratio *= (err1/err0)**0.245
			if err2 is not None:
				dt_ratio *= (err1**2/(err2*err0))**0.05
				
			dt_ratio = np.min([dt_ratio, 2])
			self.dt = self.dt*dt_ratio
						
			# cycle error terms
			err2,err1,err0 = err1,err0,None
			
			# increment iteration counter
			self.nit +=1
			
			# save output
				# all stats output
			self.tv[self.nit-1] = 1.*self.t
			self.vmax[self.nit-1] = np.max(self.v)
			self.ivmax[self.nit-1] = np.argmax(self.v)
				# selected vector output
			#if self.nit%(self.dtn/self.nout) == 0:
			if (self.t-self.tv[0])>tout_next:
				self.uv[:,io] = 1.*self.u
				self.vv[:,io] = 1.*self.v
				self.fv[:,io] = 1.*self.f
				self.tvv[io] = 1.*self.t
				io += 1
				if len(tout_inds) == 0:
					tout_next = 1.e32
				else:
					tout_next = tout_inds.pop()
			
			# test exit conditions
				# number iterations
			if self.nit >= self.dtn:
				break
				# simulated length
			if (self.t-self.tv[0]) >= self.tmax:
				break
		# summarise simulation
	def summarise(self):
		''' prints output statisictics about simulation
		'''
		# exit condition
		if self.nit == self.dtn:
			print('simulation exited prematurely after exceeding iteration count')
		else:
			print('simulation exited after {:d} iterations'.format(self.nit))
			
	# ODE METHODS	
		# compute sliding velocity on crack
	def compute_velocity(self, t, u, f):
		''' compute velocity as a function of time, slip and friction
		'''		
		# compute velocity by rearranging Eq (2.18)
		v = (self.s0+self.dsdt*t - f*self.sn - self.stiffness*self.stress_drop(u))/self.impedance
		# set minimum velocity
		v[np.where(v<1.e-18)] = 1.e-18
		return v
		# compute friction on crack
	def compute_friction(self, f, du):
		''' compute change in friction
		'''
		# compute slip weakening
		f1 = f - (self.fs-self.fd)/self.dc*du
		# set minimum friction
		f1[np.where(f1<self.fd)]=self.fd
		return f1
		# compute stress changes on crack
	def stress_drop(self, u):
		''' use Fourier transform to compute stress drop
		'''
		# computed via EQs (2.17) and (2.18)
		return np.real(np.fft.ifft(abs(self.k)*np.fft.fft(u)))
		
	# PLOTTING METHODS
		# initial stress
	def show_stress(self, xlim = None, save = None):
		''' Display the stress state along the fault, relative to the various fault strengths.
		
			params:
				xlim - truncate the plot limits (mainly to exclude the zero pad)
				save - instead of displaying the plot, save to filename
		'''
		# make axes
		f,ax = plt.subplots(1,1)
		
		# plot stress
		ax.plot(self.x/1.e3,self.s/1.e6,'b-',label='shear stress')
		
		xl = [self.x[0]/1.e3, self.x[-1]/1.e3]
		ax.set_xlim(xl)
		
		# plot strengths
		ax.plot(xl,self.fs*self.sn*np.array([1,1])/1.e6,'k-',label='static strength')
		ax.plot(xl,self.fd*self.sn*np.array([1,1])/1.e6,'k--',label='dynamic strength')
		
		# plot upkeep
		ax.legend()
		ax.set_xlabel('along fault distance [km]')
		ax.set_ylabel('stress [MPa]')
		if xlim is not None:
			xlim = np.array(xlim)/1.e3		
			ax.set_xlim(xlim)
		
		# save or show figure
		if save is None:
			plt.show()
		else:
			plt.savefig(save, dpi=300)
		# summary of earthquake
	def show_eq(self, xlim = None, save = None):
		''' Display the slip along the fault.
		
			params:
				xlim - truncate the plot limits (mainly to exclude the zero pad)
				save - instead of displaying the plot, save to filename
		'''
		# make axes
		f,((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2)
		f.set_size_inches(16,12)
		#cmw = cm.get_cmap('winter_r')  
		cmw = cm.get_cmap('brg')     	
		
		# AX 1 - initial stress
		ax1.plot(self.x/1.e3, self.s/1.e6, '-', label='initial stress', color = cmw(0))
		
		xl = [self.x[0]/1.e3, self.x[-1]/1.e3]
		ax1.plot(xl,self.fs*self.sn*np.array([1,1])/1.e6,'k-',label='static strength')
		ax1.plot(xl,self.fd*self.sn*np.array([1,1])/1.e6,'k--',label='dynamic strength')
		
		# shading to show areas of positive and negative stress drop
		sd = self.fd*self.sn/1.e6
		for xi,si in zip(self.x/1.e3,self.s/1.e6):
			if si > sd:
				ax1.plot([xi,xi], [sd, si], zorder = -1, color = [0.8,0.8,1], lw=2)
			else:
				ax1.plot([xi,xi], [si, sd], zorder = -1, color = [1,0.8,0.8], lw=2)
		
		ax1.set_ylabel('stress [MPa]')
		
		# AX 2 - stress evolution
			# final stress
		s = self.s+self.dsdt*self.t - self.stiffness*self.stress_drop(self.u) - self.impedance*self.v
		ax1.plot(self.x/1.e3,s/1.e6,'r-',lw=2,label='final stress', color = cmw(0.5))
		ax2.plot(self.x/1.e3,s/1.e6,'r-',lw=2,label='final stress', color = cmw(0.5))
			# stress snapshots
		imax = int(self.nit/self.dtn*self.nout)
		for i in range(imax):
			s = self.s+self.dsdt*self.tv[int(i*self.dtn/self.nout)] - self.stiffness*self.stress_drop(self.uv[:,i]) - self.impedance*self.vv[:,i]
			
			ax2.plot(self.x/1.e3,s/1.e6,'-',color = cmw(i/imax/2.))
		
		ax2.set_ylabel('stress [MPa]')
		
		# AX 3 - slip evolution
			# final slip
		ax3.plot(self.x/1.e3,self.u,'-',lw=2, color = cmw(0.5))
			# slip snapshots
		for i in range(imax):
			u = self.uv[:,i]			
			ax3.plot(self.x/1.e3, u, '-', color = cmw(i/imax/2.))
			
		ax3.set_ylabel('slip [m]')
		
		# AX4 - velocity evolution		
		ax4.plot(self.tv[1:]-self.tv[0],self.vmax[1:],'k-')
		for i,tvi in enumerate(self.tvv):
			ind = np.argmin(abs(self.tv[1:] - tvi))
			ax4.plot(self.tv[ind]-self.tv[0], self.vmax[ind], 'o', color = cmw(i/imax/2.))
		ax4.set_xscale('log')
		ax4.set_yscale('log')
		ax4.set_xlabel('time')
		ax4.set_ylabel('max slipping velocity')		
				
		# AX 5 - friction evolution
			# final friction
		ax5.plot(self.x/1.e3,self.f,'-',lw=2,label='final friction', color = cmw(0.5))
			# friction snapshots
		for i in range(imax):
			f = self.fv[:,i]			
			ax5.plot(self.x/1.e3,f,'-',color = cmw(i/imax/2.))
		
		ax5.plot(self.x/1.e3,self.fs + 0.*self.x,'-',lw=2,label='initial friction', color = cmw(0.0))
		ax5.set_ylabel('friction')
		
		# AX 6 - velocity evolution
			# velocity snapshots
		for i in range(imax):
			v = self.vv[:,i]			
			ax6.plot(self.x/1.e3,v,'-',color = cmw(i/imax/2.))
		
		ax6.set_ylabel('slip velocity [m/s]')
		
		# axis upkeep
		if xlim is not None:
			xlim = np.array(xlim)/1.e3		
		for ax in [ax1,ax2,ax3,ax5,ax6]:
			if xlim is not None:
				ax.set_xlim(xlim)
			ax.legend()
			ax.set_xlabel('distance along fault [km]')
		
		# save or show figure
		if save is None:
			plt.show()
		else:
			plt.savefig(save, dpi=300)
			
	
	
	
		
		
		
		
		
		
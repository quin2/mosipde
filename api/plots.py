"""
Wrapper class for isodat plotting functions
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
from matplotlib import gridspec

import math
import glob
import os

from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

class ISOplot:

	def __init__(self, filePath):
		df = pd.read_excel(filePath)

		#load _all chart
		self._all = pd.concat([self.__generateChem(df2) for _, df2 in df.groupby(['Identifier 1', 'Identifier 2'])])
		self._all = self._all.sort_index()

		#create aa chart
		temp_aa = self._all[self._all['Identifier 1'] == 'AA std']
		temp_aa = temp_aa.drop(['Identifier 1', 'Identifier 2'], axis=1)
		aa_std = temp_aa.apply(np.std, axis=0)
		aa_avg = temp_aa.apply(np.mean, axis=0)
		self.aa = pd.concat([aa_std, aa_avg], axis=1, keys=['SD', 'mean'])
		self.aa = self.aa.dropna()

		#create IS chart
		temp_is = self._all[['nLeu', 'Nonadecane', 'Caffeine']]
		is_std = temp_is.apply(np.std, axis=0)
		is_avg = temp_is.apply(np.mean, axis=0)
		self._is = pd.concat([is_std, is_avg], axis=1, keys=['SD', 'mean'])

	def overview(self, fig, gW=4):

		compounds = self._all.drop(['Identifier 1', 'Identifier 2'], axis=1).columns

		gH = math.ceil(len(compounds) / gW)

		gs = gridspec.GridSpec(gH, gW) #32 samples, will have to fix later.
		#gs.update(wspace=0., hspace=0.2)

		for idx, compound in enumerate(compounds):
			i = idx % gW
			j = math.floor(idx / gW)

			gsx = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[j, i])
			ax1 = fig.add_subplot(gsx[0, 0])
			ax2 = fig.add_subplot(gsx[1, 0])

			self.__plotCompoundTimeSeries(self._all[self._all['Identifier 1'] != 'AA std'], compound, 'Green', 'Yellow', ax1)
			self.__plotCompoundTimeSeries(self._all[self._all['Identifier 1'] == 'AA std'], compound, 'Blue', 'Yellow', ax2)

			self.__displayTrendLine(self._all[self._all['Identifier 1'] == 'AA std'], compound, 'Blue', ax2)

			bbox = gs[j,i].get_position(figure=fig).get_points()
			xpos = (bbox[0][0] + bbox[1][0]) / 2

			t = fig.text(x=xpos, y=bbox[1][1] + 0.0025, s=str(compound), fontweight='bold', fontsize=18)

			r = fig.canvas.get_renderer()
			bb = t.get_window_extent(renderer=r)
			tt = fig.transFigure.inverted().transform((bb.width, bb.height))

			t.set_x(t.get_position()[0] - (tt[0] / 2))

			fig.add_subplot(gsx[1, 0])

		#fig.tight_layout()
		#gs.tight_layout(fig)
		

		return


	def __generateChem(self, df):
	    aa = df[['Row', 'Component', 'd 13C/12C']]
	    aa = aa.drop(aa[aa['Component'] == 'Blank'].index)
	    
	    aa = aa.drop(aa[pd.isna(aa['Component'])].index)

	    out = aa.pivot(index='Row', columns='Component', values='d 13C/12C').bfill().iloc[[0],:]
	    out.insert(0, 'Identifier 1', df['Identifier 1'].unique())
	    out.insert(1, 'Identifier 2', df['Identifier 2'].unique())

	    out.columns.name = None

	    return out

	def __outlier(self, data):
		osd = np.std(data)
		om = np.mean(data)
		upper = om + (2*osd)
		lower = om - (2*osd)

		return upper, lower

	def __plotCompoundTimeSeries(self, data, compound, primaryColor, secondaryColor, ax):
		work = data[compound]
		 
		upper, lower = self.__outlier(work)
		color = [secondaryColor if x > upper or x < lower else primaryColor for x in work]

		#draw scatter
		ax.scatter(data.index, work, c=color)

		#draw mean
		ax.axhline(np.mean(work), color='black')

		#set scatter ticks
		ax.set_xticks(self._all.index)

		#set tick spacing
		tick_spacing = 5
		ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

		return

	def __displayTrendLine(self,data, compound, color, ax3):
		work = data[compound]
		upper, lower = self.__outlier(work)

		work = work[(work < upper) & (work > lower)]

		if len(work) < 2:
		    return

		X = np.array(work.index).reshape(-1,1)

		y = np.array(work.values).reshape(-1, 1)

		pf = PolynomialFeatures(degree=2)
		X_poly = pf.fit_transform(X)

		clf = LinearRegression().fit(X_poly, y)


		model_range = self._all.index
		model_range_transform = pf.transform(np.array(model_range).reshape(-1, 1))

		r2 = r2_score(y, clf.predict(X_poly))


		ax3.plot(model_range, clf.predict(model_range_transform), color=color)

		ax3.text(x=0.7, y=0.05, transform=ax3.transAxes, s="r^2=%0.3f" % r2)

		return

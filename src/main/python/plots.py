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

		self.done = True

	def done():
		return self.done

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

			t = fig.text(x=xpos, y=bbox[1][1] + 0.0025, s=str(compound), fontweight='bold', fontsize=15)

			r = fig.canvas.get_renderer()
			bb = t.get_window_extent(renderer=r)
			tt = fig.transFigure.inverted().transform((bb.width, bb.height))

			t.set_x(t.get_position()[0] - (tt[0] / 2))

			fig.add_subplot(gsx[1, 0])

		return

	def aa_hist(self, gW=4, figsize=(20,50), dpi=20):
		gH = math.ceil(len(self.aa.index) / gW)

		fig, ax = plt.subplots(gH, gW, figsize=figsize, dpi=dpi)
		plt.subplots_adjust(hspace=0.5)

		for idx, compound in enumerate(self.aa.index):
		    i = idx % gW
		    j = math.floor(idx / gW)
		    
		    std_raw = self._all[self._all['Identifier 1'] == 'AA std']
		    std = std_raw[compound]
		    std = std.dropna()
		    
		    self.__generate_hist(ax[j, i], std, toff=-0.25)
		    
		    ax[j, i].title.set_text("AA Standard " + compound)
		    
		return fig

	def aa_out(self):
		std_raw = self._all[self._all['Identifier 1'] == 'AA std']
		return self.__find_outl_array(std_raw, self.aa.index)

	def std_hist(self, gW=4, figsize=(20,50), dpi=20):
		compounds = self._all.drop(['Identifier 1', 'Identifier 2'], axis=1).columns

		samples = self._all['Identifier 1'].unique()
		samples = np.delete(samples, np.where(samples == 'AA std'))
		    
		all_sd = [self.__my_std(self._all[self._all['Identifier 1'] == sample]) for sample in samples]
		all_sd = pd.concat(all_sd, keys=samples, axis=1).T
		all_sd = all_sd.drop('Identifier 2', axis=1)

		gH = math.ceil(len(all_sd.columns) / gW)

		
		fig, ax = plt.subplots(gH, gW, figsize=figsize, dpi=dpi)
		plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.5)

		for idx, col in enumerate(all_sd.columns):
		    i = idx % gW
		    j = math.floor(idx / gW)
		    
		    self.__generate_hist(ax[j, i], all_sd[col].dropna(), toff=-0.16)
		    
		    ax[j, i].set_title(col, fontdict={'fontsize': 16, 'fontweight': 'bold'})
		    
		return fig

	def std_out(self):
		compounds = self._all.drop(['Identifier 1', 'Identifier 2'], axis=1).columns

		samples = self._all['Identifier 1'].unique()
		samples = np.delete(samples, np.where(samples == 'AA std'))
		    
		all_sd = [self.__my_std(self._all[self._all['Identifier 1'] == sample]) for sample in samples]
		all_sd = pd.concat(all_sd, keys=samples, axis=1).T
		all_sd = all_sd.drop('Identifier 2', axis=1)

		all_sd = all_sd.round(3)

		final = self.__find_outl_array(all_sd, all_sd.columns)

		return final

	def is_hist(self, figsize=(20,30), dpi=20):
		#calculate summery for all compounds in internal standard
		std_raw = self._all[self._all['Identifier 1'] == 'AA std']
		test_raw = self._all[self._all['Identifier 1'] != 'AA std']

		fig, ax = plt.subplots(len(self._is.index), 2, figsize=figsize, dpi=dpi)

		    
		for idx, compound in enumerate(self._is.index):    
		    std = std_raw[compound]
		    test = test_raw[compound]
		    
		    self.__generate_hist(ax[idx, 0], std, toff=-0.12)
		    self.__generate_hist(ax[idx, 1], test, toff=-0.12)
		    
		    ax[idx, 0].title.set_text("AA Standard " + compound)
		    ax[idx, 1].title.set_text("Test " + compound)
		    
		return fig

	def is_out(self):
		def list_add(x, text):
			x.insert(0, text)
			return x


		std_raw = self._all[self._all['Identifier 1'] == 'AA std']
		test_raw = self._all[self._all['Identifier 1'] != 'AA std']

		standard = self.__find_outl_array(std_raw, self._is.index)
		test = self.__find_outl_array(test_raw, self._is.index)

		standard = [list_add(x, "IS") for x in standard]
		test = [list_add(x, "Test") for x in test]

		return standard + test

	def __find_outl_array(self, data, compounds):
		final = []

		for compound in compounds:
			find = data[compound]
			outl = find[self.__find_outliers_boolean(find)].dropna()
			for idx, value in outl.items():
				final.append([compound, idx, value])

		return final

	def __find_outliers_boolean(self, data):
	    upper, lower = self.__outlier(data)
	    outl = ~data.between(lower, upper)
	        
	    return outl

	def __my_std(self, data):
	    if len(data) == 3:
	        return np.std(data)
	    else:
	        return data.iloc[0] - data.iloc[1]

	def __iqr(self, x):
	    q75, q25 = np.percentile(x, [75 ,25])
	    iqr = q75 - q25
	    return iqr
    
	#use Freedman-Diaconis rule to decide bin size
	def __binSize(self, data):
	    bw = 2 * self.__iqr(data) / (len(data) ** 1/3)
	    if bw == 0:
	        return 1
	    bins = math.ceil((data.max() - data.min()) / bw)
	    return bins

	#generate histogram for axis
	def __generate_hist(self, ax, data, toff=-0.15):
		binsize = self.__binSize(data)
		ax.hist(data, bins=binsize, color='lightblue')
		ax.vlines(np.mean(data), *ax.get_ylim())

		ht = ax.transLimits.inverted().transform((1,1))[1]
		boxWidth = self.__outlier(data)[0] - self.__outlier(data)[1]
		aa_patch = Rectangle((self.__outlier(data)[1],0), width=boxWidth, height=ht, color='lightgreen', alpha=0.5)
		ax.add_patch(aa_patch)    

		ax.vlines(self.__outlier(data), *ax.get_ylim(), linestyle="dotted")

		infoSummery = "mean: %f, sd: %f" % (np.mean(data), np.std(data))
		ax.text(0.0, toff, s=infoSummery, transform=ax.transAxes, fontsize=16)

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

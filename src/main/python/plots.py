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
import gc
import copy

from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

import xlsxwriter

class ISOplot:

	def __init__(self, filePath):
		self.filePath = filePath
		
	def load(self):
		self.df = pd.read_excel(self.filePath)
		self.original_df = copy.deepcopy(self.df)
		self.chart_all()

	def chart_all(self):
		#load _all chart
		self._all = pd.concat([self.__generateChem(df2) for _, df2 in self.df.groupby(['Identifier 1', 'Identifier 2'])])
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

		#create NACME
		nacme = []
		out = self._all['Identifier 1'].unique()
		out = np.delete(out, np.argwhere(out=='AA std'))

		for sample in out:
			data = self._all[self._all['Identifier 1'] == sample]
			for compound in data.columns[2:]:
				alj = data[compound].values
				if not np.isnan(np.sum(alj)):
					nacme.append([data.index.values[0], sample, compound, alj[0], alj[1], alj[2], 
								  np.mean(alj), np.std(alj), (np.std(alj)/np.sqrt(len(alj)))])

		cNames = ['Row', 'Sample', 'AA', 'Inj_1', 'Inj_2', 'Inj_3', 'Mean', 'SD_inj', 'SE']
		self.nacme = pd.DataFrame(data=nacme, columns=cNames)

		#create check
		self.check = self._all[self._all['Identifier 1'] == 'Check S']


	def generate_all(self, tw, dpi):
		self.chart_all()

		std_width = 230
		aa_width = 200
		is_width = 250

		w = (tw-100)/dpi
		#self.OVER = plt.figure(figsize=(w, w*2), dpi=dpi)
		self.OVER = self.overview(figsize=(w, w*2), dpi=dpi)

		w = (tw-aa_width)/dpi
		self.AA_H = self.aa_hist(gW=3, figsize=(w,w*3), dpi=dpi)
		self.AA_T = self.aa_out()

		w = (tw-std_width)/dpi
		self.STD_H = self.std_hist(gW=3, figsize=(w, w * 4), dpi=dpi)
		self.STD_T = self.std_out()

		w = (tw-is_width)/dpi
		self.IS_H = self.is_hist(figsize=(w, w*2), dpi=dpi)
		self.IS_T = self.is_out()

	def overview(self, gW=4, figsize=(20,50), dpi=20):
		fig = plt.figure(figsize=figsize, dpi=dpi)
		r = fig.canvas.get_renderer()

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

			bb = t.get_window_extent(renderer=r)
			tt = fig.transFigure.inverted().transform((bb.width, bb.height))

			t.set_x(t.get_position()[0] - (tt[0] / 2))

			fig.add_subplot(gsx[1, 0])

		return fig

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

	def changeCell(self, row, component, new):
		selector = (self.df['Row'] == row) & (self.df.Component == component)
		old = self.original_df[selector]['d 13C/12C'].values[0]
		comment = "Changed from %.3f to %.3f" % (old, new)
		#write comment
		self.df.loc[selector, 'Notes'] = comment
		#write value
		self.df.loc[selector, 'd 13C/12C'] = new

		return


	def export(self):
		self.chart_all()

		_, oldFileName = os.path.split(self.filePath)
		oldFileName = os.path.splitext(oldFileName)[0]
		path = os.path.join(self.out_folder_path, oldFileName + '.xlsx')

		with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
			#self.df.to_excel(writer, sheet_name='GC.wke', header=False, index=False)
			self.__writeWorkbook(writer=writer, sheet_name='GC.wke', df=self.original_df)
			self.__writeWorkbook(writer=writer, sheet_name='newGC', df=self.df) #TODO
			self.__writeWorkbook(writer=writer, sheet_name='Log', indexName="Row", df=self._all)
			self.__writeWorkbook(writer=writer, sheet_name='CHECK', indexName="Row", df=self.check)
			self.__writeWorkbook(writer=writer, sheet_name='AA', indexName="Compound", df=self.aa)
			self.__writeWorkbook(writer=writer, sheet_name='IS', indexName="Compound", df=self._is)
			self.__writeWorkbook(writer=writer, sheet_name='NACME', df=self.nacme) #may have to do index=False

		return

	def __writeWorkbook(self, writer, sheet_name, df, indexName=None):
		column_list = [x for x in df.columns]

		offset = 0
		if indexName:
			offset = 1
			column_list.insert(0, indexName)
		
		df.to_excel(writer, sheet_name=sheet_name, startrow=1, startcol=offset, header=False, index=False)

		worksheet = writer.sheets[sheet_name]

		for idx, val in enumerate(column_list):
			worksheet.write(0, idx, val)

		row_list = df.index
		if indexName:
			for idx, val in enumerate(row_list):
				worksheet.write(idx+1, 0, val)

		return


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
		q75, q25 = np.nanpercentile(x, [75 ,25])
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
		if len(data) > 0:
			binsize = self.__binSize(data)
			ax.hist(data, bins=binsize, color='lightblue')
			ax.vlines(np.mean(data), *ax.get_ylim())

			ht = ax.transLimits.inverted().transform((1,1))[1]
			boxWidth = self.__outlier(data)[0] - self.__outlier(data)[1]
			aa_patch = Rectangle((self.__outlier(data)[1],0), width=boxWidth, height=ht, color='lightgreen', alpha=0.5)
			ax.add_patch(aa_patch)    

			ax.vlines(self.__outlier(data), *ax.get_ylim(), linestyle="dotted")

			infoSummery = "mean: %f, sd: %f" % (np.mean(data), np.std(data))
			ax.text(0.0, toff, s=infoSummery, transform=ax.transAxes, fontsize=14)

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

class Corrections:
	#load file where standards are stored
	def __init__(self, data):
		self.standard = data

	#get list of all standards in file
	def get_all_standards(self):
		return self.standard.Standard.unique().tolist()

	#set local version of ISOplot data. 
	def set_ip(self, ISOplot):
		self.data = ISOplot

	#correct based on all internal QC
	def correct_all(self, s_name, sw, exclusions=[], p_C=0.5):
		#set standard
		corr = self.standard[self.standard.Standard == s_name]
		
		allQC = self.data._all[self.data._all['Identifier 1'] == 'AA std']

		for compound in corr.Compound:
			der_sample_mean = self.data.nacme[self.data.nacme.AA == compound].Mean
			
			indicies = self.data.nacme[self.data.nacme.AA == compound].Row.values
			
			before = [allQC[allQC.index < x].iloc[0][compound] for x in indicies]
			after = [allQC[allQC.index > x].iloc[0][compound] for x in indicies]
			
			if sw == 0: der_standard = np.mean(allQC[compound])  
			elif sw == 1: der_standard = before
			elif sw == 2: der_standard = after
			elif sw == 3: der_standard = np.mean([before, after])
			else: return
				
			d13C = corr[corr.Compound == compound].d13C.values[0]

			corrected = (der_sample_mean - der_standard) * p_C + d13C

			current_sample = self.data.nacme[(self.data.nacme.AA == compound) & (self.data.nacme.Sample != 'AA std')].Sample

			for id1 in current_sample:
				#danger: will modefy seed data
				self.data.df.loc[(self.data.df['Identifier 1'] == id1) & (self.data.df.Component == compound), 'd 13C/12C'] = corrected
		return

	"""
	0:sample mean
	1:std before
	2:std after
	3:mean of before/after
	"""
	def correct_individual(self, s_name, sw, exclusions = [], p_C=0.5):
		#set standard
		corr = self.standard[self.standard.Standard == s_name]
		
		allQC = self.data._all[self.data._all['Identifier 1'] == 'AA std']
		
		for idx, row in self.data.nacme.iterrows(): #could reduce this to map...
			samp = np.mean([row.Inj_1, row.Inj_2, row.Inj_3])
			
			before = allQC[allQC.index < row.Row].iloc[0][row.AA]
			after = allQC[allQC.index > row.Row].iloc[0][row.AA]
			bam = np.mean([before, after])
			
			if sw == 0: der_standard = np.mean(allQC[row.AA])
			elif sw == 1: der_standard = before
			elif sw == 2: der_standard = after
			elif sw == 3: der_standard = bam
			
			if len(corr[corr.Compound == row.AA]) > 0:
				d13C = corr[corr.Compound == row.AA].d13C.values[0]

				corrected = (samp - der_standard) * p_C + d13C

				#danger: below will modefy seed data
				self.data.df.loc[(self.data.df['Identifier 1'] == row.Sample) & (self.data.df.Component == row.AA), 'd 13C/12C'] = corrected
		return
		

	def make_chart(self, corr, dpi=22):
		fig = plt.figure(dpi=dpi)

		der_standard = self.data._all[self.data._all['Identifier 1'] == 'AA std'][corr.AA]
		y = self.data._all[self.data._all['Identifier 1'] == 'AA std'].index

		pn = range(len(der_standard))
		fig.scatter(y=der_standard, x=pn)
		fig.scatter(y=[corr.d13C] * len(der_standard), x=pn, marker="_")
		fig.set_xticks(ticks=pn, labels=y)
		fig.ylabel("d13c" + corr.AA)
		fig.xlabel("Row")

		self.chart = fig

		return

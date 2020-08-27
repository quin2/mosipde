"""
Wrapper class for isodat plotting functions
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
from matplotlib import gridspec

import math, os, gc, copy

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
		self.fileName = self.getFileName() #will not work with windows paths on unix, edge case
		#wire in.
		self.include = []
		self.is_c = []
		self.sName = ""
		self.qaName = ""
		self.projName = ""

		self.ext_std = None
		self.std_method = None
		self.corr_info = None

	def getFileName(self):
		_, fileName = os.path.split(self.filePath)
		fileName = os.path.splitext(fileName)[0]
		return fileName

	def preload(self):
		self.df = pd.read_excel(self.filePath)
		
	def load(self):
		self.df = pd.read_excel(self.filePath)
		self.df['corrected'] = False
		self.original_df = copy.deepcopy(self.df)

	def chart_all(self):
		#remove rows from new df
		clean_toinclude = list(self.include) + self.is_c
		self.df = self.df[self.df.Component.isin(clean_toinclude)]

		#load _all chart
		self._all = pd.concat([self.__generateChem(df2) for _, df2 in self.df.groupby(['Identifier 1', 'Identifier 2'])])
		self._all = self._all.sort_index()

		#create aa chart
		temp_aa = self._all[self._all['Identifier 1'] == self.sName]
		temp_aa = temp_aa.drop(['Identifier 1', 'Identifier 2'], axis=1)
		aa_std = temp_aa.apply(np.std, axis=0)
		aa_avg = temp_aa.apply(np.mean, axis=0)
		self.aa = pd.concat([aa_std, aa_avg], axis=1, keys=['SD', 'mean'])
		self.aa = self.aa.dropna()

		#create IS chart
		temp_is = self._all[self.is_c]
		is_std = temp_is.apply(np.std, axis=0)
		is_avg = temp_is.apply(np.mean, axis=0)
		self._is = pd.concat([is_std, is_avg], axis=1, keys=['SD', 'mean'])

		#create NACME
		nacme = []
		out = self._all['Identifier 1'].unique()
		out = np.delete(out, np.argwhere(out==self.sName))

		for sample in out:
			data = self._all[self._all['Identifier 1'] == sample]
			for compound in data.columns[2:]:
				alj = data[compound].values
				if not np.isnan(np.sum(alj)):
					nacme.append([data.index.values[0], sample, compound, alj[0], alj[1], alj[2], 
								  np.mean(alj), np.std(alj), (np.std(alj)/np.sqrt(len(alj)))])

		cNames = ['Row', 'Sample', 'AA', 'Inj_1', 'Inj_2', 'Inj_3', 'Mean', 'SD_inj', 'SE']
		self.nacme = pd.DataFrame(data=nacme, columns=cNames)

		#create ref_meas
		samp = self._all[self._all['Identifier 1'] == self.qaName]
		samp = samp[self.include].T

		new_label = ["Inj_%d" % x for x in range(len(samp.columns))]
		samp = samp.rename(columns=dict(zip(samp.columns, new_label)))

		sd = samp.std(axis=1)
		mean = samp.mean(axis=1)
		samp.insert(loc=0, column='SD', value=sd)
		samp.insert(loc=0, column='Mean_d13C', value=mean)

		samp.index = samp.index.rename('Compound')
		samp = samp.reset_index()

		samp.insert(loc=0, column='Reference', value=self.qaName)
		self.ref_meas = samp

		#create slope array
		self.slopes = pd.DataFrame(data=self.all_slopes(), columns=['compound', 'slope', 'r'])


	def generate_all(self, tw, dpi):
		self.chart_all()

		std_width = 230
		aa_width = 200
		is_width = 250

		w = (tw-100)/dpi
		ht = len(self.include) + len(self.is_c)
		#each plot should be like 5x5, factor was like 2 when there were like 10, 
		self.OVER = self.overview(gW=3, figsize=(w, w*(0.3 * (ht/3))), dpi=dpi)

		self.SLOPE_T = self.all_slopes()

		w = (tw-aa_width)/dpi
		self.AA_H = self.aa_hist(gW=3, figsize=(w,w*(0.3 * (ht/3))), dpi=dpi)
		self.AA_T = self.aa_out()

		w = (tw-std_width)/dpi
		self.STD_H = self.std_hist(gW=2, figsize=(w * 0.75, w*(0.4 * (ht/2))), dpi=dpi)
		self.STD_T = self.std_out()

		w = (tw-is_width)/dpi
		self.IS_H = self.is_hist(figsize=(w, w*(0.55 * (ht/2))), dpi=dpi)
		self.IS_T = self.is_out()

	def overview(self, gW=4, figsize=(20,50), dpi=20):
		fig = plt.figure(figsize=figsize, dpi=dpi)
		r = fig.canvas.get_renderer()

		compounds = self._all.drop(['Identifier 1', 'Identifier 2'], axis=1).columns.values
		compounds = [x for x in compounds if "CO2" not in x]

		gH = math.ceil(len(compounds) / gW)

		gs = gridspec.GridSpec(gH, gW) #32 samples, will have to fix later.
		#gs.update(wspace=0., hspace=0.2)

		for idx, compound in enumerate(compounds):
			i = idx % gW
			j = math.floor(idx / gW)

			gsx = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[j, i])
			ax1 = fig.add_subplot(gsx[0, 0])
			ax2 = fig.add_subplot(gsx[1, 0])

			self.__plotCompoundTimeSeries(self._all[self._all['Identifier 1'] != self.sName], compound, 'Black', 'Red', ax1)
			self.__plotCompoundTimeSeries(self._all[self._all['Identifier 1'] == self.sName], compound, 'Blue', 'Red', ax2)

			self.__displayTrendLine(self._all[self._all['Identifier 1'] == self.sName], compound, 'Blue', ax2)

			bbox = gs[j,i].get_position(figure=fig).get_points()
			xpos = (bbox[0][0] + bbox[1][0]) / 2

			t = fig.text(x=xpos, y=bbox[1][1] + 0.0025, s=str(compound), fontweight='bold', fontsize=15)

			bb = t.get_window_extent(renderer=r)
			tt = fig.transFigure.inverted().transform((bb.width, bb.height))

			t.set_x(t.get_position()[0] - (tt[0] / 2))

			fig.add_subplot(gsx[1, 0])

		return fig

	def all_slopes(self):
		def ctl(compound):
			slope, r, _ = self.__calcTrendLine(self._all[self._all['Identifier 1'] == self.sName], compound)
			return [compound, slope, r]

		#will need to manage this in future for better code reuse
		compounds = self._all.drop(['Identifier 1', 'Identifier 2'], axis=1).columns.values
		compounds = [x for x in compounds if "CO2" not in x]
		
		return [ctl(c) for c in compounds]

	def aa_hist(self, gW=4, figsize=(20,50), dpi=20):
		gH = math.ceil(len(self.aa.index) / gW)

		fig, ax = plt.subplots(gH, gW, figsize=figsize, dpi=dpi)
		plt.subplots_adjust(hspace=0.5)

		for idx, compound in enumerate(self.aa.index):
			i = idx % gW
			j = math.floor(idx / gW)
			
			std_raw = self._all[self._all['Identifier 1'] == self.sName]
			std = std_raw[compound]
			std = std.dropna()
			
			self.__generate_hist(ax[j, i], std, toff=-0.25)
			
			ax[j, i].title.set_text("AA Standard " + compound)
			
		return fig

	def aa_out(self):
		std_raw = self._all[self._all['Identifier 1'] == self.sName]
		return self.__find_outl_array(std_raw, self.aa.index)

	def std_hist(self, gW=4, figsize=(20,50), dpi=20):
		compounds = self._all.drop(['Identifier 1', 'Identifier 2'], axis=1).columns

		samples = self._all['Identifier 1'].unique()
		samples = np.delete(samples, np.where(samples == self.sName))
			
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
		samples = np.delete(samples, np.where(samples == self.sName))
			
		all_sd = [self.__my_std(self._all[self._all['Identifier 1'] == sample]) for sample in samples]
		all_sd = pd.concat(all_sd, keys=samples, axis=1).T
		all_sd = all_sd.drop('Identifier 2', axis=1)

		all_sd = all_sd.round(3)

		final = self.__find_outl_array(all_sd, all_sd.columns)

		#add in row data to premade array
		for idx, outl in enumerate(final):
			outidx = self._all[self._all['Identifier 1'] == outl[1]].index.values
			final[idx] = [outl[0], outl[1], outidx, outl[2]]

		return final

	def is_hist(self, figsize=(20,30), dpi=20):
		#calculate summery for all compounds in internal standard
		std_raw = self._all[self._all['Identifier 1'] == self.sName]
		test_raw = self._all[self._all['Identifier 1'] != self.sName]

		fig, ax = plt.subplots(len(self._is.index), 2, figsize=figsize, dpi=dpi)

			
		for idx, compound in enumerate(self._is.index):    
			std = std_raw[compound]
			test = test_raw[compound]
			
			self.__generate_hist(ax[idx, 0], std, toff=-0.12)
			self.__generate_hist(ax[idx, 1], test, toff=-0.12)
			
			ax[idx, 0].title.set_text("AA Standard " + compound)
			ax[idx, 1].title.set_text("Sample " + compound)
			
		return fig

	def is_out(self):
		def list_add(x, text):
			x.insert(0, text)
			return x

		std_raw = self._all[self._all['Identifier 1'] == self.sName]
		test_raw = self._all[self._all['Identifier 1'] != self.sName]

		standard = self.__find_outl_array(std_raw, self._is.index)
		test = self.__find_outl_array(test_raw, self._is.index)

		standard = [list_add(x, "AA Std") for x in standard]
		test = [list_add(x, "Sample") for x in test]

		return standard + test

	"""
	options is an array of dicts, each dict has the following format:
	{"compound": name_of_analyte, "is": bool, "is_comp": name_of_is_compound, "ro": bool}
	"""
	def aa_correct(self, options):
		def digest(x):
			out = {
				'AA': x['compound'],
				'IS-Corrected?': 'Y' if x['is'] else 'N', 
				'IS': x['is_comp'] if x['is'] else 'N/A',
				'RO-Corrected?': 'Y' if x['ro'] else 'N',
				'RO Slope': self.slopes[self.slopes['compound'] == x['compound']]['slope'].values[0]
				}
			return out
		self.corr_info = pd.DataFrame([digest(x) for x in options])

		for setting in options:
			if setting['is']:
				standardData = self._all[self._all['Identifier 1'] == self.sName]
				IS_residual = standardData[setting['is_comp']] - np.mean(standardData[setting['is_comp']])
				corr = standardData[setting['compound']]-IS_residual
				self.df.loc[(self.df['Identifier 1'] == self.sName) & (self.df.Component == setting['compound']),'d 13C/12C'] = corr.values
				#write changes if both are selected?
				self.chart_all()
			if setting['ro']:
				standardData = self._all[self._all['Identifier 1'] == self.sName]
				slope = self.slopes[self.slopes['compound'] == setting['compound']]['slope'].values[0]
				Runorder = standardData[setting['compound']].index
				corr = standardData[setting['compound']] - Runorder * slope
				self.df.loc[(self.df['Identifier 1'] == self.sName) & (self.df.Component == setting['compound']),'d 13C/12C'] = corr.values
				self.chart_all()

		self.export2()
		return

	def export2(self):
		#self.chart_all() #make this implicit before we save....
		def substance_mean(self, x):
			means = x[self.include].mean()
			#means['Identifier 1'] = x['Identifier 1'].iloc[0]
			return pd.concat([pd.Series({"Identifier 1": x['Identifier 1'].iloc[0]}), means])

		cgc = self.df.rename(index=str, columns={"Identifier 2": "Inj", "Component": "Compound", "d 13C/12C": "d13C"})
		cgc = cgc[["Row", "Identifier 1", "Inj", "Compound", "d13C", "Notes"]]
		cgc.insert(0, 'Sequence', self.fileName)

		#fix NACME if we haven't done any de-deriv yet
		self.nacme['Ext_std'] = self.ext_std
		self.nacme['Std_method'] = self.std_method

		sdd = self.nacme.rename(index=str, columns={"Sample": "Sample_ID", "AA": "Compound","Mean": "Mean_d13c_dd", "SD_inj": "SD"})
		sdd = sdd[['Sample_ID', 'Ext_std', 'Std_method', 'Compound', 'Inj_1', 'Inj_2', 'Inj_3', 'Mean_d13c_dd', 'SD']]

		#remove all cols that wern't corrected!
		notCorrected = self.df[(self.df.corrected == False) & (self.df['Identifier 1'] != self.sName)].Component.unique()
		sdd = sdd.drop(sdd[sdd.Compound.isin(notCorrected)].index)

		sdd_qa = sdd[sdd['Sample_ID'] == self.qaName]#problem is here, we're comparing the wrong things!
		sdd_samp = sdd[sdd['Sample_ID'] != self.qaName]

		#format final_dederiv tab
		injs = self._all['Identifier 1'].unique()

		final_dederiv = pd.DataFrame([substance_mean(self, self._all[self._all['Identifier 1'] == inj]) for inj in injs]) 
		final_dederiv = final_dederiv.drop(final_dederiv[final_dederiv['Identifier 1'] == self.sName].index)   
		final_dederiv = final_dederiv.rename(columns={"Identifier 1": "Sample_ID"})

		final_dederiv.insert(0, 'Project', self.projName)
		final_dederiv.insert(1, 'Sequence', self.fileName)

		fn = self.fileName
		path = os.path.join(self.out_folder_path, fn + '.xlsx')

		with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
			self.__writeWorkbook(writer=writer, sheet_name='Corrected_GC.wke', df=cgc)
			if self.corr_info is not None:
				self.__writeWorkbook(writer=writer, sheet_name='Corrected_GC.wke', df=self.corr_info, indexName=None, startcol=8)
			self.__writeWorkbook(writer=writer, sheet_name='Reference_measured', df=self.ref_meas, indexName=None)
			#only write below if de-deriv has been run...
			if self.ext_std is not None:
				self.__writeWorkbook(writer=writer, sheet_name='QA_de-deriv', df=sdd_qa)
				self.__writeWorkbook(writer=writer, sheet_name='Samples_de-deriv', df=sdd_samp)
				self.__writeWorkbook(writer=writer, sheet_name='Final_de-deriv', df=final_dederiv)

		return

	def __writeWorkbook(self, writer, sheet_name, df, indexName=None, startcol=0, startrow=1):
		column_list = [x for x in df.columns]

		if indexName:
			startcol += 1
			column_list.insert(0, indexName)
		
		df.to_excel(writer, sheet_name=sheet_name, startrow=startrow, startcol=startcol, header=False, index=False)

		worksheet = writer.sheets[sheet_name]

		if indexName:
			startcol -=1

		for idx, val in enumerate(column_list):
			worksheet.write(0, idx+startcol, val)

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

			mymax = np.max(np.histogram(data, bins=binsize)[0])
			ax.vlines(np.mean(data), ymin=ax.get_ylim()[0], ymax=mymax)

			#dead code, replaced by mymax
			#ht = ax.transLimits.inverted().transform((1,1))[1]
			boxWidth = self.__outlier(data)[0] - self.__outlier(data)[1]
			aa_patch = Rectangle((self.__outlier(data)[1],0), width=boxWidth, height=mymax, color='lightgreen')
			ax.add_patch(aa_patch)    

			outl = self.__outlier(data)
			ax.vlines(outl, ymin=0, ymax=mymax, linestyle="dotted")

			ax.hist(data, bins=binsize, color='lightblue', alpha=1.0)

			ax.text(outl[0], mymax-1, s="+2sd", rotation='vertical', fontsize=12)
			ax.text(outl[1], mymax-1, s="-2sd", rotation='vertical', fontsize=12)

			infoSummery = "mean: %0.2f, sd: %0.2f" % (np.mean(data), np.std(data))
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

	def __displayTrendLine(self, data, compound, color, ax3):
		_, r2, clf = self.__calcTrendLine(data, compound)

		if clf is None:
			return

		model_range = np.array(self._all.index).reshape(-1, 1)
		ax3.plot(model_range, clf.predict(model_range), color=color)

		return

	def __calcTrendLine(self, data, compound):
		work = data[compound]
		upper, lower = self.__outlier(work)

		work = work[(work < upper) & (work > lower)]

		if len(work) < 2:
			return None, None, None

		X = np.array(work.index).reshape(-1,1)
		y = np.array(work.values).reshape(-1, 1)

		clf = LinearRegression().fit(X, y)

		r2 = r2_score(y, clf.predict(X))

		return clf.coef_[0][0].round(3), r2.round(3), clf

class Corrections:
	#load file where standards are stored
	def __init__(self, data, pval):
		self.standard = data
		self.pval = pval

		self.methods = ["avg_all", "std_before", "std_after", "mean_before_after"]

	#get list of all standards in file
	def get_all_standards(self):
		return self.standard.Standard.unique().tolist()

	#set local version of ISOplot data. 
	def set_ip(self, ISOplot):
		self.data = ISOplot

	"""
	0:sample mean
	1:std before
	2:std after
	3:mean of before/after
	"""
	#correct based on triplicate mean, then resolve those three values rather than every single value. 
	#I think we want this one...
	def correct_individual(self, s_name, sw, exclusions = []):
		#set standard
		corr = self.standard[self.standard.Standard == s_name]
		
		allQC = self.data._all[self.data._all['Identifier 1'] == self.data.sName]
		
		allCorrected = []
		for idx, row in self.data.nacme.iterrows(): #could reduce this to map...
			samp = [row.Inj_1, row.Inj_2, row.Inj_3]
			
			before = allQC[allQC.index < row.Row].iloc[0][row.AA]
			after = allQC[allQC.index > row.Row].iloc[0][row.AA]
			bam = np.mean([before, after])
			
			if sw == 0: der_standard = np.mean(allQC[row.AA])
			elif sw == 1: der_standard = before
			elif sw == 2: der_standard = after
			elif sw == 3: der_standard = bam
			
			if len(corr[corr.Compound == row.AA]) > 0 and len(self.pval[(self.pval.Compound == row.AA)].p.values) > 0:
				d13C = corr[corr.Compound == row.AA].d13C.values[0]
				p_C = self.pval[(self.pval.Compound == row.AA)].p.values[0]

				corrected = ((samp - der_standard) / p_C) + d13C

				#danger: below will modefy seed data
				self.data.df.loc[(self.data.df.Component == row.AA) & (self.data.df['Identifier 1'] == row.Sample), 'd 13C/12C'] = corrected
				self.data.df.loc[(self.data.df.Component == row.AA) & (self.data.df['Identifier 1'] == row.Sample), 'corrected'] = ([True] * len(corrected)) 

		self.data.ext_std = s_name
		self.data.std_method = self.methods[sw]

		return
		

	def make_chart(self, corr, dpi=22):
		fig = plt.figure(dpi=dpi)

		der_standard = self.data._all[self.data._all['Identifier 1'] == self.data.sName][corr.AA]
		y = self.data._all[self.data._all['Identifier 1'] == self.data.sName].index

		pn = range(len(der_standard))
		fig.scatter(y=der_standard, x=pn)
		fig.scatter(y=[corr.d13C] * len(der_standard), x=pn, marker="_")
		fig.set_xticks(ticks=pn, labels=y)
		fig.ylabel("d13c" + corr.AA)
		fig.xlabel("Row")

		self.chart = fig

		return

from mido import MidiFile
import os
for filename in os.listdir('.'):
	if filename.endswith('.midi'):
		try:
			print(filename)
			mid=MidiFile(filename).length
			if 30<mid<450:
				os.rename(filename, str('good/%s' % filename))
		except:
			os.rename(filename, str('bad/%s' % filename))

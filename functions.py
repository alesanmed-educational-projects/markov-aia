def obscode_to_bitarray(code):
	if code==0:
		# --
		return [0, 0, 0, 0];
	elif code==1:
		# N
		return [1, 0, 0, 0];
	elif code==2:
		# E
		return [0, 1, 0, 0];
	elif code==3:
		# S
		return [0, 0, 1, 0];
	elif code==4:
		# O
		return [0, 0, 0, 1];
	elif code==5:
		# NE
		return [1, 1, 0, 0];
	elif code==6:
		# NS
		return [1, 0, 1, 0];
	elif code==7:
		# NO
		return [1, 0, 0, 1];
	elif code==8:
		# ES
		return [0, 1, 1, 0];
	elif code==9:
		# EO
		return [0, 1, 0, 1];
	elif code==10:
		# SO
		return [0, 0, 1, 1];
	elif code==11:
		# NES
		return [1, 1, 1, 0];
	elif code==12:
		# NEO
		return [1, 1, 0, 1];
	elif code==13:
		# NSO
		return [1, 0, 1, 1];
	elif code==14:
		# ESO
		return [0, 1, 1, 1];
	elif code==15:
		# NESO
		return [1, 1, 1, 1];
	else:
		return None

def manhattan_distance(point1, point2):
	distance = abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

	return distance
import numpy as np
'''
dccol : 1-8
dcpad : 1-10
mcecol: 0,1
mcerow: 0-32

darks:
mce col, mce row, postion, postion
+1, 10,	beneath frame,	left
+1, 9,	beneath frame,	left
+1, 8,	light,		left
+1, 7,	light,		left
+0, 7,	beneath frame,	right
+0, 8,	beneath frame,	right
+0, 9,	light,		right
+0, 10,	light,		right
'''

w,h = 10,8
dc2mcecol = [[[0 for i in range(w)] for j in range(h)] for k in range(12)]
dc2mcerow = [[[0 for i in range(w)] for j in range(h)] for k in range(12)]

for k in range(12):
	dc2mcecol[k][0] = [k*2,k*2,k*2,k*2,k*2,k*2,k*2,k*2,k*2,k*2]
	dc2mcecol[k][1] = [k*2,k*2,k*2,k*2,k*2,k*2,k*2,k*2,k*2,k*2]
	dc2mcecol[k][2] = [k*2,k*2,k*2,k*2,   -1,   -1,   -1,   -1,   -1,   -1]
	dc2mcecol[k][3] = [   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1]
	dc2mcecol[k][4] = [  k*2,  k*2,  k*2,  k*2,  k*2,  k*2,  k*2+1,  k*2+1,  k*2+1,  k*2+1]
	dc2mcecol[k][5] = [  k*2+1,  k*2+1,  k*2+1,  k*2+1,   -1,   -1,   -1,   -1,   -1,   -1]
	dc2mcecol[k][6] = [  k*2+1,  k*2+1,  k*2+1,  k*2+1,  k*2+1,  k*2+1,  k*2+1,  k*2+1,  k*2+1,  k*2+1]
	dc2mcecol[k][7] = [  k*2+1,  k*2+1,  k*2+1,  k*2+1,  k*2+1,  k*2+1,  k*2+1,  k*2+1,  k*2+1,  k*2+1]
	
	dc2mcerow[k][0] = [ 0,32,31,30,29,28,27,26,25,24] 
	dc2mcerow[k][1] = [22,21,20,19,18,17,16,15,14,13]
	dc2mcerow[k][2] = [11,10, 9, 8,-1,-1,-1,-1,-1,-1]
	dc2mcerow[k][3] = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
	dc2mcerow[k][4] = [ 7, 6, 5, 4, 3, 2, 4, 5, 6, 7]
	dc2mcerow[k][5] = [ 8, 9,10,11,-1,-1,-1,-1,-1,-1]
	dc2mcerow[k][6] = [13,14,15,16,17,18,19,20,21,22]
	dc2mcerow[k][7] = [24,25,26,27,28,29,30,31,32, 0]

#dc2mcecol = np.array(dc2mcecol)
#dc2mcerow = np.array(dc2mcerow)

mce2module = [[-1 for i in range(33)] for j in range(24)]
mce2dccol =  [[-1 for i in range(33)] for j in range(24)]
mce2dcpad =  [[-1 for i in range(33)] for j in range(24)]

for k in range(12):
	for ii in range(h):
		for jj in range(w):
			if dc2mcecol[k][ii][jj]<0 or dc2mcerow[k][ii][jj]<0:
					continue
			mce2dccol[dc2mcecol[k][ii][jj]][dc2mcerow[k][ii][jj]] = ii+1
			mce2dcpad[dc2mcecol[k][ii][jj]][dc2mcerow[k][ii][jj]] = jj+1
			mce2module[dc2mcecol[k][ii][jj]][dc2mcerow[k][ii][jj]] = k

det2dccol = [[[-1 for i in range(4)] for j in range(4)] for p in range(2)]
det2dcpad = [[[-1 for i in range(4)] for j in range(4)] for p in range(2)]

det2dccol[0][0] = [ 1, 1, 1, 1]
det2dccol[0][1] = [ 2, 2, 2, 2]
det2dccol[0][2] = [ 7, 7, 7, 7]
det2dccol[0][3] = [ 8, 8, 8, 8]

det2dccol[1][0] = [ 1, 1, 1, 1]
det2dccol[1][1] = [ 2, 2, 2, 2]
det2dccol[1][2] = [ 7, 7, 7, 7]
det2dccol[1][3] = [ 8, 8, 8, 8]

det2dcpad[0][0] = [ 4, 6, 2, 8]
det2dcpad[0][1] = [ 4, 6, 2, 8]
det2dcpad[0][2] = [ 4, 6, 2, 8]
det2dcpad[0][3] = [ 4, 6, 2, 8]

det2dcpad[1][0] = [ 3, 5, 1, 7]
det2dcpad[1][1] = [ 3, 5, 1, 7]
det2dcpad[1][2] = [ 3, 5, 1, 7]
det2dcpad[1][3] = [ 3, 5, 1, 7]

dc2detpol = [[-1 for i in range(w)] for j in range(h)]
dc2detcol = [[-1 for i in range(w)] for j in range(h)]
dc2detrow = [[-1 for i in range(w)] for j in range(h)]

for pp in range(2):
	for cc in range(4):
		for rr in range(4):
			if det2dccol[pp][cc][rr]<0 or det2dcpad[pp][cc][rr]<0:
				continue
			dc2detpol[det2dccol[pp][cc][rr]-1][det2dcpad[pp][cc][rr]-1] = pp
			dc2detcol[det2dccol[pp][cc][rr]-1][det2dcpad[pp][cc][rr]-1] = cc+1
			dc2detrow[det2dccol[pp][cc][rr]-1][det2dcpad[pp][cc][rr]-1] = rr+1
dc2detpol[4][0] = 99
dc2detpol[4][1] = 99
dc2detpol[4][2] = 99
dc2detpol[4][3] = 99
dc2detpol[4][4] = 99
dc2detpol[4][5] = 99
dc2detpol[4][6] = 99
dc2detpol[4][7] = 99

def dc2mce(iM,dccol,dcpad):
	mcecol = dc2mcecol[iM][dccol-1][dcpad-1]
	mcerow = dc2mcerow[iM][dccol-1][dcpad-1]
	return mcecol,mcerow

def mce2dc(mcecol,mcerow):
	dccol = mce2dccol[mcecol][mcerow]
	dcpad = mce2dcpad[mcecol][mcerow]
	iM = mce2module[mcecol][mcerow]
	return iM,dccol,dcpad

def dc2det(dccol,dcpad):
	detcol = dc2detcol[dccol-1][dcpad-1]
	detrow = dc2detrow[dccol-1][dcpad-1]
	if dc2detpol[dccol-1][dcpad-1]==1:
		detpol = 'B'
	elif dc2detpol[dccol-1][dcpad-1]==0:
		detpol = 'A'
	elif dc2detpol[dccol-1][dcpad-1]==99:
		detpol = 'D'
	else:
		detpol = 'nan'
	return detcol,detrow,detpol

def det2dc(detcol,detrow,detpol):
	if detpol == 'A':
		dccol = det2dccol[0][detcol-1][detrow-1]
		dcpad = det2dcpad[0][detcol-1][detrow-1] 
	elif detpol == 'B':
                dccol = det2dccol[1][detcol-1][detrow-1]
                dcpad = det2dcpad[1][detcol-1][detrow-1]
	else:
		dccol = -1
		dcpad = -1
	return dccol,dcpad

def det2mce(iM, detcol,detrow,detpol):
	dccol,dcpad = det2dc(detcol,detrow,detpol)
	if dccol<0 or dcpad<0:
			return -1,-1
	mcecol,mcerow = dc2mce(iM,dccol,dcpad)
	return mcecol,mcerow

def mce2det(mcecol,mcerow):
	iM,dccol,dcpad = mce2dc(mcecol,mcerow)
	if dccol<0 or dcpad<0:
			return -1,-1,-1,'nan'
	detcol,detrow,detpol = dc2det(dccol,dcpad)
	return iM,detcol,detrow,detpol





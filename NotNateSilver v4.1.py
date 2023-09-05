import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import random

#modifiers
epv = 6.0
epvm = 2.5
margbase = np.array([0.,0.,0.,0.,0.,0.,0.,0.],dtype='float')
margvars = np.array([1.5,2.0,1.5,0.5,2.0,0.5,1.2,0.8],dtype='float')
modbase = np.array([82.0,28.0,5.0,10.0,18.9],dtype=float)
modvar = np.array([0.03,0.05,0.02,0.02,0.005],dtype=float)
modprop = np.array([0.128,0.208,0.700,0.34])
reglines = []
houseleans = []
with open('regvals.csv','r') as regvals:
    reglines1 = regvals.readlines()
with open('houseleans.csv','r') as hsl:
    hslines = hsl.readlines()
for w in hslines:
    z = w.split(',')
    houseleans.append(z)
for k in reglines1:
    h = k.split(',')
    reglines.append(h)
statevals = []
pvvals = []
evvals = []
elasvals = []
regvals = []
modvals = []
senvals = []
distnames = []
leanvals = []
hregs = []
hstates = []

for i in range(len(reglines)):
    statevals.append(reglines[i][0])
    pvvals.append(int(float(reglines[i][1])))
    evvals.append(int(float(reglines[i][2])))
    elasvals.append(float(reglines[i][3]))
    regvalsa = []
    for j in range(4,len(reglines[0])-6):
        regvalsa.append(float(reglines[i][j]))
    modvals.append([float(reglines[i][-6]),float(reglines[i][-5]),float(reglines[i][-4])-11,float(reglines[i][-3])-1,float(reglines[i][-2])-1])
    regvals.append(regvalsa)
    senvals.append(float(reglines[i][-1]))
    
for a in range(len(houseleans)):
    b = houseleans[a]
    distnames.append(b[0])
    leanvals.append(float(b[1]))
    hregs.append([float(b[2]),float(b[3]),float(b[4]),float(b[5]),float(b[6]),float(b[7]),float(b[8]),float(b[9])])
    hstates.append(int(b[-1]))
    
def ElecModel():
    pv = random.gauss(epv,epvm)
    margrng = np.random.standard_normal(len(margbase))
    margvals = margbase + margvars*margrng

    rvmat = np.array(regvals,dtype='float')
    regfinv = np.matmul(rvmat,margvals)

    modrng = np.random.standard_normal(len(modbase))
    modv1 = (modrng*np.array(modvar))
    modv1 *= np.array(modbase)
    modvot = np.zeros((len(elasvals),len(modv1)),dtype=float)

    for m in range(len(modv1)):
        for j in range(len(elasvals)):
            modvot[j][m] += modv1[m]*elasvals[j]
        
    modsum = np.sum(modvot,axis=1)
    pvsum = np.array(pvvals) + np.array(elasvals)*((pv-epv)*np.ones(len(elasvals)) + 1.5*np.random.standard_normal(len(elasvals)))

    tpv = pvsum + modsum + regfinv
    
    hrfinv = np.matmul(hregs,margvals)
    distmv = np.zeros(435,dtype=float)
    for i in range(len(hregs)):
        distmv[i] += np.sum(hrfinv[i])
        distst = hstates[i]
        distmv[i] += modsum[distst]
    hrn = np.random.normal(pv-1,1,size=435)
    hadd = hrn + distmv
    hvotes = hadd + np.array(leanvals)

    voteslist = [[],[],[],[]]
    demevtot = 0
    for i in range(len(statevals)):
        voteslist[0].append(statevals[i])
        voteslist[1].append(tpv[i])
        if tpv[i] < 0.0:
            voteslist[2].append(0)
            voteslist[3].append(evvals[i])
        else:
            voteslist[3].append(0)
            voteslist[2].append(evvals[i])
    
    senselec = []
    for i in range(len(statevals)):
        if int(senvals[i]) == 100:
            senselec.append(-1)
        else:
            newtot = tpv[i] + random.gauss(senvals[i],1.5)
            if newtot < 0:
                senselec.append(0)
            else:
                senselec.append(1)
    return voteslist, pv, senselec, hvotes

def ReturnStats(n=100):
    demevlist = []
    pvlist = []
    statecounts = np.zeros(56,dtype=int)
    statelist = []
    maxminlist = np.zeros((2,56),dtype=int)
    maxv = 269
    minv = 269
    senct = np.zeros(56,dtype=float)
    sental = 0
    nsens = np.zeros(34,dtype=float)
    nhwins = 0
    hsseattots = np.zeros(435,dtype=int)
    seathdist = np.zeros(130,dtype=int)
    for i in range(n):
        numsw = 0
        housrt = 0
        vl, pv1, senslist, housevt = ElecModel()
        for c in range(435):
            if housevt[c] > 0:
                housrt += 1
                hsseattots[c] += 1
        if housrt > 217:
            nhwins += 1
        seathdist[housrt-170] += 1 
        nevs = sum(vl[2])
        demevlist.append(nevs) 
        pvlist.append(pv1)
        scl = np.zeros(56,dtype=int)
        for j in range(56):
            if int(vl[3][j]) == 0:
                scl[j] += 1
        if i == 0:
            statelist = vl[0]
            for t in range(56):
                if senslist[t] == -1:
                   senct[t] += -1*n
        for u in range(56):
            if senslist[u] == 1:
                senct[u] += 1
                numsw += 1
        statecounts += scl
        
        if numsw == 22:
            nsens[22] += 1
            if nevs > 269:
                sental += 1
        elif numsw>23:
            nsens[numsw] += 1
            sental += 1
        else:
            nsens[numsw] += 1

        if nevs > maxv:
            maxminlist[0] += -1*np.copy(maxminlist[0])
            maxv = nevs
            maxminlist[0] += scl
        elif nevs < minv:
            minv = nevs
            maxminlist[1] += -1*np.copy(maxminlist[1])
            maxminlist[1] += scl
    devar = np.array(demevlist,dtype=int)
    avgev = np.average(devar)
    stdev = np.std(devar)
    senct *= 1/n
    sental *= 1/n
    nhwins *= 1/n
    confintev = [int(avgev + 3*stdev), int(avgev + 2*stdev), int(avgev + stdev), avgev, int(avgev - stdev), int(avgev - 2*stdev), int(avgev - 3*stdev)]
    return confintev, statelist, statecounts, demevlist, pvlist, maxminlist, senct, sental, nsens, nhwins, hsseattots, seathdist

numruns = 10000
ci, sl, stct, evs, pvs, mami, senl, sental, numsens, hwinrat, hsseatprob, seathdistr = ReturnStats(numruns)

#probability of Dem win
wintot = 0
for s in range(numruns):
    if evs[s] > 269:
        wintot += 1
wintot *= 100/numruns
print("There is a "+str(wintot)+"% chance of Democrats winning the White House")

#for getting 95% CIs
print("Sigma CI of EVs (3-2-1-Mean-1-2-3): "+str(ci))

##for making representative maps
print(np.percentile(np.array(evs),np.array([5.,15.,25.,35.,45.,55.,65.,75.,85.,95.])))


#for getting State Flip Probabilities
for p in range(56):
    print(sl[p]+' has a '+str(100*stct[p]/numruns)+'% chance of going blue')
    if int(senl[p]) == -1: 
        pass
    else:
        print(senl[p]*100)



pvsm = np.array(pvs).reshape((-1,1))
evsm = np.array(evs)
linreg = LinearRegression().fit(pvsm,evsm)
r_sq = linreg.score(pvsm,evsm)
yint = linreg.intercept_
mslo = linreg.coef_[0]
print('\n')
print('Regression of PV-EV')
print('R^2: '+str(r_sq))
print('y-int: '+str(yint))
print('slope: '+str(mslo))
print('Biden loss PV: '+str((269-yint)/mslo))

print("Senate Kept By Dems How Often? "+str(sental*100)+"% of the time")
#for making regression plot:
xpts = np.linspace(-5,15,51)
yintv = np.ones(51)*yint
ypts = xpts*mslo + yintv

print("House Gained By Dems How Often? "+str(100*hwinrat)+"% of the time")
print(hsseatprob)

#plt.scatter(pvsm[::25],evsm[::25],marker='x')
#plt.plot(xpts,ypts,color='r')
#plt.show()

#plt.bar(np.arange(0,34,1),numsens/100,width=0.8)
#plt.show()

plt.bar(np.arange(170,300,1),seathdistr/100,width=0.8)
plt.show()

#for r in range(56):
    #print(str(mami[0][r])+' max for '+sl[r])
    #print(str(mami[1][r])+' min for '+sl[r])
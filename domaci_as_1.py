
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

# postavljamo generalne uslove za sve grafike koje cemo praviti, za velicinu grafika i font teksa na njima 
plt.rcParams['figure.figsize'] = (7, 5)
plt.rcParams['font.size'] = 11

# Seed stavljamo da bismo uvek dobili iste random brojeve u delu sa centralnom granicnom teoremom 
rng = np.random.default_rng(42)

# ucitavamo fajlove neophodne za rad 
# inc i az su dati u radijanima pa ih prebacujemo u stepene pomocu np.degrees
data = np.load('sunspot.npz')

B = data['B']                       # jacina magnetnog polja u gausima [G]
theta = np.degrees(data['inc'])     # inklinacija prebacena iz radijana u stepene 
phi = np.degrees(data['az'])        # azimut prebacen iz radijana u stepene 

print('Oblik niza B:', B.shape)
print('Oblik niza theta:', theta.shape)
print('Oblik niza phi:', phi.shape)

# Podaci su dati kao 2d niz, a nama za histograme i dalju statistiku trebaju u 1d nizu; pomocu ravel() fukcije ih prebacujemo 
B_1d = B.ravel()
theta_1d = theta.ravel()
phi_1d = phi.ravel()

# Izbacujemo NaN i beskonacne vrednosti, ako ih ima; vracamo samo boolean true vrednosti niza 
B_1d = B_1d[np.isfinite(B_1d)]
theta_1d = theta_1d[np.isfinite(theta_1d)]
phi_1d = phi_1d[np.isfinite(phi_1d)]

## ZADATAK 1 - prikaz mapa i histograma za B, theta i phi ##################################################################

# na osama nisu fizicke koordinate nego indeksi piksela 
fig, axs = plt.subplots(3, 1, figsize=(16, 4))

im0 = axs[0].imshow(B, cmap = 'viridis', origin='lower', aspect='auto') # preko imshow prikazujemo niz kao sliku gde je svaki element niza jedan piksel; nemastamo poecetak koor sistema u donji levi ugao i dozvoljavamo da se automatski rastegne slika u prostoru a da se pri tome ne deformise
axs[0].set_title('Mapa: B [G]')
axs[0].set_xlabel('x piksel')
axs[0].set_ylabel('y piksel')
plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
# isto radimo i za ostale mape 
im1 = axs[1].imshow(theta,cmap='plasma', origin='lower', aspect='auto')
axs[1].set_title('Mapa: theta [deg]')
axs[1].set_xlabel('x piksel')
axs[1].set_ylabel('y piksel')
plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

im2 = axs[2].imshow(phi,cmap='inferno', origin='lower', aspect='auto')
axs[2].set_title('Mapa: phi [deg]')
axs[2].set_xlabel('x piksel')
axs[2].set_ylabel('y piksel')
plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

plt.tight_layout() 
plt.show()

# crtamo histograme 
plt.figure()
plt.hist(B_1d, bins='auto', density=True, alpha=0.6, edgecolor='none', color='green')
plt.title('Histogram: B [G]')
plt.xlabel('B [G]')
plt.ylabel('gustina verovatnoce')
plt.show()

plt.figure()
plt.hist(theta_1d, bins='auto', density=True, alpha=0.6, edgecolor='none', color='blue')
plt.title('Histogram: theta [deg]')
plt.xlabel('theta [deg]')
plt.ylabel('gustina verovatnoce')
plt.show()

plt.figure()
plt.hist(phi_1d, bins='auto', density=True, alpha=0.6, edgecolor='none', color='red')
plt.title('Histogram: phi [deg]')
plt.xlabel('phi [deg]')
plt.ylabel('gustina verovatnoce')
plt.show()

## ZADATAK 2 ################################################################################

EPS = 1e-300

# Ova funkcija od bilo koja 3 broja napravi 3 pozitivne tezine koje  se sabiraju
def softmax_3(a, b, c):
    najveci = max(a, b, c)
    ea = np.exp(a - najveci)
    eb = np.exp(b - najveci)
    ec = np.exp(c - najveci)
    zbir = ea + eb + ec
    return ea / zbir, eb / zbir, ec / zbir

def softmax_2(a, b):
    najveci = max(a, b)
    ea = np.exp(a - najveci)
    eb = np.exp(b - najveci)
    zbir = ea + eb
    return ea / zbir, eb / zbir


B_min = np.min(B_1d)
B_max = np.max(B_1d)
B_span = B_max - B_min
B_mean_start = np.mean(B_1d)
B_std_start = np.std(B_1d)
B_exp_scale_start = np.mean(B_1d - B_min)

print('Pocetne vrednosti za B fit:')
print('B_min =', B_min)
print('B_max =', B_max)
print('mean =', B_mean_start)
print('std =', B_std_start)
print()

# minimizujemo funkciju za B 
def neg_log_likelihood_B(p):
    # p[0], p[1], p[2] sluze samo da napravimo tezine
    w_norm, w_exp, w_uni = softmax_3(p[0], p[1], p[2])

    mu = p[3]
    sigma = np.exp(p[4])
    exp_scale = np.exp(p[5])

    normalni_deo = stats.norm.pdf(B_1d, loc=mu, scale=sigma)
    eksponencijalni_deo = stats.expon.pdf(B_1d, loc=B_min, scale=exp_scale)
    uniformni_deo = stats.uniform.pdf(B_1d, loc=B_min, scale=B_span)

    pdf = w_norm * normalni_deo + w_exp * eksponencijalni_deo + w_uni * uniformni_deo
    nll = -np.sum(np.log(pdf + EPS))
    return nll

# Pocetne vrednosti za minimizaciju
p0_B = np.array([0.0, 0.0, -1.0, B_mean_start, np.log(B_std_start), np.log(B_exp_scale_start)])
rez_B = minimize(neg_log_likelihood_B, p0_B, method='Nelder-Mead', options={'maxiter': 20000})

wB_norm, wB_exp, wB_uni = softmax_3(rez_B.x[0], rez_B.x[1], rez_B.x[2])
mu_B = rez_B.x[3]
sigma_B = np.exp(rez_B.x[4])
scale_B_exp = np.exp(rez_B.x[5])

print('Da li je minimizacija uspela:', rez_B.success)
print('Tezina normalne =', wB_norm)
print('Tezina eksponencijalne =', wB_exp)
print('Tezina uniformne =', wB_uni)
print('mu_B =', mu_B)
print('sigma_B =', sigma_B)
print('scale eksponencijalne =', scale_B_exp)
print()

# minimizujemo funkciju za theta 
theta_min = np.min(theta_1d)
theta_mean_start = np.mean(theta_1d)
theta_std_start = np.std(theta_1d)
theta_exp_scale_start = np.mean(theta_1d - theta_min)

print('Pocetne vrednosti za theta fit:')
print('theta_min =', theta_min)
print('mean =', theta_mean_start)
print('std =', theta_std_start)
print()

def neg_log_likelihood_theta(p):
    w_norm, w_exp = softmax_2(p[0], p[1])

    mu = p[2]
    sigma = np.exp(p[3])
    exp_scale = np.exp(p[4])

    normalni_deo = stats.norm.pdf(theta_1d, loc=mu, scale=sigma)
    eksponencijalni_deo = stats.expon.pdf(theta_1d, loc=theta_min, scale=exp_scale)

    pdf = w_norm * normalni_deo + w_exp * eksponencijalni_deo
    nll = -np.sum(np.log(pdf + EPS))
    return nll

p0_theta = np.array([0.0, 0.0, theta_mean_start, np.log(theta_std_start), np.log(theta_exp_scale_start)])
rez_theta = minimize(neg_log_likelihood_theta, p0_theta, method='Nelder-Mead', options={'maxiter': 20000})

wtheta_norm, wtheta_exp = softmax_2(rez_theta.x[0], rez_theta.x[1])
mu_theta = rez_theta.x[2]
sigma_theta = np.exp(rez_theta.x[3])
scale_theta_exp = np.exp(rez_theta.x[4])

print('Da li je minimizacija uspela:', rez_theta.success)
print('Tezina normalne =', wtheta_norm)
print('Tezina eksponencijalne =', wtheta_exp)
print('mu_theta =', mu_theta)
print('sigma_theta =', sigma_theta)
print('scale eksponencijalne =', scale_theta_exp)
print()

# fitujemo laplasovu raspodelu za phi 
phi_loc, phi_scale = stats.laplace.fit(phi_1d)

print('loc =', phi_loc)
print('scale =', phi_scale)
print()

# Crtamo fitove preko histograma 

x_B = np.linspace(B_min, B_max, 1000)
B_pdf_norm = stats.norm.pdf(x_B, loc=mu_B, scale=sigma_B)
B_pdf_exp = stats.expon.pdf(x_B, loc=B_min, scale=scale_B_exp)
B_pdf_uni = stats.uniform.pdf(x_B, loc=B_min, scale=B_span)
B_pdf_total = wB_norm * B_pdf_norm + wB_exp * B_pdf_exp + wB_uni * B_pdf_uni

plt.figure()
plt.hist(B_1d, bins='auto', density=True, alpha=0.6, edgecolor='none', label='podaci', color='green')
plt.plot(x_B, B_pdf_total, lw=2, label='fit = normalna + eksponencijalna + uniformna', color='black')
plt.title('Histogram i fit: B [G]')
plt.xlabel('B [G]')
plt.ylabel('gustina verovatnoce')
plt.legend()
plt.show()

x_theta = np.linspace(np.min(theta_1d), np.max(theta_1d), 1000)
theta_pdf_norm = stats.norm.pdf(x_theta, loc=mu_theta, scale=sigma_theta)
theta_pdf_exp = stats.expon.pdf(x_theta, loc=theta_min, scale=scale_theta_exp)
theta_pdf_total = wtheta_norm * theta_pdf_norm + wtheta_exp * theta_pdf_exp

plt.figure()
plt.hist(theta_1d, bins='auto', density=True, alpha=0.6, edgecolor='none', label='podaci', color='blue')
plt.plot(x_theta, theta_pdf_total, lw=2, label='fit = normalna + eksponencijalna', color='black')
plt.title('Histogram i fit: theta [deg]')
plt.xlabel('theta [deg]')
plt.ylabel('gustina verovatnoce')
plt.legend()
plt.show()

x_phi = np.linspace(np.min(phi_1d), np.max(phi_1d), 1000)
phi_pdf = stats.laplace.pdf(x_phi, loc=phi_loc, scale=phi_scale)

plt.figure()
plt.hist(phi_1d, bins='auto', density=True, alpha=0.6, edgecolor='none',color='red', label='podaci')
plt.plot(x_phi, phi_pdf, lw=2, label='fit = Laplas', color='black')
plt.title('Histogram i fit: phi [deg]')
plt.xlabel('phi [deg]')
plt.ylabel('gustina verovatnoce')
plt.legend()
plt.show()

## ZADATAK 3 #################################################################################################
# statisticke vrednosti 
# Za B
B_mean = np.mean(B_1d)
B_std = np.std(B_1d, ddof=1)
B_median = np.median(B_1d)
B_q25 = np.percentile(B_1d, 25)
B_q75 = np.percentile(B_1d, 75)
B_sigma_G = B_q75 - B_q25
B_skew = stats.skew(B_1d, bias=False)
B_kurt = stats.kurtosis(B_1d, fisher=True, bias=False)

# Za theta
theta_mean = np.mean(theta_1d)
theta_std = np.std(theta_1d, ddof=1)
theta_median = np.median(theta_1d)
theta_q25 = np.percentile(theta_1d, 25)
theta_q75 = np.percentile(theta_1d, 75)
theta_sigma_G = theta_q75 - theta_q25
theta_skew = stats.skew(theta_1d, bias=False)
theta_kurt = stats.kurtosis(theta_1d, fisher=True, bias=False)

# Za phi
phi_mean = np.mean(phi_1d)
phi_std = np.std(phi_1d, ddof=1)
phi_median = np.median(phi_1d)
phi_q25 = np.percentile(phi_1d, 25)
phi_q75 = np.percentile(phi_1d, 75)
phi_sigma_G = phi_q75 - phi_q25
phi_skew = stats.skew(phi_1d, bias=False)
phi_kurt = stats.kurtosis(phi_1d, fisher=True, bias=False)

# Pravimo tabelu sa vrednostima svih parametara 
tabela = pd.DataFrame({
    'parametar': ['B [G]', 'theta [deg]', 'phi [deg]'],
    'x_bar': [B_mean, theta_mean, phi_mean],
    's': [B_std, theta_std, phi_std],
    'medijana': [B_median, theta_median, phi_median],
    'sigma_G=q75-q25': [B_sigma_G, theta_sigma_G, phi_sigma_G],
    'asimetricnost': [B_skew, theta_skew, phi_skew],
    'kurtosis_excess': [B_kurt, theta_kurt, phi_kurt]
})

print(tabela)
print()

# Crtamo srednju vrednost i medijanu preko histograma
plt.figure()
plt.hist(B_1d, bins='auto', density=True, alpha=0.6, edgecolor='none', color='green')
plt.axvline(B_mean, lw=2, linestyle='-', label='srednja', color='black')
plt.axvline(B_median, lw=2, linestyle='--', label='medijana', color='orange')
plt.title('B: srednja vrednost i medijana')
plt.xlabel('B [G]')
plt.ylabel('gustina verovatnoce')
plt.legend()
plt.show()

plt.figure()
plt.hist(theta_1d, bins='auto', density=True, alpha=0.6, edgecolor='none', color='blue')
plt.axvline(theta_mean, lw=2, linestyle='-', label='srednja', color='black')
plt.axvline(theta_median, lw=2, linestyle='--', label='medijana', color='orange')
plt.title('theta: srednja vrednost i medijana')
plt.xlabel('theta [deg]')
plt.ylabel('gustina verovatnoce')
plt.legend()
plt.show()

plt.figure()
plt.hist(phi_1d, bins='auto', density=True, alpha=0.6, edgecolor='none', color='red')
plt.axvline(phi_mean, lw=2, linestyle='-', label='srednja', color='black')
plt.axvline(phi_median, lw=2, linestyle='--', label='medijana', color='orange')
plt.title('phi: srednja vrednost i medijana')
plt.xlabel('phi [deg]')
plt.ylabel('gustina verovatnoce')
plt.legend()
plt.show()

print('Deskriptivne statistike zavise od oblika raspodele podataka.')
print('Na histogramima se vidi da raspodele B, theta i phi nisu idealno normalne posebno kod magnetnog polja B postoji izraženiji rep ka većim vrednostima, dok su theta i phi asimetrične. Zbog toga srednja vrednost i standardna devijacija nisu potpuno robusne mere, jer su osetljive na ekstremne vrednosti i repove raspodele.')
print('Medijana i sigma_G=q75-q25 daju pouzdaniji opis centralnog dela podataka, jer manje zavise od udaljenih vrednosti.')
print('Asimetričnost i kurtosis dodatno potvrđuju da raspodele odstupaju od normalne, pa ih treba posmatrati zajedno sa histogramima.')

## ZADATAK 4 #####################################################################################################################

# da bismo dosli do raspodele srednjih vrednosti sto slicnojoj normalnoj raspodeli moramo da biramo sto veci uzorak; kako bismo videli kako se to menja uzimamo random izabrane m uzorke za racunanje 
# na m = 30 CLT vec picinje lepo da se vidi
m_values = [2, 5, 10, 30, 100]
n_rep = 5000

# za B 
print('CLT za B')

fig, axs = plt.subplots(1, len(m_values), figsize=(18, 4))

x_norm = np.linspace(-4, 4, 500)

for i in range(len(m_values)):
    m = m_values[i]

    uzorci = rng.choice(B_1d, size=(n_rep, m), replace=True)
    srednje = np.mean(uzorci, axis=1)

    # standardizujemo srednje vrednosti da ih poredimo sa N(0,1)
    z = (srednje - B_mean) / (B_std / np.sqrt(m))

    print('m =', m, '  mean(z) =', np.mean(z), '  std(z) =', np.std(z, ddof=1))

    axs[i].hist(z, bins=50, density=True,
                alpha=0.6,
                edgecolor='none',
                color='green',
                label='srednje vrednosti')

    axs[i].plot(x_norm, stats.norm.pdf(x_norm),
                lw=2,
                label='N(0,1)',
                color='black')

    axs[i].set_title('m = ' + str(m))
    axs[i].set_xlabel('z')
    axs[i].set_ylabel('gustina')

fig.suptitle('CLT za B')
plt.tight_layout()
plt.show()

# za theta

print('CLT za theta')

fig, axs = plt.subplots(1, len(m_values), figsize=(18, 4))

x_norm = np.linspace(-4, 4, 500)

for i in range(len(m_values)):
    m = m_values[i]

    uzorci = rng.choice(theta_1d, size=(n_rep, m), replace=True)
    srednje = np.mean(uzorci, axis=1)

    # standardizujemo srednje vrednosti da ih poredimo sa N(0,1)
    z = (srednje - theta_mean) / (theta_std / np.sqrt(m))

    print('m =', m, '  mean(z) =', np.mean(z), '  std(z) =', np.std(z, ddof=1))

    axs[i].hist(z, bins=50, density=True,
                alpha=0.6,
                edgecolor='none',
                color='blue',
                label='srednje vrednosti')

    axs[i].plot(x_norm, stats.norm.pdf(x_norm),
                lw=2,
                label='N(0,1)',
                color='black')

    axs[i].set_title('m = ' + str(m))
    axs[i].set_xlabel('z')
    axs[i].set_ylabel('gustina')

fig.suptitle('CLT za theta')
plt.tight_layout()
plt.show()

# za phi

print('CLT za phi')

fig, axs = plt.subplots(1, len(m_values), figsize=(18, 4))

x_norm = np.linspace(-4, 4, 500)

for i in range(len(m_values)):
    m = m_values[i]

    uzorci = rng.choice(phi_1d, size=(n_rep, m), replace=True)
    srednje = np.mean(uzorci, axis=1)

    # standardizujemo srednje vrednosti da ih poredimo sa N(0,1)
    z = (srednje - phi_mean) / (phi_std / np.sqrt(m))

    print('m =', m, '  mean(z) =', np.mean(z), '  std(z) =', np.std(z, ddof=1))

    axs[i].hist(z, bins=50, density=True,
                alpha=0.6,
                edgecolor='none',
                color='red',
                label='srednje vrednosti')

    axs[i].plot(x_norm, stats.norm.pdf(x_norm),
                lw=2,
                label='N(0,1)',
                color='black')

    axs[i].set_title('m = ' + str(m))
    axs[i].set_xlabel('z')
    axs[i].set_ylabel('gustina')

fig.suptitle('CLT za phi')
plt.tight_layout()
plt.show()


print('Komentar za CLT:')
print('Sa porastom veličine uzorka m histogrami srednjih vrednosti postaju sve sličniji normalnoj raspodeli N(0,1), što potvrđuje centralnu graničnu teoremu.') 
print('Za male vrednosti m, posebno za m=2 i m=5, raspodele još uvek zadržavaju asimetriju i oblik originalnih podataka. Kako m raste na 30 i 100, histogrami postaju sve simetričniji i bolje prate teorijsku normalnu raspodelu.') 
print('Ovo se vidi kod sva tri parametra B, theta i phi, iako je kod magnetnog polja B odstupanje izraženije za male uzorke zbog dužeg repa raspodele.') 
print('Takođe, izračunate vrednosti mean(z) ostaju blizu nule, a std(z) blizu jedinice, što dodatno potvrđuje očekivanja centralne granične teoreme.')
print()

## ZADATAK 5 ############################################################################################################

mu0 = 4 * np.pi * 1e-7 # H/m
B_T = B_1d * 1e-4
P = B_T**2 / (2 * mu0)

B_T_mean = np.mean(B_T)
B_T_std = np.std(B_T, ddof=1)
P_mean = np.mean(P)

# Izvod od P po B je B/mu0
# U formuli za propagaciju uzimamo izvod u srednjoj vrednosti B
dP_dB = B_T_mean / mu0
sP_propagacija = B_T_std * abs(dP_dB)

# Direktna standardna devijacija pritiska
sP_direktno = np.std(P, ddof=1)

aps_odstupanje = abs(sP_propagacija - sP_direktno)
rel_odstupanje = aps_odstupanje / sP_direktno * 100

print('Magnetni pritisak:')
print('<B> =', B_T_mean, 'T')
print('sB =', B_T_std, 'T')
print('<P> =', P_mean, 'Pa')
print('sP iz propagacije =', sP_propagacija, 'Pa')
print('sP direktno iz P =', sP_direktno, 'Pa')
print('apsolutno odstupanje =', aps_odstupanje, 'Pa')
print('relativno odstupanje =', rel_odstupanje, '%')
print()

print('Komentar za propagaciju greske:')
print('Propagacija greske koristi linearnu aproksimaciju funkcije P(B).')
print('Posto je ovde P proporcionalno B^2, funkcija nije linearna.')
print('Zbog toga sP iz propagacije ne mora da bude isto kao sP dobijeno direktno iz svih P vrednosti.')
print('Ako su B vrednosti dosta rasute, odstupanje moze biti vece.')




